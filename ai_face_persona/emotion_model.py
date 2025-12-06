
import numpy as np
import cv2
from typing import List, Tuple
from collections import deque, defaultdict
import threading
import time
import os


# Attempt to import Deepface; if not available, set df to None
try:
    from deepface import DeepFace as df
except ImportError:
    df = None

# Attempt to import ONNX Runtime; if not available, set ort to None
try:
    import onnxruntime as ort
except ImportError:
    ort = None

# Attempt to import transformers pipeline; if not available, set hf_pipeline to None
try:
    from transformers import pipeline as hf_pipeline
except ImportError:
    hf_pipeline = None


# Emotionmodel class definition
class EmotionModel:
    """ This list of emotions the model can recognize"""
    PERSONA_MAP = {
        'joy': 'AI Dreamer',
        'surprise': 'Curious Synth',
        'anger': 'Chrome Rebel',
        'sadness': 'Neon Loner',
        'confused': 'Quantum Puzzler',
        'happy': 'Sunset Coder',
        'excited': 'Pulse Rider',
        'fear': 'Circuit Warden',
        'disgust': 'Acid Critic',
        'neutral': 'Calm Sentinel'
    }

    # 1. Constructor
    def __init__(self, model_name="joeddav/distilbert-base-uncased-go-emotions-student", mode="image"):
        # HF text classifier
        self.model_name = model_name
        self.mode = mode
        self.classifier = None  # lazy-loaded
        self._load_lock = threading.Lock()

        # Smoothed display state
        self.last_label = None
        self.display_label = None
        self.display_alpha = 1.0
        self.last_change_time = time.time()

        # Smoothing history for recent predictions
        self.recent = deque(maxlen=8)
        self.recent_decay = 0.85

        # Bounding box smoothing (lerp)
        self.bbox_lerp = 0.22

        # DL backend ("deepface" or "onnx")
        self.dl_backend = "deepface"

        # Cached ONNX session
        self._onnx_sess = None

        # ONNX model URL (FER+ small model)
        self._onnx_model_url = 'https://raw.githubusercontent.com/onnx/models/main/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx'

    # 2. Load HF text classifier
    def load(self):
        """Lazy-load HuggingFace text classifier if in text/hybrid mode."""
        if hf_pipeline is None:
            raise RuntimeError("transformers library not available")
        with self._load_lock:
            if self.classifier is None:
                self.classifier = hf_pipeline("text-classification", model=self.model_name, return_all_scores=True)

    # 3. Convert landmarks to descriptive text
    def _landmarks_to_text(self, lm, shape: Tuple[int,int]) -> str:
        """Convert face landmarks to descriptive text for HF text model."""
        if not lm:
            return "neutral face"

        h, w = shape

        # Mouth
        left_mouth = lm[61] if len(lm) > 61 else lm[-1]
        right_mouth = lm[291] if len(lm) > 291 else lm[-1]
        top_lip = lm[13] if len(lm) > 13 else lm[-1]
        bottom_lip = lm[14] if len(lm) > 14 else lm[-1]

        # Eyes
        left_eye_top = lm[159] if len(lm) > 159 else lm[-1]
        left_eye_bottom = lm[145] if len(lm) > 145 else lm[-1]
        right_eye_top = lm[386] if len(lm) > 386 else lm[-1]
        right_eye_bottom = lm[374] if len(lm) > 374 else lm[-1]

        # Compute normalized metrics
        mouth_width = np.hypot(right_mouth[0]-left_mouth[0], right_mouth[1]-left_mouth[1]) / (h*0.3 + 1e-6)
        mouth_height = np.hypot(top_lip[0]-bottom_lip[0], top_lip[1]-bottom_lip[1]) / (h*0.05 + 1e-6)
        eye_height = ((abs(left_eye_top[1]-left_eye_bottom[1]) + abs(right_eye_top[1]-right_eye_bottom[1])) / 2) / (h*0.03 + 1e-6)

        parts = []
        if mouth_width > 0.28 and mouth_height < 0.06:
            parts.append("smiling")
        elif mouth_height > 0.06:
            parts.append("mouth open")
        else:
            parts.append("neutral mouth")

        if eye_height > 1.2:
            parts.append("eyes wide")
        elif eye_height < 0.6:
            parts.append("eyes squint")
        else:
            parts.append("eyes normal")

        return ", ".join(parts)

    # 4. Predict emotion from landmarks
    def _predict_from_landmarks(self, lm, shape: Tuple[int,int]):
        """Lightweight heuristic prediction from landmarks only."""
        if not lm:
            return "neutral", 0.5

        h, w = shape

        def landmark(i):
            return lm[i] if i < len(lm) else lm[-1]

        def distance(a, b):
            return np.hypot(a[0]-b[0], a[1]-b[1])

        # Mouth
        left_mouth = landmark(61)
        right_mouth = landmark(291)
        top_lip = landmark(13)
        bottom_lip = landmark(14)
        mouth_width = distance(left_mouth, right_mouth)/h
        mouth_height = distance(top_lip, bottom_lip)/h

        # Eyes
        left_eye_height = abs(landmark(159)[1]-landmark(145)[1])/h
        right_eye_height = abs(landmark(386)[1]-landmark(374)[1])/h
        eye_height = (left_eye_height + right_eye_height)/2

        # Heuristic scores
        smile_score = mouth_width - 0.8*mouth_height
        open_mouth_score = mouth_height*3
        surprise_score = open_mouth_score + eye_height*2
        squint_score = 1 - eye_height

        # Label assignment
        if smile_score > 0.035 and eye_height < 0.02:
            return "happy", 0.6
        if surprise_score > 0.12:
            return "surprise", 0.7
        if squint_score > 0.04:
            return "anger", 0.6
        if mouth_height > 0.06:
            return "sadness", 0.55
        return "neutral", 0.5

    # 5. Predict emotion using DeepFace
    def _predict_from_deepface(self, frame):
        """Predict emotion using DeepFace."""
        if df is None:
            return "neutral", 0.0
        try:
            out = df.analyze(frame, actions=["emotion"], enforce_detection=False)
            emo = out[0]["dominant_emotion"] if isinstance(out, list) else out.get("dominant_emotion")
            score = max(out[0]["emotion"].values())/100 if isinstance(out, list) else max(out["emotion"].values())/100
            mapping = {
                "happy":"joy", "sad":"sadness","angry":"anger","surprise":"surprise",
                "neutral":"neutral","disgust":"disgust","fear":"fear"
            }
            return mapping.get(emo.lower(), "neutral"), score
        except Exception:
            return "neutral", 0.0

    # 6. Predict emotion using ONNX
    def _predict_from_onnx(self, frame):
        """Predict using cached ONNX FER+ model."""
        if ort is None or self._onnx_sess is None:
            return "neutral", 0.0
        try:
            x = frame.astype('float32')[None, None, :, :]
            out = self._onnx_sess.run(None, {"input": x})[0][0]
            idx = int(np.argmax(out))
            conf = float(np.max(out))
            labels = ["neutral","happiness","surprise","sadness","anger","disgust","fear","contempt"]
            lab = labels[idx] if idx < len(labels) else "neutral"
            mapping = {"happiness":"happy","neutral":"neutral","surprise":"surprise","sadness":"sadness",
                       "anger":"anger","disgust":"disgust","fear":"confused","contempt":"neutral"}
            return mapping.get(lab,"neutral"), conf
        except Exception:
            return "neutral", 0.0

    # 7. Ensure ONNX model exists locally
    def _ensure_onnx_model(self, model_path):
        """Download ONNX model if not present and initialize session."""
        if self._onnx_sess is not None:
            return True
        if ort is None:
            return False
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        import requests
        r = requests.get(self._onnx_model_url, stream=True)
        with open(model_path, 'wb') as f:
            for chunk in r.iter_content(8192):
                if chunk:
                    f.write(chunk)
        self._onnx_sess = ort.InferenceSession(model_path)
        return True

    # 8. Unified predict method
    def predict(self, lm, shape, frame=None):
        """Unified predict method: landmarks → DL/text → smoothing → fade."""
        # Landmarks heuristic
        label, conf = self._predict_from_landmarks(lm, shape)

        # DL mode
        if self.mode == "dl" and frame is not None:
            dl_label, dl_conf, _, _ = self.predict_dl(frame)
            if dl_label:
                label, conf = dl_label, dl_conf

        # HF text model
        if self.classifier:
            desc = self._landmarks_to_text(lm, shape)
            try:
                scores = self.classifier(desc)[0]
                top = max(scores, key=lambda x: x['score'])
                if top['score'] > conf:
                    label, conf = top['label'], float(top['score'])
            except Exception:
                pass

        # Smoothing
        self.recent.append((label, conf))
        agg = defaultdict(float)
        w = 1.0
        for L, C in reversed(self.recent):
            agg[L] += C*w
            w *= self.recent_decay
        final_label = max(agg.items(), key=lambda kv: kv[1])[0]
        final_conf = agg[final_label]/max(sum(agg.values()),1e-6)

        # Fade animation
        if final_label != self.last_label:
            self.last_label = final_label
            self.last_change_time = time.time()
            self.display_alpha = 0.0
        self.display_alpha = min(1.0,(time.time()-self.last_change_time)/0.25)

        persona = self.PERSONA_MAP.get(final_label.lower(), self.PERSONA_MAP["neutral"])
        return final_label, final_conf, persona, self.display_alpha

    # 9. DL wrapper
    def predict_dl(self, frame):
        """Convenience DL wrapper: ONNX → DeepFace fallback"""
        lab, conf = self._predict_from_onnx(frame)
        if lab == "neutral":
            lab, conf = self._predict_from_deepface(frame)
        persona = self.PERSONA_MAP.get(lab.lower(), self.PERSONA_MAP["neutral"])
        if lab != self.last_label:
            self.last_label = lab
            self.last_change_time = time.time()
            self.display_alpha = 0.0
        self.display_alpha = min(1.0,(time.time()-self.last_change_time)/0.25)
        return lab, conf, persona, self.display_alpha

# Main test
if __name__ == "__main__":
    em = EmotionModel()
    try:
        em.load()
        print("Loaded model (may download first time).")
    except Exception as e:
        print("Could not load transformer model:", e)
