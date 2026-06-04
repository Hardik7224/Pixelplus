# emotion_model.py
import numpy as np
import cv2
from typing import List, Tuple, Optional
from collections import deque, defaultdict
import threading
import time
import os


# ── Optional imports ───────────────────────────────────────────────────────────
try:
    from deepface import DeepFace as df
    print("[EmotionModel] DeepFace available.")
except ImportError:
    df = None
    print("[EmotionModel] DeepFace not installed – will use heuristics/ONNX only.")

try:
    import onnxruntime as ort
    print("[EmotionModel] ONNX Runtime available.")
except ImportError:
    ort = None

try:
    from transformers import pipeline as hf_pipeline
    print("[EmotionModel] Transformers available.")
except ImportError:
    hf_pipeline = None


# ── EmotionModel ───────────────────────────────────────────────────────────────
class EmotionModel:

    PERSONA_MAP = {
        "joy":      "AI Dreamer",
        "surprise": "Curious Synth",
        "anger":    "Chrome Rebel",
        "sadness":  "Neon Loner",
        "confused": "Quantum Puzzler",
        "happy":    "Sunset Coder",
        "excited":  "Pulse Rider",
        "fear":     "Circuit Warden",
        "disgust":  "Acid Critic",
        "neutral":  "Calm Sentinel",
    }

    # Unified label map from DeepFace / ONNX → our keys
    _LABEL_MAP = {
        "happy":    "happy",
        "happiness":"happy",
        "sad":      "sadness",
        "sadness":  "sadness",
        "angry":    "anger",
        "anger":    "anger",
        "surprise": "surprise",
        "neutral":  "neutral",
        "disgust":  "disgust",
        "fear":     "fear",
        "contempt": "neutral",
    }

    # ── constructor ─────────────────────────────────────────────────────────────
    def __init__(
        self,
        model_name: str = "joeddav/distilbert-base-uncased-go-emotions-student",
        mode: str = "image",
    ):
        self.model_name = model_name
        self.mode = mode
        self.classifier = None
        self._load_lock = threading.Lock()

        # Fade animation state
        self.last_label: Optional[str] = None
        self.display_alpha = 1.0
        self.last_change_time = time.time()

        # Temporal smoothing
        self.recent: deque = deque(maxlen=8)
        self.recent_decay = 0.85
        self.bbox_lerp = 0.22

        # DL backend preference
        self.dl_backend = "deepface"   # "deepface" | "onnx"

        # ONNX session (lazy)
        self._onnx_sess = None
        self._onnx_model_url = (
            "https://github.com/onnx/models/raw/main/vision/body_analysis/"
            "emotion_ferplus/model/emotion-ferplus-8.onnx"
        )

        # DeepFace call throttle: run at most once per N seconds per face
        self._last_df_time = 0.0
        self._df_interval = 0.25          # seconds between DeepFace calls
        self._df_cache: Tuple[str, float] = ("neutral", 0.0)

    # ── public: load HF classifier ──────────────────────────────────────────────
    def load(self):
        if hf_pipeline is None:
            raise RuntimeError("transformers not installed")
        with self._load_lock:
            if self.classifier is None:
                self.classifier = hf_pipeline(
                    "text-classification",
                    model=self.model_name,
                    return_all_scores=True,
                )

    # ── landmark → text ─────────────────────────────────────────────────────────
    def _landmarks_to_text(self, lm, shape: Tuple[int, int]) -> str:
        if not lm:
            return "neutral face"
        h, _ = shape

        def lmk(i):
            return lm[i] if i < len(lm) else lm[-1]

        mouth_w = np.hypot(lmk(291)[0] - lmk(61)[0], lmk(291)[1] - lmk(61)[1]) / (h * 0.3 + 1e-6)
        mouth_h = np.hypot(lmk(13)[0] - lmk(14)[0],  lmk(13)[1] - lmk(14)[1])  / (h * 0.05 + 1e-6)
        eye_h   = (abs(lmk(159)[1] - lmk(145)[1]) + abs(lmk(386)[1] - lmk(374)[1])) / 2 / (h * 0.03 + 1e-6)

        parts = []
        if mouth_w > 0.28 and mouth_h < 0.06:
            parts.append("smiling")
        elif mouth_h > 0.06:
            parts.append("mouth open")
        else:
            parts.append("neutral mouth")

        if eye_h > 1.2:
            parts.append("eyes wide")
        elif eye_h < 0.6:
            parts.append("eyes squint")
        else:
            parts.append("eyes normal")

        return ", ".join(parts)

    # ── landmark heuristic ───────────────────────────────────────────────────────
    def _predict_from_landmarks(self, lm, shape: Tuple[int, int]) -> Tuple[str, float]:
        if not lm:
            return "neutral", 0.3

        h, _ = shape

        def lmk(i):
            return lm[i] if i < len(lm) else lm[-1]

        def dist(a, b):
            return np.hypot(a[0] - b[0], a[1] - b[1])

        mouth_w  = dist(lmk(61),  lmk(291)) / (h + 1e-6)
        mouth_h  = dist(lmk(13),  lmk(14))  / (h + 1e-6)
        eye_h    = (abs(lmk(159)[1] - lmk(145)[1]) + abs(lmk(386)[1] - lmk(374)[1])) / 2 / (h + 1e-6)

        smile  = mouth_w - 0.8 * mouth_h
        open_m = mouth_h * 3
        surp   = open_m + eye_h * 2
        sqint  = 1 - eye_h * 40   # normalised

        if smile > 0.035 and eye_h * 40 < 0.02:
            return "happy", 0.60
        if surp > 0.12:
            return "surprise", 0.70
        if sqint > 0.04:
            return "anger", 0.60
        if mouth_h > 0.06:
            return "sadness", 0.55
        return "neutral", 0.45

    # ── DeepFace ─────────────────────────────────────────────────────────────────
    def _predict_from_deepface(self, frame: np.ndarray) -> Tuple[str, float]:
        """Run DeepFace; result is cached between calls to limit latency."""
        if df is None:
            return "neutral", 0.0

        now = time.time()
        if now - self._last_df_time < self._df_interval:
            return self._df_cache          # return cached result

        try:
            # DeepFace expects BGR; enforce_detection=False avoids crash if face not found
            result = df.analyze(frame, actions=["emotion"], enforce_detection=False, silent=True)
            entry  = result[0] if isinstance(result, list) else result
            raw    = entry.get("dominant_emotion", "neutral")
            scores = entry.get("emotion", {})
            conf   = max(scores.values()) / 100.0 if scores else 0.5
            label  = self._LABEL_MAP.get(raw.lower(), "neutral")
            self._df_cache = (label, conf)
            self._last_df_time = now
            return label, conf
        except Exception as e:
            print(f"[EmotionModel] DeepFace error: {e}")
            return "neutral", 0.0

    # ── ONNX FER+ ────────────────────────────────────────────────────────────────
    def _predict_from_onnx(self, gray_frame: np.ndarray) -> Tuple[str, float]:
        if ort is None or self._onnx_sess is None:
            return "neutral", 0.0
        try:
            resized = cv2.resize(gray_frame, (64, 64)).astype("float32")
            x       = resized[None, None, :, :]          # (1,1,64,64)
            out     = self._onnx_sess.run(None, {"input": x})[0][0]
            idx     = int(np.argmax(out))
            conf    = float(np.max(out))
            labels  = ["neutral","happiness","surprise","sadness","anger","disgust","fear","contempt"]
            raw     = labels[idx] if idx < len(labels) else "neutral"
            label   = self._LABEL_MAP.get(raw, "neutral")
            return label, conf
        except Exception as e:
            print(f"[EmotionModel] ONNX error: {e}")
            return "neutral", 0.0

    def _ensure_onnx_model(self, model_path: str) -> bool:
        if self._onnx_sess is not None:
            return True
        if ort is None:
            return False
        try:
            os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
            if not os.path.exists(model_path):
                print("[EmotionModel] Downloading ONNX FER+ model …")
                import urllib.request
                tmp = model_path + ".tmp"
                urllib.request.urlretrieve(self._onnx_model_url, tmp)
                os.replace(tmp, model_path)
                print("[EmotionModel] ONNX model downloaded.")
            self._onnx_sess = ort.InferenceSession(model_path)
            print("[EmotionModel] ONNX session ready.")
            return True
        except Exception as e:
            print(f"[EmotionModel] ONNX model load failed: {e}")
            return False

    # ── smoothing helper ─────────────────────────────────────────────────────────
    def _smooth_and_fade(self, label: str, conf: float):
        """Temporal smoothing + fade-in alpha."""
        self.recent.append((label, conf))
        agg: dict = defaultdict(float)
        w = 1.0
        for L, C in reversed(self.recent):
            agg[L] += C * w
            w *= self.recent_decay
        total = max(sum(agg.values()), 1e-6)
        final = max(agg.items(), key=lambda kv: kv[1])[0]
        final_conf = agg[final] / total

        if final != self.last_label:
            self.last_label = final
            self.last_change_time = time.time()
            self.display_alpha = 0.0
        self.display_alpha = min(1.0, (time.time() - self.last_change_time) / 0.25)
        persona = self.PERSONA_MAP.get(final, self.PERSONA_MAP["neutral"])
        return final, final_conf, persona, self.display_alpha

    # ── main predict (heuristic + optional HF text) ──────────────────────────────
    def predict(self, lm, shape: Tuple[int, int], frame: Optional[np.ndarray] = None):
        """
        Primary path: landmarks → heuristic (→ HF text if loaded) → smooth.
        If a frame is supplied and dl_backend is active, DL result blends in.
        """
        label, conf = self._predict_from_landmarks(lm, shape)

        # DL blend when frame available
        if frame is not None:
            dl_label, dl_conf = self._dl_predict(frame)
            if dl_conf > conf:
                label, conf = dl_label, dl_conf

        # HF text model (optional)
        if self.classifier and lm:
            desc = self._landmarks_to_text(lm, shape)
            try:
                scores = self.classifier(desc)[0]
                top    = max(scores, key=lambda x: x["score"])
                if float(top["score"]) > conf:
                    label, conf = self._LABEL_MAP.get(top["label"].lower(), top["label"]), float(top["score"])
            except Exception:
                pass

        return self._smooth_and_fade(label, conf)

    # ── DL-only predict (used by main.py when use_dl=True) ───────────────────────
    def predict_dl(self, frame: np.ndarray):
        """
        DL path: ONNX (grayscale crop) → DeepFace fallback.
        Called by main.py with a face-crop grey image.
        """
        # Ensure ONNX session
        if self._onnx_sess is None:
            self._ensure_onnx_model(os.path.join("models", "emotion-ferplus-8.onnx"))

        # Try ONNX first
        label, conf = self._predict_from_onnx(frame)

        # Fall back to DeepFace on low confidence
        if conf < 0.4 or label == "neutral":
            # DeepFace wants a colour frame; convert if needed
            colour = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame
            df_label, df_conf = self._predict_from_deepface(colour)
            if df_conf > conf:
                label, conf = df_label, df_conf

        return self._smooth_and_fade(label, conf)

    # ── internal DL dispatcher ───────────────────────────────────────────────────
    def _dl_predict(self, frame: np.ndarray) -> Tuple[str, float]:
        """Route to ONNX or DeepFace based on dl_backend setting."""
        if self.dl_backend == "onnx":
            if self._onnx_sess is None:
                self._ensure_onnx_model(os.path.join("models", "emotion-ferplus-8.onnx"))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            return self._predict_from_onnx(gray)
        # Default: DeepFace
        return self._predict_from_deepface(frame)


# ── standalone test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    em = EmotionModel()
    try:
        em.load()
        print("HF model loaded.")
    except Exception as e:
        print("HF model unavailable:", e)