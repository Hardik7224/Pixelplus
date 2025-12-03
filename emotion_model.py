# emotion_model.py

import numpy as np
import cv2
from typing import List, Tuple
from deepface import DeepFace
from collections import deque, defaultdict
import threading
import time
import os

# Global variables for model initialization
DeepFace = None
onnxruntime = None
pipeline = None

# Attempt to import ONNX Runtime; if not available, set ort to None
try:
    import onnxruntime as ort
except ImportError:
    ort = None


# Emotionmodel class definition
class EmotionModel:
    # This list of emotions the model can recognize
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
    
    # Constructor for the EmotionModel class
    def __init__(self, model_name="joeddav/distilbert-base-uncased-go-emotions",
                 mode="image"):

        # public configuration
        self.model_name = model_name
        self.mode = mode
        self.dl_backend = "deepface"

        # runtime state
        self._classifier = None
        self._onnx_sess = None
        self._load_lock = threading.Lock()
        
        # ONNX model URL
        self._onnx_url = (
            "https://raw.githubusercontent.com/onnx/models/"
            "main/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx")

        # smoothing
        self.recent = deque(maxlen=8)
        self.recent_decay = 0.85

        # visual label
        self.last_label = None
        self.display_label = None
        self.display_alpha = 1.0
        self.last_change_time = time.time()

        # HUD(Heads-Up DIsplay) smoothing
        self.bbox_lerp = 0.22