# face_locator.py
import cv2
import numpy as np
from typing import Tuple, List, Optional

# ── MediaPipe import (tries every known path across versions) ──────────────────
mp_mesh = None
_mp_imported = False

try:
    import mediapipe as mp
    # Try legacy path first (mediapipe < 0.10)
    try:
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            mp_mesh = mp.solutions.face_mesh
            _mp_imported = True
    except Exception:
        pass

    # Try new path (mediapipe >= 0.10)
    if not _mp_imported:
        try:
            from mediapipe.python.solutions import face_mesh as _fm
            mp_mesh = _fm
            _mp_imported = True
        except Exception:
            pass

    # Last resort: direct attribute walk
    if not _mp_imported:
        try:
            import mediapipe.python.solutions.face_mesh as _fm2
            mp_mesh = _fm2
            _mp_imported = True
        except Exception:
            pass

except ImportError:
    pass

if not _mp_imported:
    print("[FaceLocator] MediaPipe unavailable – falling back to OpenCV Haar cascade.")


# ── OpenCV Haar cascade fallback ───────────────────────────────────────────────
def _load_haar():
    """Load OpenCV's built-in frontal-face cascade."""
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    clf = cv2.CascadeClassifier(path)
    if clf.empty():
        print("[FaceLocator] Haar cascade also unavailable!")
        return None
    print("[FaceLocator] Using OpenCV Haar cascade for face detection.")
    return clf


class FaceLocator:
    """
    Detects one face and returns its bounding-box + 468 mesh landmarks (if MediaPipe
    is available) or just a bounding-box with synthesised landmark stubs (Haar fallback).
    """

    def __init__(
        self,
        fine_points: bool = True,
        face_limit: int = 1,
        detect_conf: float = 0.5,
        track_conf: float = 0.5,
    ):
        self.max_allowed = face_limit
        self._use_mp = _mp_imported and mp_mesh is not None
        self._processor = None
        self._haar = None

        if self._use_mp:
            try:
                self._processor = mp_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=face_limit,
                    refine_landmarks=fine_points,
                    min_detection_confidence=detect_conf,
                    min_tracking_confidence=track_conf,
                )
                print("[FaceLocator] MediaPipe FaceMesh ready.")
            except Exception as e:
                print(f"[FaceLocator] FaceMesh init failed ({e}); falling back to Haar.")
                self._use_mp = False

        if not self._use_mp:
            self._haar = _load_haar()

    # ── public API ──────────────────────────────────────────────────────────────

    def scan(
        self, bgr_img: np.ndarray
    ) -> Tuple[Optional[Tuple[int, int, int, int]], List[Tuple[int, int]]]:
        """Return (bbox, landmarks).  bbox = (x, y, w, h) or None."""
        if bgr_img is None or bgr_img.size == 0:
            return None, []

        if self._use_mp and self._processor is not None:
            return self._scan_mediapipe(bgr_img)

        if self._haar is not None:
            return self._scan_haar(bgr_img)

        return None, []

    # ── MediaPipe path ──────────────────────────────────────────────────────────

    def _scan_mediapipe(self, bgr_img):
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        height, width = bgr_img.shape[:2]

        try:
            output = self._processor.process(rgb)
        except Exception as e:
            print(f"[FaceLocator] MediaPipe process error: {e}")
            return None, []

        if not output.multi_face_landmarks:
            return None, []

        mesh = output.multi_face_landmarks[0]
        landmarks: List[Tuple[int, int]] = []
        xs, ys = [], []

        for item in mesh.landmark:
            cx = int(item.x * width)
            cy = int(item.y * height)
            landmarks.append((cx, cy))
            xs.append(cx)
            ys.append(cy)

        if not xs:
            return None, landmarks

        pad = 8
        x1 = max(min(xs) - pad, 0)
        y1 = max(min(ys) - pad, 0)
        x2 = min(max(xs) + pad, width)
        y2 = min(max(ys) + pad, height)
        return (x1, y1, x2 - x1, y2 - y1), landmarks

    # ── Haar fallback path ──────────────────────────────────────────────────────

    def _scan_haar(self, bgr_img):
        """Detect face with Haar cascade; return bbox + empty landmark list."""
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = self._haar.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        if len(faces) == 0:
            return None, []

        # Pick largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        return (int(x), int(y), int(w), int(h)), []

    # ── debug paint ─────────────────────────────────────────────────────────────

    @staticmethod
    def paint(
        frame: np.ndarray,
        box: Optional[Tuple[int, int, int, int]],
        dots: List[Tuple[int, int]],
    ) -> np.ndarray:
        preview = frame.copy()
        if box:
            bx, by, bw, bh = box
            cv2.rectangle(preview, (bx, by), (bx + bw, by + bh), (0, 255, 255), 2)
        for px, py in dots[::12]:
            cv2.circle(preview, (px, py), 1, (255, 0, 0), -1)
        return preview


# ── quick standalone test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    cam = cv2.VideoCapture(0)
    watcher = FaceLocator()

    while True:
        ok, snap = cam.read()
        if not ok:
            break
        area, lmks = watcher.scan(snap)
        display = FaceLocator.paint(snap, area, lmks)
        cv2.imshow("Face Locator Preview", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()