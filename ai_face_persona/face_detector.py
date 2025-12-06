import cv2
import numpy as np
from typing import Tuple, List, Optional

# Try using mediapipe, if not found disable detection silently
try:
    import mediapipe as mp
    mp_mesh = mp.solutions.face_mesh
except ImportError:
    mp_mesh = None


class FaceLocator:

    def __init__(
        self,
        fine_points: bool = True,
        face_limit: int = 1,
        detect_conf: float = 0.5,
        track_conf: float = 0.5,
    ):
        self.max_allowed = face_limit
        self._active = mp_mesh is not None

        if not self._active:
            self._processor = None
            print("[FaceLocator] Mediapipe unavailable. Skipping detection.")
            return

        self._processor = mp_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=face_limit,
            refine_landmarks=fine_points,
            min_detection_confidence=detect_conf,
            min_tracking_confidence=track_conf,
        )

    def scan(
        self, bgr_img: np.ndarray
    ) -> Tuple[Optional[Tuple[int, int, int, int]], List[Tuple[int, int]]]:


        if bgr_img is None or bgr_img.size == 0 or not self._active:
            return None, []

        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        height, width = bgr_img.shape[:2]

        try:
            output = self._processor.process(rgb)
        except Exception:
            return None, []

        if not output.multi_face_landmarks:
            return None, []

        # Use first detected face only
        mesh = output.multi_face_landmarks[0]

        landmarks: List[Tuple[int, int]] = []
        xs, ys = [], []

        for item in mesh.landmark:
            cx = int(item.x * width)
            cy = int(item.y * height)
            landmarks.append((cx, cy))
            xs.append(cx)
            ys.append(cy)

        if not xs or not ys:
            return None, landmarks

        pad = 8
        x1 = max(min(xs) - pad, 0)
        y1 = max(min(ys) - pad, 0)
        x2 = min(max(xs) + pad, width)
        y2 = min(max(ys) + pad, height)

        bbox = (x1, y1, x2 - x1, y2 - y1)
        return bbox, landmarks

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

        for px, py in dots[::12]:  # draw sparse points (avoid clutter)
            cv2.circle(preview, (px, py), 1, (255, 0, 0), -1)

        return preview


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
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Quit by 'q'
            break

    cam.release()
    cv2.destroyAllWindows()

