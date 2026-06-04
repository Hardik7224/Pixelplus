# main.py
import time
import os
import cv2
import numpy as np

try:
    import winsound
except ImportError:
    winsound = None

from face_locator import FaceLocator
from emotion_model import EmotionModel
import overlay_utils as ou


# ── helpers ────────────────────────────────────────────────────────────────────

def play_shutter_sound():
    if winsound is None:
        return
    try:
        winsound.Beep(1000, 80)
        winsound.Beep(1400, 60)
    except Exception:
        pass


def crop_to_bbox(img: np.ndarray, bbox, pad: int = 8) -> np.ndarray:
    x, y, w, h = bbox
    ih, iw = img.shape[:2]
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(iw, x + w + pad)
    y2 = min(ih, y + h + pad)
    return img[y1:y2, x1:x2]


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam. Connect a camera and retry.")
        return

    locator = FaceLocator()

    # Use "image" mode: DeepFace runs on every frame that has a detected face.
    em = EmotionModel(mode="image")
    em.dl_backend = "deepface"   # primary DL engine; ONNX is secondary

    try:
        em.load()
        print("HuggingFace text model loaded.")
    except Exception as e:
        print(f"HF model unavailable → heuristic/DeepFace only: {e}")

    fps_smooth  = 30.0
    last_time   = time.time()
    scan_y      = 0
    display_bbox = None

    label_decay = em.recent_decay
    bbox_lerp   = em.bbox_lerp

    # ── Deep Learning is ON by default now ──────────────────────────────────────
    use_dl = True

    print("\nControls:")
    print("  +/-   : adjust smoothing")
    print("  [/]   : adjust bbox lerp")
    print("  d     : toggle DL (DeepFace/ONNX)")
    print("  s     : save screenshot")
    print("  ESC   : exit\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Warning: Cannot read frame from webcam.")
                break

            h, w = frame.shape[:2]

            # ── face detection ─────────────────────────────────────────────────
            bbox, landmarks = locator.scan(frame)

            # smooth bbox
            if bbox:
                if display_bbox is None:
                    display_bbox = bbox
                else:
                    x0, y0, w0, h0 = display_bbox
                    x1, y1, w1, h1 = bbox
                    display_bbox = (
                        int(x0 + (x1 - x0) * bbox_lerp),
                        int(y0 + (y1 - y0) * bbox_lerp),
                        int(w0 + (w1 - w0) * bbox_lerp),
                        int(h0 + (h1 - h0) * bbox_lerp),
                    )
            else:
                display_bbox = None

            # ── default output ─────────────────────────────────────────────────
            label, conf, persona, alpha = (
                "neutral", 0.0, em.PERSONA_MAP["neutral"], 1.0
            )

            # ── emotion prediction ─────────────────────────────────────────────
            if display_bbox:
                face_crop = crop_to_bbox(frame, display_bbox)

                if use_dl and face_crop.size > 0:
                    # DL path: pass colour crop to DeepFace; grayscale to ONNX
                    try:
                        if em.dl_backend == "onnx":
                            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if face_crop.ndim == 3 else face_crop
                            label, conf, persona, alpha = em.predict_dl(gray)
                        else:
                            # DeepFace works best on the full frame, but a tight
                            # crop is fine too – avoids re-detecting the face.
                            label, conf, persona, alpha = em.predict_dl(face_crop)
                    except Exception as e:
                        print(f"[main] DL predict failed: {e}")
                        # graceful fallback
                        if landmarks:
                            label, conf, persona, alpha = em.predict(landmarks, (h, w), frame=face_crop)
                        else:
                            label, conf, persona, alpha = em.predict([], (h, w), frame=face_crop)

                else:
                    # Heuristic path (no DL or DL toggled off)
                    label, conf, persona, alpha = em.predict(
                        landmarks, (h, w), frame=face_crop if not use_dl else None
                    )

            # ── draw HUD ───────────────────────────────────────────────────────
            disp = frame.copy()

            if display_bbox:
                disp = ou.draw_rounded_rect(disp, display_bbox, ou.NEON_BLUE, 2, 12, glow=True)
                ou.draw_emotion_label(disp, label, conf, persona, display_bbox, alpha)

            # scanline
            now     = time.time()
            scan_y  = (scan_y + int((now - last_time) * 180)) % h
            ou.draw_scanline(disp, scan_y, color=ou.NEON_YELLOW, thickness=2)

            # header
            ou.draw_glitch_text(disp, "NEON EMOTION & PERSONA SCANNER", (20, 50), base_color=ou.NEON_BLUE)

            # FPS
            fps         = 1.0 / max(1e-6, now - last_time)
            last_time   = now
            fps_smooth  = fps_smooth * 0.85 + fps * 0.15
            ou.draw_fps(disp, fps_smooth)

            # status
            dl_label = f"{'ON' if use_dl else 'OFF'}  Backend: {em.dl_backend}"
            status = [
                f"Smoothing: {label_decay:.2f}   BBox lerp: {bbox_lerp:.2f}",
                f"Deep Learning: {dl_label}",
                "Keys: +/- smooth | [/] bbox | d DL | s save | ESC exit",
            ]
            ou.draw_status_panel(disp, status, (10, 70))

            cv2.imshow("Neon Persona Scanner", disp)

            # ── keyboard ───────────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("+"), ord("=")):
                label_decay     = min(0.98, label_decay + 0.03)
                em.recent_decay = label_decay
            elif key == ord("-"):
                label_decay     = max(0.50, label_decay - 0.03)
                em.recent_decay = label_decay
            elif key == ord("]"):
                bbox_lerp     = min(0.90, bbox_lerp + 0.05)
                em.bbox_lerp  = bbox_lerp
            elif key == ord("["):
                bbox_lerp     = max(0.02, bbox_lerp - 0.05)
                em.bbox_lerp  = bbox_lerp
            elif key == ord("d"):
                use_dl = not use_dl
                print(f"Deep Learning: {'ON' if use_dl else 'OFF'}")
            elif key in (ord("s"), ord("S")):
                path = ou.save_screenshot(disp)
                play_shutter_sound()
                print("Screenshot saved:", path)
            elif key == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()