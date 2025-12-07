import time
import os
import cv2
import numpy as np

# Optional shutter sound on Windows
try:
    import winsound
except Exception:
    winsound = None

from face_locator import FaceLocator
from emotion_model import EmotionModel
import overlay_utils as ou


# Helper
def play_shutter_sound():
    """Play a small shutter sound on Windows systems."""
    if winsound is None:
        return
    try:
        winsound.Beep(1000, 80)
        winsound.Beep(1400, 60)
    except Exception:
        pass


def crop_to_bbox(img, bbox, pad=8):
    """Crop image using bounding box + padding."""
    x, y, w, h = bbox
    h_img, w_img = img.shape[:2]
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w_img, x + w + pad)
    y2 = min(h_img, y + h + pad)
    return img[y1:y2, x1:x2]

# Main
def main():
    # initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam. Connect a camera and retry.")
        return

    locator = FaceLocator()

    # initialize emotion model
    em = EmotionModel(mode="image")  # default: heuristic + HF text
    em.dl_backend = "onnx"  # prefer ONNX if available

    try:
        em.load()
        print("HuggingFace text model loaded (or ready to download).")
    except Exception as e:
        print("HF model unavailable → running in pure heuristic mode:", e)

    fps_smooth = 30.0
    last_time = time.time()
    scan_y = 0
    display_bbox = None

    # smoothing parameters
    label_decay = em.recent_decay
    bbox_lerp = em.bbox_lerp

    use_dl = False  # deep learning OFF by default

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Warning: Cannot read frame from webcam.")
                break

            h, w = frame.shape[:2]

            # FACE + LANDMARKS
            bbox, landmarks = locator.scan(frame)

            # smooth bbox using lerp
            if bbox:
                if display_bbox is None:
                    display_bbox = bbox
                else:
                    x0, y0, w0, h0 = display_bbox
                    x1, y1, w1, h1 = bbox

                    nx = int(x0 + (x1 - x0) * bbox_lerp)
                    ny = int(y0 + (y1 - y0) * bbox_lerp)
                    nw = int(w0 + (w1 - w0) * bbox_lerp)
                    nh = int(h0 + (h1 - h0) * bbox_lerp)

                    display_bbox = (nx, ny, nw, nh)
            else:
                display_bbox = None

            # default output
            label, conf, persona, alpha = (
                "neutral",
                0.0,
                em.PERSONA_MAP.get("neutral"),
                1.0,
            )

            # EMOTION PREDICTION
            if bbox and use_dl:
                try:
                    face_crop = crop_to_bbox(frame, bbox)

                    # ONNX requires grayscale for FER+
                    if face_crop.ndim == 3:
                        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                    else:
                        gray = face_crop

                    label, conf, persona, alpha = em.predict_dl(gray)

                except Exception:
                    # fallback to heuristic
                    if landmarks:
                        try:
                            label, conf, persona, alpha = em.predict(
                                landmarks, (h, w)
                            )
                        except Exception:
                            pass

            elif landmarks:
                # heuristic prediction
                try:
                    label, conf, persona, alpha = em.predict(
                        landmarks, (h, w), frame=None
                    )
                except Exception:
                    pass

            # DRAW HUD OVERLAY
            disp = frame.copy()

            if display_bbox:
                disp = ou.draw_rounded_rect(
                    disp, display_bbox, ou.NEON_BLUE, 2, 12, glow=True
                )
                ou.draw_emotion_label(
                    disp, label, conf, persona, display_bbox, alpha
                )

            # moving scanline
            now = time.time()
            scan_y += int((now - last_time) * 180)
            scan_y %= h
            ou.draw_scanline(disp, scan_y, color=ou.NEON_YELLOW, thickness=2)

            # header + fps
            ou.draw_glitch_text(disp, "NEON EMOTION & PERSONA SCANNER", (20,50), base_color=ou.NEON_BLUE)

            fps = 1.0 / max(1e-6, (now - last_time))
            last_time = now

            fps_smooth = fps_smooth * 0.85 + fps * 0.15
            ou.draw_fps(disp, fps_smooth)

            # status panel
            status = [
                f"Smoothing: {label_decay:.2f}   BBox lerp: {bbox_lerp:.2f}",
                f"Deep Learning: {'ON' if use_dl else 'OFF'}   Backend: {em.dl_backend}",
                "Keys: +/- smoothing | [/] bbox | d DL-toggle | s save | ESC exit",
            ]
            ou.draw_status_panel(disp, status, (10,60))

            # display frame
            cv2.imshow("Neon Persona Scanner", disp)

            # KEYBOARD EVENTS
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('+'), ord('=')):
                label_decay = min(0.98, label_decay + 0.03)
                em.recent_decay = label_decay

            elif key == ord('-'):
                label_decay = max(0.50, label_decay - 0.03)
                em.recent_decay = label_decay

            elif key == ord(']'):
                bbox_lerp = min(0.90, bbox_lerp + 0.05)
                em.bbox_lerp = bbox_lerp

            elif key == ord('['):
                bbox_lerp = max(0.02, bbox_lerp - 0.05)
                em.bbox_lerp = bbox_lerp

            elif key == ord('d'):
                use_dl = not use_dl

            elif key in (ord('s'), ord('S')):
                path = ou.save_screenshot(disp)
                play_shutter_sound()
                print("Screenshot saved:", path)

            elif key == 27:  # ESC
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()