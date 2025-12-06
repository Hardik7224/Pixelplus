import cv2
import time
import os
from typing import Tuple, List


TECH_BLUE   = (230, 160, 80)   # main blue
TECH_CYAN   = (220, 230, 200)  # text
TECH_BG     = (40, 45, 60)     # panel bg
TECH_WHITE  = (245, 245, 245)  # title text


def draw_rounded_rect(
    img,
    rect: Tuple[int, int, int, int],
    color,
    thickness: int = 2,
    radius: int = 10,
    glow: bool = False,  # kept for compatibility, not used
):

    x, y, w, h = rect
    x1, y1 = x, y
    x2, y2 = x + w, y + h

    # Just draw a rectangle; keep it minimal
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    return img


def draw_scanline(img, y_pos: int, color=None, thickness: int = 1):

    h, w = img.shape[:2]
    if color is None:
        color = (60, 65, 80)
    cv2.line(img, (0, max(0, min(h - 1, y_pos))), (w, max(0, min(h - 1, y_pos))), color, thickness)
    return img


def draw_fps(img, fps: float, pos=(10, 24)):

    txt = f"FPS: {int(fps):02d}"
    cv2.putText(img, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.55, TECH_CYAN, 2, cv2.LINE_AA)


def draw_emotion_label(
    img,
    label: str,
    conf: float,
    persona: str,
    bbox: Tuple[int, int, int, int],
    alpha: float = 1.0,  # kept for compatibility
):

    x, y, w, h = bbox
    top_x = x
    top_y = max(24, y - 18)

    main_text = f"{label.upper()}  {int(conf * 100)}%"
    cv2.putText(
        img,
        main_text,
        (top_x, top_y),
        cv2.FONT_HERSHEY_DUPLEX,
        0.6,
        TECH_BLUE,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        img,
        persona,
        (top_x, top_y + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        TECH_CYAN,
        1,
        cv2.LINE_AA,
    )


def draw_glitch_text(
    img,
    text: str,
    pos=(20, 40),
    base_color=TECH_WHITE,
):

    x, y = pos
    cv2.putText(
        img,
        text,
        (x + 1, y + 1),
        cv2.FONT_HERSHEY_DUPLEX,
        0.8,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_DUPLEX,
        0.8,
        base_color,
        1,
        cv2.LINE_AA,
    )


def draw_status_panel(
    img,
    lines: List[str],
    pos=(10, 60),
    bg_color=TECH_BG,
    alpha: float = 0.6,
):

    x, y = pos
    h, w = img.shape[:2]

    line_h = 18
    pad_x, pad_y = 10, 10
    panel_h = pad_y * 2 + line_h * max(1, len(lines))
    panel_w = 260

    x2 = min(w - 5, x + panel_w)
    y2 = min(h - 5, y + panel_h)

    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x2, y2), bg_color, -1)
    img[:] = cv2.addWeighted(img, 1.0, overlay, alpha, 0.0)

    for i, line in enumerate(lines):
        ty = y + pad_y + (i + 1) * line_h
        cv2.putText(
            img,
            line,
            (x + pad_x, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            TECH_CYAN,
            1,
            cv2.LINE_AA,
        )
    return img


def save_screenshot(img, out_dir: str = "screenshots") -> str:

    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    path = os.path.join(out_dir, f"frame_{ts}.png")
    cv2.imwrite(path, img)
    return path


