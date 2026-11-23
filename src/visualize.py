"""
visualize.py - drawing boxes and making figures for the report.
Green = ground truth, Red = wrong prediction (FP), Blue = correct prediction (TP).
"""
import cv2
import numpy as np

GREEN, RED, BLUE, WHITE = (0, 200, 0), (0, 0, 255), (255, 128, 0), (255, 255, 255)


def draw(img, gts, preds, matched=None, title="", pred_color=None):
    """pred_color: used when there is no ground truth (live demo), e.g. (0, 0, 255)."""
    out = img.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    t = max(2, round(min(out.shape[:2]) / 300))
    matched_preds = {i for i, _, _ in (matched or [])}
    for g in gts or []:
        x1, y1, x2, y2 = map(int, g[:4])
        cv2.rectangle(out, (x1, y1), (x2, y2), GREEN, t)
    for i, p in enumerate(preds):
        x1, y1, x2, y2 = map(int, p[:4])
        if gts is None and pred_color is not None:
            color = pred_color
        else:
            color = BLUE if i in matched_preds else RED
        cv2.rectangle(out, (x1, y1), (x2, y2), color, t)
        if p[4] < 1.0:      # YOLO confidence
            cv2.putText(out, f"{p[4]:.2f}", (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 * t, color, t)
    if title:
        scale = 0.42 * t
        (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, scale, t)
        cv2.rectangle(out, (0, 0), (tw + 10, th + 14), (0, 0, 0), -1)
        cv2.putText(out, title, (5, th + 7), cv2.FONT_HERSHEY_SIMPLEX, scale, WHITE, t)
    return out


def image_stats(img):
    """Simple 'how good does it look' numbers: mean brightness, RMS contrast, entropy."""
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = np.bincount(gray.ravel(), minlength=256) / gray.size
    nz = hist[hist > 0]
    return {"mean_brightness": round(float(gray.mean()), 1),
            "rms_contrast": round(float(gray.std()), 1),
            "entropy": round(float(-(nz * np.log2(nz)).sum()), 3)}
