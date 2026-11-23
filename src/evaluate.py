"""
evaluate.py - ground truth loading, IoU matching and metrics.      [MEMBER 2]

Ground truth: one YOLO-format .txt per image (exported from makesense.ai):
    class  x_center  y_center  width  height      (all normalised 0..1)

Matching: one-to-one, greedy by highest IoU first. A pair counts only if
IoU >= 0.50. Each ground-truth box and each prediction is used at most once.
    TP = matched predictions, FP = unmatched predictions, FN = unmatched GT
"""
from pathlib import Path

import numpy as np

import config


def load_yolo_labels(txt_path, img_w, img_h):
    """Read a YOLO .txt file and return pixel boxes [x1, y1, x2, y2]."""
    boxes = []
    txt_path = Path(txt_path)
    if not txt_path.exists():
        return None                       # caller decides what to do
    for line in txt_path.read_text().split("\n"):
        parts = line.split()
        if len(parts) < 5:
            continue
        _, xc, yc, w, h = map(float, parts[:5])
        x1 = (xc - w / 2) * img_w
        y1 = (yc - h / 2) * img_h
        x2 = (xc + w / 2) * img_w
        y2 = (yc + h / 2) * img_h
        boxes.append([x1, y1, x2, y2])
    return boxes


def iou(a, b):
    """Intersection over Union of two boxes [x1, y1, x2, y2]."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match(preds, gts, thr=None):
    """One-to-one greedy matching. Returns (tp, fp, fn, matched_pairs)."""
    thr = config.IOU_MATCH_THRESHOLD if thr is None else thr
    pairs = []
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            v = iou(p[:4], g[:4])
            if v >= thr:
                pairs.append((v, i, j))
    pairs.sort(reverse=True)               # highest IoU first
    used_p, used_g, matched = set(), set(), []
    for v, i, j in pairs:
        if i in used_p or j in used_g:
            continue
        used_p.add(i); used_g.add(j)
        matched.append((i, j, v))
    tp = len(matched)
    fp = len(preds) - tp
    fn = len(gts) - tp
    return tp, fp, fn, matched


def prf(tp, fp, fn):
    """Precision, recall, F1 (0 when undefined)."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def summarize(df, by):
    """Aggregate per-image rows (pandas DataFrame) into dataset-level metrics.
    Precision/recall/F1 are computed from the SUMMED TP/FP/FN (micro average)."""
    rows = []
    for key, g in df.groupby(by, sort=False):
        tp, fp, fn = int(g.tp.sum()), int(g.fp.sum()), int(g.fn.sum())
        p, r, f1 = prf(tp, fp, fn)
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(by, key)),
                     "images": len(g), "gt_faces": int(g.gt_count.sum()),
                     "TP": tp, "FP": fp, "FN": fn,
                     "precision": round(p, 3), "recall": round(r, 3), "F1": round(f1, 3),
                     "MAE": round(float(g.abs_count_error.mean()), 3),
                     "mean_detector_ms": round(float(g.detector_ms.mean()), 1),
                     "mean_end_to_end_ms": round(float(g.end_to_end_ms.mean()), 1)})
    return rows
