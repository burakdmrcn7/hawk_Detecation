"""
run_experiments.py - runs the WHOLE experiment and writes all tables and figures.

    python run_experiments.py

For every image in dataset/images:
    for every enhancement in [original, brightness_contrast, hist_eq, clahe]:
        for every detector in [haar, yolo]:
            detect -> match with ground truth (IoU >= 0.50) -> TP / FP / FN, count error, time

Image file names must start with their condition, e.g.
    normal_01.jpg   low_03.jpg   backlight_02.jpg   (or small_.. / occlusion_..)
and have a label file with the same name in dataset/labels (normal_01.txt).
"""
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import config                                   # noqa: E402
import enhance                                  # noqa: E402
import detectors                                # noqa: E402
import evaluate as ev                           # noqa: E402
import visualize as vz                          # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
OUT = config.RESULTS_DIR


def condition_of(path):
    return path.stem.split("_")[0].lower()


def list_images():
    imgs = sorted(p for p in config.IMAGES_DIR.iterdir() if p.suffix.lower() in IMG_EXT)
    keep = []
    for p in imgs:
        if not (config.LABELS_DIR / f"{p.stem}.txt").exists():
            print(f"  [skip] no ground-truth label for {p.name}")
            continue
        keep.append(p)
    return keep


def histogram_figure(img, name):
    """Original + 3 enhanced versions with their grayscale histograms."""
    fig, ax = plt.subplots(2, 4, figsize=(16, 6.5))
    for k, enh in enumerate(config.ENHANCEMENTS):
        e = enhance.apply(enh, img)
        gray = cv2.cvtColor(e, cv2.COLOR_BGR2GRAY)
        ax[0, k].imshow(cv2.cvtColor(e, cv2.COLOR_BGR2RGB)); ax[0, k].set_title(enh); ax[0, k].axis("off")
        ax[1, k].hist(gray.ravel(), bins=256, range=(0, 255), color="#444")
        ax[1, k].set_xlim(0, 255); ax[1, k].set_yticks([])
    fig.suptitle(f"Enhancements and histograms - {name}")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / f"histograms_{name}.png", dpi=110)
    plt.close(fig)


def bar_chart(summary, metric, fname, ylabel):
    piv = summary.pivot(index="enhancement", columns="detector", values=metric).loc[config.ENHANCEMENTS]
    ax = piv.plot(kind="bar", figsize=(8, 4.5), rot=0, color=["#c0504d", "#4f81bd"])
    ax.set_ylabel(ylabel); ax.set_xlabel(""); ax.set_title(f"{ylabel} by enhancement and detector")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", fontsize=8)
    plt.tight_layout(); plt.savefig(OUT / "figures" / fname, dpi=120); plt.close()


def condition_chart(by_cond, metric, fname):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, det in zip(axes, ["haar", "yolo"]):
        d = by_cond[by_cond.detector == det]
        piv = d.pivot(index="condition", columns="enhancement", values=metric)[config.ENHANCEMENTS]
        piv.plot(kind="bar", ax=ax, rot=0)
        ax.set_title(f"{det.upper()} - {metric} per condition"); ax.set_xlabel("")
        ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(OUT / "figures" / fname, dpi=120); plt.close()


def main():
    for sub in ["figures", "vis", "enhanced"]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    images = list_images()
    if not images:
        print("No labelled images found in dataset/images + dataset/labels."); return
    print(f"{len(images)} labelled images")

    dets = [detectors.HaarFaceDetector(), detectors.YoloFaceDetector()]

    # check that our manual histogram equalization matches OpenCV's
    g0 = cv2.cvtColor(cv2.imread(str(images[0])), cv2.COLOR_BGR2GRAY)
    he_diff = enhance.verify_manual_hist_eq(g0)
    print(f"manual hist-eq vs cv2.equalizeHist: max pixel difference = {he_diff}")

    rows, first_of_condition = [], set()
    for path in images:
        cond = condition_of(path)
        for enh in config.ENHANCEMENTS:
            for det in dets:
                t0 = time.perf_counter()
                img = cv2.imread(str(path))                                  # read
                h, w = img.shape[:2]
                e = enhance.apply(enh, img)                                  # enhance
                preds, det_ms = det.detect(e)                                # detect
                e2e_ms = (time.perf_counter() - t0) * 1000

                gts = ev.load_yolo_labels(config.LABELS_DIR / f"{path.stem}.txt", w, h)
                tp, fp, fn, matched = ev.match(preds, gts)
                p, r, f1 = ev.prf(tp, fp, fn)
                rows.append({"image": path.name, "condition": cond, "enhancement": enh,
                             "detector": det.name, "gt_count": len(gts), "pred_count": len(preds),
                             "abs_count_error": abs(len(preds) - len(gts)),
                             "tp": tp, "fp": fp, "fn": fn,
                             "precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3),
                             "detector_ms": round(det_ms, 2), "end_to_end_ms": round(e2e_ms, 2),
                             **vz.image_stats(e),
                             "pred_boxes": json.dumps([[round(v, 1) for v in b] for b in preds])})

                vis_dir = OUT / "vis" / det.name / enh
                vis_dir.mkdir(parents=True, exist_ok=True)
                title = f"{det.name}/{enh}  GT={len(gts)} pred={len(preds)} TP={tp} FP={fp} FN={fn}"
                cv2.imwrite(str(vis_dir / path.name), vz.draw(e, gts, preds, matched, title))
            (OUT / "enhanced" / enh).mkdir(exist_ok=True)
            cv2.imwrite(str(OUT / "enhanced" / enh / path.name), e)
        if cond not in first_of_condition:
            first_of_condition.add(cond)
            histogram_figure(cv2.imread(str(path)), path.stem)
        print(f"  done {path.name}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "per_image.csv", index=False)

    summary = pd.DataFrame(ev.summarize(df, ["detector", "enhancement"]))
    by_cond = pd.DataFrame(ev.summarize(df, ["detector", "enhancement", "condition"]))
    summary.to_csv(OUT / "summary_by_method.csv", index=False)
    by_cond.to_csv(OUT / "summary_by_condition.csv", index=False)

    look = df[df.detector == "haar"].groupby("enhancement")[["mean_brightness", "rms_contrast", "entropy"]].mean()
    look.loc[config.ENHANCEMENTS].round(2).to_csv(OUT / "image_quality_by_enhancement.csv")

    bar_chart(summary, "F1", "f1_by_method.png", "F1 score")
    bar_chart(summary, "MAE", "mae_by_method.png", "Counting MAE (faces)")
    bar_chart(summary, "mean_detector_ms", "time_by_method.png", "Mean detector time (ms)")
    condition_chart(by_cond, "recall", "recall_by_condition.png")
    condition_chart(by_cond, "F1", "f1_by_condition.png")

    import torch, ultralytics
    settings = {"haar": dets[0].params(), "yolo": dets[1].params(),
                "iou_match_threshold": config.IOU_MATCH_THRESHOLD,
                "brightness_contrast": {"alpha": config.BC_ALPHA, "beta": config.BC_BETA},
                "clahe": {"clip": config.CLAHE_CLIP, "tile": config.CLAHE_TILE},
                "manual_hist_eq_max_diff_vs_opencv": he_diff,
                "timing": "detector_ms = detector call only; end_to_end_ms = read + enhance + detect",
                "python": platform.python_version(), "opencv": cv2.__version__,
                "torch": torch.__version__, "ultralytics": ultralytics.__version__,
                "machine": f"{platform.system()} {platform.machine()} | {platform.processor()}",
                "seed": config.SEED, "n_images": len(images)}
    (OUT / "settings.json").write_text(json.dumps(settings, indent=2))

    pd.set_option("display.width", 200)
    print("\n=== Summary (all images) ===")
    print(summary.to_string(index=False))
    print(f"\nAll results written to {OUT}")


if __name__ == "__main__":
    main()
