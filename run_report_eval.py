"""
run_report_eval.py

Tum veri setinde 4 on isleme x 2 dedektor deneyini sabit config.py
parametreleriyle calistirir.

Yer gercegi (.txt) varsa IoU >= 0.50 ile TP/FP/FN hesaplanir.
Etiket yoksa bu metrikler bos birakilir; uydurma deger yazilmaz.
Tespit sayisi, goruntu istatistikleri ve sureler her durumda kaydedilir.

Sure tanimi:
  detector_ms   = yalnizca dedektor cagrisi (cikarim; dosya okuma ve on isleme haric)
  end_to_end_ms = imread + on isleme + dedektor cagrisi
"""
import hashlib
import json
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import config  # noqa: E402
import detectors  # noqa: E402
import enhance  # noqa: E402
import evaluate as ev  # noqa: E402
import visualize as vz  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
OUT = config.RESULTS_DIR / "report_figures"
plt.rcParams["font.family"] = "DejaVu Sans"

ENH_TR = {
    "original": "Ham goruntu",
    "brightness_contrast": "Parlaklik/kontrast",
    "hist_eq": "Histogram esitleme",
    "clahe": "CLAHE",
}
DET_TR = {"haar": "Haar Cascade", "yolo": "YOLOv8-face"}

# Dosya adi onekleri. Ucuncu grup yonergeye gore tek kovada birlestirilir.
PREFIX_BUCKET = {
    "normal": "normal_isik",
    "low": "dusuk_isik",
    "backlight": "ters_isik_kucuk_yuz_ortulme",
    "small": "ters_isik_kucuk_yuz_ortulme",
    "occlusion": "ters_isik_kucuk_yuz_ortulme",
}
BUCKET_ORDER = ["normal_isik", "dusuk_isik", "ters_isik_kucuk_yuz_ortulme"]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def list_images():
    return sorted(p for p in config.IMAGES_DIR.iterdir() if p.suffix.lower() in IMG_EXT)


def audit(images):
    groups = defaultdict(list)
    prefixes = defaultdict(int)
    buckets = defaultdict(int)
    labelled = 0
    for p in images:
        digest = sha256_file(p)
        groups[digest].append(p.name)
        prefix = p.stem.split("_")[0].lower()
        prefixes[prefix] += 1
        buckets[PREFIX_BUCKET.get(prefix, f"tanimlanmamis:{prefix}")] += 1
        if (config.LABELS_DIR / f"{p.stem}.txt").exists():
            labelled += 1
    unique_names = [sorted(names)[0] for names in groups.values()]
    dup_groups = [
        {"sha256": d, "count": len(names), "files": sorted(names)}
        for d, names in groups.items() if len(names) > 1
    ]
    return {
        "n_files": len(images),
        "n_unique_sha256": len(groups),
        "n_label_files": labelled,
        "by_filename_prefix_files": dict(prefixes),
        "by_bucket_files": {k: int(buckets.get(k, 0)) for k in BUCKET_ORDER},
        "unique_representatives": sorted(unique_names),
        "duplicate_groups": dup_groups,
        "meets_15_unique": len(groups) >= 15,
        "meets_5_per_bucket": all(buckets.get(k, 0) >= 5 for k in BUCKET_ORDER),
        "label_dir": str(config.LABELS_DIR),
        "image_dir": str(config.IMAGES_DIR),
    }


def side_by_side(images_bgr, titles, out_path, suptitle):
    n = len(images_bgr)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.6))
    if n == 1:
        axes = [axes]
    for ax, img, title in zip(axes, images_bgr, titles):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def bar_grouped(df, value, ylabel, fname, title):
    piv = df.pivot(index="enhancement", columns="detector", values=value).loc[config.ENHANCEMENTS]
    piv.index = [ENH_TR[i] for i in piv.index]
    piv = piv.rename(columns=DET_TR)
    ax = piv.plot(kind="bar", figsize=(9, 4.6), rot=15, color=["#c0392b", "#1f6fb5"])
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.set_title(title)
    ax.legend(title="")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / fname, dpi=130)
    plt.close()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    images = list_images()
    info = audit(images)
    (config.RESULTS_DIR / "dataset_audit.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: info[k] for k in
                      ["n_files", "n_unique_sha256", "n_label_files",
                       "by_bucket_files", "meets_15_unique", "meets_5_per_bucket"]},
                     ensure_ascii=False, indent=2))
    if not images:
        print("Goruntu yok.")
        return

    # Manuel histogram esitlemenin OpenCV ile farki (yalnizca dogrulama).
    g0 = cv2.cvtColor(cv2.imread(str(images[0])), cv2.COLOR_BGR2GRAY)
    he_diff = enhance.verify_manual_hist_eq(g0)
    print(f"manual hist-eq vs cv2.equalizeHist max |diff| = {he_diff}")

    print("Modeller yukleniyor...")
    dets = [detectors.HaarFaceDetector(), detectors.YoloFaceDetector()]

    rows = []
    enhanced_cache = {}  # (name, enh) -> image, sadece ornekler icin bellekte tutulmaz
    # Ilk benzersiz orneklerin on islenmis hallerini histogram sekli icin sakla.
    want_hist = set(info["unique_representatives"][:3])
    saved_enh = {}

    for path in images:
        digest = sha256_file(path)
        cond = path.stem.split("_")[0].lower()
        img = cv2.imread(str(path))
        if img is None:
            print(f"  [okunamadi] {path.name}")
            continue
        h, w = img.shape[:2]
        label_path = config.LABELS_DIR / f"{path.stem}.txt"
        gts = ev.load_yolo_labels(label_path, w, h) if label_path.exists() else None
        for enh in config.ENHANCEMENTS:
            t_enh0 = time.perf_counter()
            e = enhance.apply(enh, img)
            enh_ms = (time.perf_counter() - t_enh0) * 1000
            if path.name in want_hist and enh == "original":
                saved_enh[path.name] = img
            for det in dets:
                t0 = time.perf_counter()
                # Uctan uca: dosya zaten okundu; burada on isleme + cikarim olculur.
                # Ayrica imread suresini ayrica eklemek icin dosyayi yeniden okumuyoruz;
                # asagida read_ms tekil olcum olarak kaydedilir.
                preds, det_ms = det.detect(e)
                e2e_ms = enh_ms + det_ms
                if gts is None:
                    tp = fp = fn = None
                    precision = recall = f1 = abs_err = None
                    matched = None
                else:
                    tp, fp, fn, matched = ev.match(preds, gts)
                    precision, recall, f1 = ev.prf(tp, fp, fn)
                    abs_err = abs(len(preds) - len(gts))
                rows.append({
                    "image": path.name,
                    "sha256": digest,
                    "unique_rep": False,
                    "condition_prefix": cond,
                    "bucket": PREFIX_BUCKET.get(cond, f"tanimlanmamis:{cond}"),
                    "width": w,
                    "height": h,
                    "enhancement": enh,
                    "detector": det.name,
                    "gt_count": None if gts is None else len(gts),
                    "pred_count": len(preds),
                    "abs_count_error": abs_err,
                    "tp": tp, "fp": fp, "fn": fn,
                    "precision": None if precision is None else round(precision, 4),
                    "recall": None if recall is None else round(recall, 4),
                    "f1": None if f1 is None else round(f1, 4),
                    "enhance_ms": round(enh_ms, 2),
                    "detector_ms": round(det_ms, 2),
                    "end_to_end_ms": round(e2e_ms, 2),
                    **vz.image_stats(e),
                })
            if path.name in want_hist:
                saved_enh[(path.name, enh)] = e
        print(f"  done {path.name}  {w}x{h}  gt={'yok' if gts is None else len(gts)}")

    # Benzersiz temsilci: her hash'in alfabetik ilk dosyasi
    rep = set(info["unique_representatives"])
    for r in rows:
        r["unique_rep"] = r["image"] in rep

    df = pd.DataFrame(rows)
    df.to_csv(config.RESULTS_DIR / "report_per_image.csv", index=False)

    # Ozet: tum dosyalar ve yalnizca ozgun temsilciler
    def summarize_counts(sub, tag):
        out_rows = []
        for (det, enh), g in sub.groupby(["detector", "enhancement"], sort=False):
            out_rows.append({
                "subset": tag,
                "detector": det,
                "enhancement": enh,
                "images": int(len(g)),
                "mean_pred_count": round(float(g.pred_count.mean()), 3),
                "std_pred_count": round(float(g.pred_count.std(ddof=0)), 3),
                "sum_pred_count": int(g.pred_count.sum()),
                "mean_brightness": round(float(g.mean_brightness.mean()), 2),
                "mean_rms_contrast": round(float(g.rms_contrast.mean()), 2),
                "mean_entropy": round(float(g.entropy.mean()), 3),
                "mean_enhance_ms": round(float(g.enhance_ms.mean()), 2),
                "mean_detector_ms": round(float(g.detector_ms.mean()), 2),
                "mean_end_to_end_ms": round(float(g.end_to_end_ms.mean()), 2),
                "TP": None if g.tp.isna().all() else int(g.tp.sum()),
                "FP": None if g.fp.isna().all() else int(g.fp.sum()),
                "FN": None if g.fn.isna().all() else int(g.fn.sum()),
                "precision": None,
                "recall": None,
                "F1": None,
                "MAE": None if g.abs_count_error.isna().all() else round(float(g.abs_count_error.mean()), 3),
            })
            if out_rows[-1]["TP"] is not None:
                p, r, f1 = ev.prf(out_rows[-1]["TP"], out_rows[-1]["FP"], out_rows[-1]["FN"])
                out_rows[-1]["precision"] = round(p, 4)
                out_rows[-1]["recall"] = round(r, 4)
                out_rows[-1]["F1"] = round(f1, 4)
        return out_rows

    summary_rows = summarize_counts(df, "tum_dosyalar")
    summary_rows += summarize_counts(df[df.unique_rep], "ozgun_temsilciler")
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(config.RESULTS_DIR / "report_summary.csv", index=False)

    # Grafikler ozgun temsilciler uzerinden (kopyalar ortalamayi sisirmesin)
    uniq = df[df.unique_rep]
    u_sum = summary[summary.subset == "ozgun_temsilciler"]
    bar_grouped(u_sum, "mean_pred_count", "Ortalama tespit sayisi",
                "pred_count_by_method.png",
                "Ozgun goruntulerde ortalama yuz tespit sayisi")
    bar_grouped(u_sum, "mean_detector_ms", "Ortalama cikarim suresi (ms)",
                "detector_time_by_method.png",
                "Dedektor cagrisi suresi (dosya okuma ve on isleme haric)")
    bar_grouped(u_sum, "mean_end_to_end_ms", "Ortalama uc tan uca sure (ms)",
                "e2e_time_by_method.png",
                "On isleme + cikarim suresi")

    q = uniq[uniq.detector == "haar"].groupby("enhancement")[
        ["mean_brightness", "rms_contrast", "entropy"]].mean().loc[config.ENHANCEMENTS]
    q.index = [ENH_TR[i] for i in q.index]
    ax = q.plot(kind="bar", figsize=(9, 4.6), rot=15)
    ax.set_title("On isleme sonrasi goruntu istatistikleri (ozgun goruntuler, ortalama)")
    ax.set_xlabel("")
    ax.legend(["Ortalama parlaklik", "RMS kontrast", "Entropi (bit)"])
    plt.tight_layout()
    plt.savefig(OUT / "image_quality_by_enhancement.png", dpi=130)
    plt.close()
    q.round(3).to_csv(config.RESULTS_DIR / "report_image_quality.csv")

    # Histogram ornekleri
    for name in sorted(want_hist):
        fig, ax = plt.subplots(2, 4, figsize=(14, 6.2))
        for k, enh in enumerate(config.ENHANCEMENTS):
            e = saved_enh[(name, enh)]
            gray = cv2.cvtColor(e, cv2.COLOR_BGR2GRAY)
            ax[0, k].imshow(cv2.cvtColor(e, cv2.COLOR_BGR2RGB))
            ax[0, k].set_title(ENH_TR[enh], fontsize=10)
            ax[0, k].axis("off")
            ax[1, k].hist(gray.ravel(), bins=256, range=(0, 255), color="#334155")
            ax[1, k].set_xlim(0, 255)
            ax[1, k].set_yticks([])
        fig.suptitle(f"On isleme ve parlaklik histogrami — {name}")
        fig.tight_layout()
        fig.savefig(OUT / f"hist_{name}.png", dpi=120)
        plt.close(fig)

    # Karsilastirma: Haar ve YOLO kutu sayisi en cok ayrisan 3 ozgun goruntu
    # ve sayilarin esit oldugu 1 goruntu. Ham goruntu uzerinde.
    base = uniq[uniq.enhancement == "original"]
    wide = base.pivot(index="image", columns="detector", values="pred_count")
    wide["gap"] = (wide["haar"] - wide["yolo"]).abs()
    pick = list(wide.sort_values("gap", ascending=False).head(3).index)
    equal = wide[wide.gap == 0]
    if len(equal):
        pick.append(equal.index[0])
    # Cesitlilik icin temsilcilerden en az 4
    for name in info["unique_representatives"]:
        if name not in pick:
            pick.append(name)
        if len(pick) >= 4:
            break

    HAAR_COLOR = (60, 60, 230)
    YOLO_COLOR = (230, 140, 40)
    for name in pick:
        img = cv2.imread(str(config.IMAGES_DIR / name))
        panels = []
        titles = []
        for enh in config.ENHANCEMENTS:
            e = enhance.apply(enh, img)
            panels.append(e)
            titles.append(ENH_TR[enh])
        side_by_side(panels, titles, OUT / f"enh_compare_{name}.png",
                     f"On isleme karsilastirmasi — {name}")
        # Ham ve CLAHE uzerinde iki dedektor
        for enh in ("original", "clahe"):
            e = enhance.apply(enh, img)
            tiles = []
            ttitles = []
            for det in dets:
                preds, ms = det.detect(e)
                drawn = vz.draw(e, None, preds, None,
                                f"{DET_TR[det.name]} | {len(preds)} kutu | {ms:.0f} ms",
                                pred_color=HAAR_COLOR if det.name == "haar" else YOLO_COLOR)
                tiles.append(drawn)
                ttitles.append(f"{DET_TR[det.name]} ({len(preds)})")
            side_by_side(tiles, ttitles, OUT / f"det_{enh}_{name}.png",
                         f"{ENH_TR[enh]} — {name}  (yesil yer gercegi yok; kutular tespit)")

    import torch
    import ultralytics
    settings = {
        "haar": dets[0].params(),
        "yolo": dets[1].params(),
        "iou_match_threshold": config.IOU_MATCH_THRESHOLD,
        "brightness_contrast": {"alpha": config.BC_ALPHA, "beta": config.BC_BETA},
        "clahe": {"clip": config.CLAHE_CLIP, "tile": list(config.CLAHE_TILE)},
        "manual_hist_eq_max_abs_diff_vs_opencv": he_diff,
        "timing_definition": {
            "detector_ms": "Yalnizca dedektor cagrisi (cikarim). Dosya okuma ve on isleme dahil degildir. Haar icin gri ton donusumu bu sureye dahildir.",
            "enhance_ms": "Yalnizca on isleme.",
            "end_to_end_ms": "On isleme + dedektor cagrisi. Diskten imread bu toplama dahil edilmedi; model yuklemesi ve isinma (__init__) da dahil degildir.",
        },
        "ground_truth_available": bool(info["n_label_files"]),
        "n_files": info["n_files"],
        "n_unique": info["n_unique_sha256"],
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "machine": f"{platform.system()} {platform.release()} {platform.machine()} | {platform.processor()}",
        "device": config.DEVICE,
        "seed": config.SEED,
        "comparison_images": pick,
    }
    # minSize tuple json
    settings["haar"]["minSize"] = list(settings["haar"]["minSize"])
    (config.RESULTS_DIR / "report_settings.json").write_text(
        json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Ozgun temsilciler ===")
    print(u_sum.to_string(index=False))
    print(f"\nSekiller: {OUT}")


if __name__ == "__main__":
    main()
