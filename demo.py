"""
demo.py - live demonstration on ANY image (also new images the instructor brings).

Examples
    python demo.py --image dataset/images/low_01.jpg
    python demo.py --image new_photo.jpg --show
    python demo.py --image new_photo.jpg --haar-min-neighbors 3 --yolo-conf 0.4
    python demo.py --webcam                      (press SPACE to capture, Q to quit)

Output: a 2 x 4 grid (rows = Haar / YOLO, columns = original + 3 enhancements)
saved as results/demo_<name>.jpg, with face counts and times printed.
If a ground-truth .txt with the same name exists in dataset/labels, TP/FP/FN are shown too.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import config                    # noqa: E402
import enhance                   # noqa: E402
import detectors                 # noqa: E402
import evaluate as ev            # noqa: E402
import visualize as vz           # noqa: E402


def parse():
    a = argparse.ArgumentParser()
    a.add_argument("--image", help="path to an image")
    a.add_argument("--webcam", action="store_true", help="capture a frame from the webcam")
    a.add_argument("--show", action="store_true", help="open a window with the result")
    a.add_argument("--haar-scale", type=float)
    a.add_argument("--haar-min-neighbors", type=int)
    a.add_argument("--yolo-conf", type=float)
    a.add_argument("--alpha", type=float, help="contrast gain")
    a.add_argument("--beta", type=float, help="brightness offset")
    a.add_argument("--clahe-clip", type=float)
    a.add_argument("--device", help="cpu or 0")
    return a.parse_args()


def grab_webcam():
    cap = cv2.VideoCapture(0)
    frame = None
    while True:
        ok, f = cap.read()
        if not ok:
            break
        cv2.imshow("SPACE = capture, Q = quit", f)
        k = cv2.waitKey(1) & 0xFF
        if k == ord(" "):
            frame = f; break
        if k == ord("q"):
            break
    cap.release(); cv2.destroyAllWindows()
    return frame


def main():
    a = parse()
    # command-line overrides (same value then applies to every enhancement)
    if a.alpha is not None: config.BC_ALPHA = a.alpha
    if a.beta is not None: config.BC_BETA = a.beta
    if a.clahe_clip is not None: config.CLAHE_CLIP = a.clahe_clip

    if a.webcam:
        img, name = grab_webcam(), "webcam"
    else:
        img, name = cv2.imread(a.image), Path(a.image).stem
    if img is None:
        print("could not read image"); return

    dets = [detectors.HaarFaceDetector(scale_factor=a.haar_scale, min_neighbors=a.haar_min_neighbors),
            detectors.YoloFaceDetector(conf=a.yolo_conf, device=a.device)]
    print("Haar:", dets[0].params()); print("YOLO:", dets[1].params())

    h, w = img.shape[:2]
    gts = ev.load_yolo_labels(config.LABELS_DIR / f"{name}.txt", w, h)

    rows = []
    for det in dets:
        tiles = []
        for enh in config.ENHANCEMENTS:
            e = enhance.apply(enh, img)
            preds, ms = det.detect(e)
            txt = f"{det.name}/{enh}: {len(preds)} faces  {ms:.0f} ms"
            matched = None
            if gts is not None:
                tp, fp, fn, matched = ev.match(preds, gts)
                txt += f"  (GT={len(gts)} TP={tp} FP={fp} FN={fn})"
            print(txt)
            tiles.append(vz.draw(e, gts, preds, matched, f"{det.name} | {enh} | {len(preds)} faces"))
        rows.append(np.hstack(tiles))
    grid = np.vstack(rows)
    scale = 1920 / grid.shape[1]
    grid = cv2.resize(grid, None, fx=scale, fy=scale) if scale < 1 else grid

    out = config.RESULTS_DIR / f"demo_{name}.jpg"
    out.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(out), grid)
    print("saved", out)
    if a.show:
        cv2.imshow("demo (any key to close)", grid); cv2.waitKey(0); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
