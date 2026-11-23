# Face Detection and People Counting under Challenging Imaging Conditions
### Classical (Haar Cascade) vs. YOLOv8-face — Digital Image Processing Project

Group: No1 (Member 1), No2 (Member 2)  ← replace with your student numbers and rename the folder `Proje1_No1_No2`

---

## 1. Folder structure
```
Proje1_No1_No2/
├── README.md               this file
├── requirements.txt        Python dependencies
├── config.py               ALL fixed parameters (thresholds, enhancement settings)
├── run_experiments.py      runs the full experiment -> results/
├── app.py                  web interface for the live demo (python app.py)
├── demo.py                 command-line demo on any image / webcam
├── flowchart.png           method flowchart (flowchart.dot = source)
├── src/
│   ├── enhance.py          brightness/contrast, MANUAL histogram equalization, CLAHE
│   ├── detectors.py        HaarFaceDetector, YoloFaceDetector
│   ├── evaluate.py         YOLO-label loading, IoU, one-to-one matching, metrics
│   └── visualize.py        box drawing, image statistics
├── models/
│   ├── haarcascade_frontalface_default.xml
│   └── yolov8n-face.pt
├── dataset/
│   ├── images/             normal_01.jpg ... low_01.jpg ... backlight_01.jpg ...
│   └── labels/             normal_01.txt ...  (YOLO format, one line per face)
└── results/                created by run_experiments.py
```

## 2. Installation (Windows / Linux / macOS, Python 3.10–3.13)
```bash
python -m venv venv
venv\Scripts\activate            # Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```
Everything runs on the **CPU**; no GPU is needed. To use an NVIDIA GPU, set `DEVICE = "0"` in `config.py`
(the same device must then be used for all runs).

## 3. Dataset and ground truth
1. Put the photos in `dataset/images/`. The file name starts with the condition:
   `normal_XX.jpg`, `low_XX.jpg`, `backlight_XX.jpg` (or `small_XX`, `occlusion_XX`).
2. Annotate at https://www.makesense.ai: upload the images → *Object Detection* → label `face` →
   draw a box around every visible face → *Actions → Export Annotations → YOLO format*.
3. Unzip the exported `.txt` files into `dataset/labels/`. Each line: `0 x_center y_center width height` (normalised).
   The number of lines is the true face count. An image with no label file is skipped.

## 4. Run the experiment
```bash
python run_experiments.py
```
Outputs in `results/`:

| File | Content |
|---|---|
| `per_image.csv` | every image × enhancement × detector: GT count, predicted count, TP/FP/FN, P/R/F1, abs. count error, times, predicted boxes, image statistics |
| `summary_by_method.csv` | dataset level: TP, FP, FN, precision, recall, F1, MAE, mean times per detector × enhancement |
| `summary_by_condition.csv` | the same, split by normal / low / backlight |
| `image_quality_by_enhancement.csv` | mean brightness, RMS contrast, entropy ("how good it looks") |
| `settings.json` | all parameters, library versions, machine |
| `figures/*.png` | F1, MAE, time and per-condition charts; histogram figures |
| `vis/<detector>/<enhancement>/` | images with boxes: **green** = ground truth, **blue** = correct detection (TP), **red** = false detection (FP) |
| `enhanced/<enhancement>/` | the enhanced images |

## 5. Live demo
**Web interface (recommended):**
```bash
python app.py
```
The browser opens at http://127.0.0.1:7860 with three tabs:
* **Live camera**: webcam stream; Haar (left) and YOLO (right) run on the same frame at the same time, with face counts and times.
  Choose the enhancement and change parameters with sliders.
* **Photo: all enhancements**: upload or capture a photo; both detectors × 4 enhancements plus a results table (TP/FP/FN if a label exists).
* **Experiment results**: tables and charts from `run_experiments.py`.

**Command line alternative:**
```bash
python demo.py --image path/to/new_image.jpg --show
python demo.py --image path/to/new_image.jpg --haar-min-neighbors 3 --yolo-conf 0.4   # parameter change
python demo.py --webcam --show                                                         # SPACE = capture
```
Shows a 2×4 grid (Haar / YOLO × original + 3 enhancements) and prints face counts and times.
If a label with the same name is in `dataset/labels/`, TP/FP/FN are printed too.

## 6. Method summary
* **Enhancement** (applied on the Y channel of YCrCb, fixed parameters):
  brightness/contrast `g = 1.5·f + 30` (own implementation); histogram equalization
  (**own implementation**, verified against `cv2.equalizeHist`: max difference 0); CLAHE (clip 2.0, tiles 8×8, OpenCV).
* **Detectors**: Haar `scaleFactor=1.1, minNeighbors=5, minSize=30×30`;
  YOLOv8n-face `conf=0.25, NMS IoU=0.45, imgsz=640`. The same settings are used for every image.
* **Evaluation**: greedy one-to-one matching by highest IoU, TP if IoU ≥ 0.50.
  Precision/recall/F1 from the summed TP/FP/FN; counting error `|pred − GT|` per image; MAE = mean over images.
* **Timing**: `detector_ms` = detector call only (incl. its own preprocessing, after one warm-up run);
  `end_to_end_ms` = read file + enhancement + detection. Both are reported.

## 7. Pretrained models
| Model | Version | Source |
|---|---|---|
| Haar Cascade frontal face | `haarcascade_frontalface_default.xml` (OpenCV 4.x) | https://github.com/opencv/opencv/tree/4.x/data/haarcascades |
| YOLOv8n-face (trained on WIDER FACE) | release 1.0.0, `yolov8n-face.pt`, SHA-256 `d545bf1a…26ca` | https://github.com/akanametov/yolo-face |
| Ultralytics YOLO framework | see `requirements.txt` / `settings.json` | https://github.com/ultralytics/ultralytics |

## 8. Reproducibility, ethics and AI use
* No randomness is involved; detection on CPU is deterministic. All parameters are in `config.py`.
* Photos were taken with the informed consent of all group members; no minors; used only for this course.
  Only face **detection** is performed, with no recognition or identity labels.
* AI assistance: Claude (Anthropic) was used to help write the code structure and documentation.
  All code was reviewed, tested and is explained by the group members.
