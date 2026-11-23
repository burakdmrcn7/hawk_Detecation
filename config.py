"""
config.py - ALL fixed experiment settings in one place.

Rule from the assignment: thresholds and settings are NOT tuned per image.
Every image, every enhancement and every detector uses exactly these values.
If the instructor asks for a parameter change during the demo, change it here
(or pass it on the command line of demo.py) and re-run.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------- paths
IMAGES_DIR = ROOT / "dataset" / "images"
LABELS_DIR = ROOT / "dataset" / "labels"      # YOLO-format .txt (from makesense.ai)
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

HAAR_XML = MODELS_DIR / "haarcascade_frontalface_default.xml"
YOLO_WEIGHTS = MODELS_DIR / "yolov8n-face.pt"

# ---------------------------------------------------------------- Haar Cascade (Viola-Jones)
HAAR_SCALE_FACTOR = 1.1     # image pyramid step (smaller = slower, finds more sizes)
HAAR_MIN_NEIGHBORS = 5      # how many overlapping hits a face needs (higher = fewer false positives)
HAAR_MIN_SIZE = (30, 30)    # smallest face in pixels

# ---------------------------------------------------------------- YOLOv8-face
YOLO_CONF = 0.25            # minimum confidence to keep a box
YOLO_NMS_IOU = 0.45         # non-maximum suppression IoU
YOLO_IMGSZ = 640            # network input size
DEVICE = "cpu"              # "cpu" or "0" for an NVIDIA GPU. Keep the same for all runs.

# ---------------------------------------------------------------- image enhancement
BC_ALPHA = 1.5              # contrast gain   g(x) = alpha * f(x) + beta
BC_BETA = 30                # brightness offset
CLAHE_CLIP = 2.0            # CLAHE clip limit
CLAHE_TILE = (8, 8)         # CLAHE tile grid

ENHANCEMENTS = ["original", "brightness_contrast", "hist_eq", "clahe"]

# ---------------------------------------------------------------- evaluation
IOU_MATCH_THRESHOLD = 0.50  # a prediction is a True Positive if IoU >= 0.50 (one-to-one)

SEED = 0                    # no randomness is used; kept for reproducibility reporting
