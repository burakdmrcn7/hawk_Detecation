"""
app.py - web interface for the live demonstration.

    python app.py        -> opens http://127.0.0.1:7860 in the browser

Tab 1  Live camera : webcam stream, Haar and YOLO side by side at the same time
Tab 2  Photo       : upload / capture one photo -> both detectors x 4 enhancements
Tab 3  Results     : tables and charts produced by run_experiments.py

All parameters start at the fixed values in config.py. The sliders exist only so
the instructor can request a parameter change during the defense.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import config                    # noqa: E402
import enhance                   # noqa: E402
import detectors                 # noqa: E402
import evaluate as ev            # noqa: E402
import visualize as vz           # noqa: E402

HAAR_COLOR = (60, 60, 230)       # BGR red-ish
YOLO_COLOR = (230, 140, 40)      # BGR blue-ish
DEFAULTS = dict(enh="original", scale=config.HAAR_SCALE_FACTOR, mn=config.HAAR_MIN_NEIGHBORS,
                conf=config.YOLO_CONF, alpha=config.BC_ALPHA, beta=config.BC_BETA, clip=config.CLAHE_CLIP)

print("Loading models ...")
YOLO = detectors.YoloFaceDetector()
_haar = {}


def haar(scale, mn):
    key = (round(scale, 3), int(mn))
    if key not in _haar:
        _haar[key] = detectors.HaarFaceDetector(scale_factor=key[0], min_neighbors=key[1])
    return _haar[key]


def set_enh_params(alpha, beta, clip):
    config.BC_ALPHA, config.BC_BETA, config.CLAHE_CLIP = float(alpha), float(beta), float(clip)


CONDITIONS = [("Normal light", "normal"), ("Low light", "low"), ("Backlight", "backlight"),
              ("Small / distant faces", "small"), ("Occlusion", "occlusion")]


def capture(cond):
    """Save the current camera frame (original, NOT enhanced) into dataset/images
    with the next free name for that condition, e.g. low_03.jpg."""
    if LAST_FRAME["img"] is None:
        return "⚠ Start the camera first (press **Record**)."
    config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    n = 1
    while any((config.IMAGES_DIR / f"{cond}_{n:02d}{ext}").exists() for ext in (".jpg", ".jpeg", ".png")):
        n += 1
    name = f"{cond}_{n:02d}.jpg"
    bgr = cv2.cvtColor(LAST_FRAME["img"], cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(config.IMAGES_DIR / name), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    imgs = [p for p in config.IMAGES_DIR.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    same = sum(p.stem.startswith(cond + "_") for p in imgs)
    h, w = bgr.shape[:2]
    return f"✓ **{name}** saved ({w}×{h}) · {cond}: **{same}**/5 · total: **{len(imgs)}**"


def redo_last(enh, scale, mn, conf, alpha, beta, clip):
    """Re-process the last camera frame with the new settings (camera may be stopped)."""
    if LAST_FRAME["img"] is None:
        return gr.update(), gr.update(), gr.update()
    return live(LAST_FRAME["img"], enh, scale, mn, conf, alpha, beta, clip)


def run_pair(bgr, enh, scale, mn, conf, gts=None):
    """Run both detectors on ONE enhanced image. Returns drawn images + numbers."""
    t0 = time.perf_counter()
    e = enhance.apply(enh, bgr)
    enh_ms = (time.perf_counter() - t0) * 1000
    hb, hms = haar(scale, mn).detect(e)
    YOLO.conf = float(conf)
    yb, yms = YOLO.detect(e)
    out = {}
    for name, boxes, ms, color in [("haar", hb, hms, HAAR_COLOR), ("yolo", yb, yms, YOLO_COLOR)]:
        matched, extra = None, ""
        if gts is not None:
            tp, fp, fn, matched = ev.match(boxes, gts)
            extra = f" | TP={tp} FP={fp} FN={fn}"
        img = vz.draw(e, gts, boxes, matched, f"{name.upper()} | {len(boxes)} face{'' if len(boxes) == 1 else 's'} | {ms:.0f} ms{extra}", color)
        out[name] = dict(img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB), n=len(boxes), ms=ms, extra=extra)
    return out, enh_ms


EMPTY_BOARD = ("<div class='metrics'>"
               "<div class='mcard haar'><div class='mh'>Haar Cascade <span>(Viola–Jones, 2001)</span></div>"
               "<div class='mv'>–</div><div class='ml'>faces detected</div><div class='mt'>t = – ms</div></div>"
               "<div class='mcard yolo'><div class='mh'>YOLOv8n-face <span>(CNN, WIDER FACE)</span></div>"
               "<div class='mv'>–</div><div class='ml'>faces detected</div><div class='mt'>t = – ms</div></div>"
               "</div><div class='mfoot'>Waiting for camera input …</div>")


def scoreboard(out, enh, enh_ms):
    h, y = out["haar"], out["yolo"]

    def card(cls, title, sub, o):
        extra = f"<div class='mt'>{o['extra'].strip(' |')}</div>" if o["extra"] else ""
        return (f"<div class='mcard {cls}'><div class='mh'>{title} <span>({sub})</span></div>"
                f"<div class='mv'>{o['n']}</div><div class='ml'>faces detected</div>"
                f"<div class='mt'>t = {o['ms']:.1f} ms</div>{extra}</div>")

    diff = abs(h["n"] - y["n"])
    return (f"<div class='metrics'>{card('haar', 'Haar Cascade', 'Viola–Jones, 2001', h)}"
            f"{card('yolo', 'YOLOv8n-face', 'CNN, WIDER FACE', y)}</div>"
            f"<div class='mfoot'>Pre-processing: <b>{ENH_NAMES.get(enh, enh)}</b> ({enh_ms:.1f} ms)"
            f" &nbsp;·&nbsp; |Δ count| = {diff}</div>")


ENH_NAMES = {"original": "None (original)", "brightness_contrast": "Brightness / contrast",
             "hist_eq": "Histogram equalization (manual)", "clahe": "CLAHE"}


# ---------------------------------------------------------------- tab 1: live
LAST_FRAME = {"img": None}     # last camera frame, so slider changes show even when paused


def live(frame, enh, scale, mn, conf, alpha, beta, clip):
    if frame is None:
        return None, None, EMPTY_BOARD
    LAST_FRAME["img"] = frame
    set_enh_params(alpha, beta, clip)
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    out, enh_ms = run_pair(bgr, enh, scale, mn, conf)
    return out["haar"]["img"], out["yolo"]["img"], scoreboard(out, enh, enh_ms)


# ---------------------------------------------------------------- tab 2: photo
def photo(path, scale, mn, conf, alpha, beta, clip):
    if not path:
        return [], pd.DataFrame(), "Upload or capture an image first."
    set_enh_params(alpha, beta, clip)
    bgr = cv2.imread(path)
    h, w = bgr.shape[:2]
    gts = ev.load_yolo_labels(config.LABELS_DIR / f"{Path(path).stem}.txt", w, h)
    gallery, rows = [], []
    for enh in config.ENHANCEMENTS:
        out, _ = run_pair(bgr, enh, scale, mn, conf, gts)
        for det in ["haar", "yolo"]:
            o = out[det]
            gallery.append((o["img"], f"{det.upper()} · {enh} · {o['n']} faces"))
            rows.append({"detector": det, "enhancement": enh, "faces": o["n"],
                         "time_ms": round(o["ms"], 1), "vs ground truth": o["extra"].strip(" |") or "-"})
    note = (f"Ground truth available: **{len(gts)} faces** (green). Blue = true positive, red = false positive."
            if gts is not None else "No ground-truth label for this file — detections only.")
    return gallery, pd.DataFrame(rows), note


# ---------------------------------------------------------------- tab 3: results
def load_results():
    r = config.RESULTS_DIR
    s = r / "summary_by_method.csv"
    if not s.exists():
        return (pd.DataFrame({"info": ["No results yet — press Run experiment."]}),
                pd.DataFrame({"info": ["—"]}), [])
    c = r / "summary_by_condition.csv"
    figs = [str(p) for p in sorted((r / "figures").glob("*.png"))]
    return pd.read_csv(s), (pd.read_csv(c) if c.exists() else pd.DataFrame()), figs


IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def dataset_status():
    imgs = [p for p in config.IMAGES_DIR.iterdir() if p.suffix.lower() in IMG_EXT]
    labelled = [p for p in imgs if (config.LABELS_DIR / f"{p.stem}.txt").exists()]
    conds = {}
    for p in labelled:
        c = p.stem.split("_")[0].lower()
        conds[c] = conds.get(c, 0) + 1
    by = ", ".join(f"{k} = {v}" for k, v in sorted(conds.items())) or "–"
    ok = "✓ ready" if len(labelled) >= 15 else "⚠ at least 15 labelled images are required"
    return (f"<div class='meta'>Dataset: {len(imgs)} images, {len(labelled)} with ground truth "
            f"&nbsp;|&nbsp; per condition: {by} &nbsp;|&nbsp; {ok}</div>")


def run_experiment_ui():
    """Runs run_experiments.py in a separate process and streams its output."""
    log = "Starting experiment ...\n"
    yield log, gr.update(), gr.update(), gr.update()
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    proc = subprocess.Popen([sys.executable, str(config.ROOT / "run_experiments.py")],
                            cwd=str(config.ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", env=env)
    for line in proc.stdout:
        log += line
        yield log, gr.update(), gr.update(), gr.update()
    proc.wait()
    log += "\nDONE." if proc.returncode == 0 else f"\nFAILED (exit code {proc.returncode})."
    yield (log, *load_results())


# ---------------------------------------------------------------- layout & style
# Scientific / laboratory style: neutral grey background, white panels with
# figure captions, blue accent, monospace numbers, formulas of each method.
SANS = '"Segoe UI", "Inter", Roboto, "Helvetica Neue", Arial, sans-serif'
MONO = '"Cascadia Mono", Consolas, "JetBrains Mono", Menlo, monospace'

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.blue, secondary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.slate, radius_size=gr.themes.sizes.radius_sm,
    font=[gr.themes.Font(f) for f in ["Segoe UI", "Inter", "Roboto", "Arial", "sans-serif"]],
    font_mono=[gr.themes.Font(f) for f in ["Cascadia Mono", "Consolas", "Menlo", "monospace"]],
).set(
    body_background_fill="#eef1f5", block_background_fill="#ffffff",
    block_border_color="#d5dbe3", block_border_width="1px", block_shadow="none",
    block_label_background_fill="#f7f9fb", block_label_text_color="#3b4a5c",
    block_title_text_weight="600", layout_gap="12px",
    button_primary_background_fill="#1f5fa8", button_primary_background_fill_hover="#174a84",
    button_primary_text_color="#ffffff", button_primary_border_color="#1f5fa8",
    button_secondary_background_fill="#ffffff", button_secondary_border_color="#b8c2ce",
    button_secondary_text_color="#1f2d3d",
    slider_color="#1f5fa8", color_accent="#1f5fa8", color_accent_soft="#e3edf8",
    checkbox_label_background_fill_selected="#1f5fa8", checkbox_label_text_color_selected="#ffffff",
    checkbox_label_border_color_selected="#1f5fa8",
)

CSS = f"""
.gradio-container {{ max-width: 100% !important; padding: 0 22px 12px !important; }}
.gradio-container, .gradio-container * {{ font-family: {SANS}; }}
footer {{ display: none !important; }}

/* header: like the title block of a paper */
#hdr {{ background: #0f2742; color: #fff; padding: 14px 22px 12px; margin: 0 -22px 4px;
        display: flex; justify-content: space-between; align-items: flex-end; gap: 24px; }}
#hdr h1 {{ font-size: 21px; font-weight: 600; margin: 0; color: #fff; letter-spacing: .1px; }}
#hdr .sub {{ font-size: 13px; color: #b9c8db; margin-top: 3px; }}
#hdr .spec {{ font-family: {MONO} !important; font-size: 11.5px; color: #d6e1ee; text-align: right;
              line-height: 1.55; white-space: nowrap; }}
#hdr .spec b {{ color: #fff; font-weight: 600; }}

/* tabs */
button[role="tab"] {{ font-size: 13.5px !important; font-weight: 500; color: #3b4a5c !important; }}
button[role="tab"][aria-selected="true"] {{ color: #1f5fa8 !important; font-weight: 600; }}
button:focus, button:focus-visible {{ outline: none !important; box-shadow: none !important; }}

/* section title */
.sec {{ font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: .08em; color: #1f5fa8;
        border-bottom: 1px solid #c9d3df; padding: 4px 0 4px; margin: 2px 0 0; display: flex;
        justify-content: space-between; }}
.sec span:last-child {{ font-weight: 400; text-transform: none; letter-spacing: 0; color: #66788c; }}

/* formula box */
.formula {{ background: #f7f9fb; border: 1px solid #d5dbe3; border-left: 3px solid #1f5fa8; border-radius: 4px;
            padding: 7px 12px; font-size: 13px; color: #1f2d3d; }}
.formula code {{ font-family: {MONO} !important; background: none; font-size: 12.5px; color: #0f2742; }}

/* metric cards */
.metrics {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
.mcard {{ background: #fff; border: 1px solid #d5dbe3; border-top: 3px solid; border-radius: 4px; padding: 10px 14px; }}
.mcard.haar {{ border-top-color: #c0392b; }}
.mcard.yolo {{ border-top-color: #1f6fb5; }}
.mh {{ font-size: 12.5px; font-weight: 600; color: #1f2d3d; }}
.mh span {{ font-weight: 400; color: #66788c; }}
.mv {{ font-family: {MONO} !important; font-size: 44px; font-weight: 600; line-height: 1.1; margin-top: 4px; color: #0f2742; }}
.ml {{ font-size: 12px; color: #66788c; }}
.mt {{ font-family: {MONO} !important; font-size: 12.5px; color: #1f2d3d; margin-top: 4px; }}
.mfoot {{ font-size: 12.5px; color: #3b4a5c; margin-top: 8px; }}
.meta {{ font-family: {MONO} !important; font-size: 12.5px; color: #1f2d3d; background: #fff;
         border: 1px solid #d5dbe3; border-radius: 4px; padding: 8px 12px; }}

/* figure captions under images */
.cap {{ font-size: 12.5px; color: #3b4a5c; margin-top: -4px; }}
.cap b {{ color: #1f2d3d; }}
.cap .h {{ color: #c0392b; font-weight: 600; }}
.cap .y {{ color: #1f6fb5; font-weight: 600; }}

/* compact buttons */
button.primary, button.secondary {{ width: auto !important; min-width: 0 !important; flex: 0 0 auto !important;
        align-self: flex-start; white-space: nowrap; padding: 8px 18px !important; font-size: 13px !important;
        font-weight: 500 !important; }}
.btnrow {{ gap: 10px !important; align-items: center; }}
button.secondary {{ border: 1px solid #b8c2ce !important; }}
textarea {{ font-family: {MONO} !important; font-size: 12px !important; }}
.note p {{ font-size: 13px !important; color: #3b4a5c; margin: 0 !important; }}
"""

# always the light look, even if Windows / the browser is in dark mode
FORCE_LIGHT = """<script>
(function () {
  const u = new URL(window.location.href);
  if (u.searchParams.get("__theme") !== "light") {
    u.searchParams.set("__theme", "light");
    window.location.replace(u.toString());
  }
})();
</script>"""

HEADER = f"""
<div id="hdr">
  <div>
    <h1>Face Detection and People Counting under Challenging Imaging Conditions</h1>
    <div class="sub">Classical Viola–Jones (Haar Cascade) vs. deep-learning YOLOv8-face · effect of image enhancement on detection accuracy</div>
  </div>
  <div class="spec">
    Detectors: <b>Haar frontalface_default</b> · <b>YOLOv8n-face</b><br>
    Match: <b>IoU ≥ {config.IOU_MATCH_THRESHOLD:.2f}</b>, one-to-one · Device: <b>{config.DEVICE.upper()}</b>
  </div>
</div>"""

FORMULAS = {
    "original": "<b>No pre-processing</b> — the captured frame <code>f(x, y)</code> is passed directly to both detectors.",
    "brightness_contrast": "<b>Linear point operation</b> &nbsp; <code>g(x, y) = α · f(x, y) + β</code>, "
                           "clipped to [0, 255] &nbsp;(α = contrast gain, β = brightness offset; applied on the Y channel)",
    "hist_eq": "<b>Histogram equalization (own implementation)</b> &nbsp; "
               "<code>T(k) = round( (cdf(k) − cdf_min) / (N − cdf_min) · 255 )</code>, &nbsp;"
               "<code>cdf(k) = Σ<sub>j≤k</sub> h(j)</code>",
    "clahe": "<b>CLAHE</b> — histogram equalization per tile (8 × 8 grid); each tile histogram is clipped at the "
             "<i>clip limit</i> before equalization and tiles are bilinearly interpolated, limiting noise amplification.",
}


def formula(enh):
    return f"<div class='formula'>{FORMULAS.get(enh, '')}</div>"


def section(no, title, right=""):
    return gr.HTML(f"<div class='sec'><span>{no}. {title}</span><span>{right}</span></div>")


def params_block(enh_radio=None):
    """Sliders for the enhancement parameters + detector settings.
    Moving a slider switches to the matching enhancement automatically."""
    with gr.Row():
        beta = gr.Slider(-50, 100, DEFAULTS["beta"], step=5, label="Brightness offset β")
        alpha = gr.Slider(0.5, 3.0, DEFAULTS["alpha"], step=0.1, label="Contrast gain α")
        clip = gr.Slider(0.5, 8.0, DEFAULTS["clip"], step=0.5, label="CLAHE clip limit")
    with gr.Row(equal_height=True):
        with gr.Column(scale=4):
            with gr.Accordion("Detector parameters", open=False):
                with gr.Row():
                    scale = gr.Slider(1.05, 1.5, DEFAULTS["scale"], step=0.05, label="Haar scaleFactor")
                    mn = gr.Slider(1, 10, DEFAULTS["mn"], step=1, label="Haar minNeighbors")
                    conf = gr.Slider(0.05, 0.9, DEFAULTS["conf"], step=0.05, label="YOLO confidence threshold")
        with gr.Column(scale=0, min_width=200):
            reset = gr.Button("Reset to config.py values", variant="secondary", size="sm")
    sliders = [scale, mn, conf, alpha, beta, clip]
    reset.click(lambda: [DEFAULTS[k] for k in ["scale", "mn", "conf", "alpha", "beta", "clip"]],
                None, sliders)
    if enh_radio is not None:
        for sl in (alpha, beta):
            sl.input(lambda: "brightness_contrast", None, enh_radio)
        clip.input(lambda: "clahe", None, enh_radio)
    return sliders


ENH_CHOICES = [("Original", "original"), ("Brightness / contrast", "brightness_contrast"),
               ("Histogram equalization", "hist_eq"), ("CLAHE", "clahe")]

with gr.Blocks(title="Face Detection: Haar Cascade vs YOLOv8-face", fill_width=True) as app:
    gr.HTML(HEADER)

    with gr.Tab("Live experiment"):
        with gr.Row(equal_height=True):
            enh_live = gr.Radio(ENH_CHOICES, value="original", label="Pre-processing method",
                                container=False, scale=0, min_width=620)
            form_live = gr.HTML(formula("original"))
        with gr.Row(equal_height=False):
            with gr.Column(scale=3, min_width=340):
                cam = gr.Image(sources=["webcam"], streaming=True, label="Input — webcam", height=160)
                with gr.Row(equal_height=True):
                    cond = gr.Dropdown(CONDITIONS, value="normal", container=False, scale=1, min_width=150)
                    cap_btn = gr.Button("📷 Capture photo", variant="primary")
                cap_msg = gr.Markdown(elem_classes="note")
                board = gr.HTML(EMPTY_BOARD)
            with gr.Column(scale=4, min_width=300):
                out_h = gr.Image(label="Haar Cascade output", height="calc(100vh - 495px)")
                gr.HTML("<div class='cap'><b>Fig. a</b> — <span class='h'>Haar Cascade</span> "
                        "(classical, hand-crafted Haar-like features + AdaBoost cascade)</div>")
            with gr.Column(scale=4, min_width=300):
                out_y = gr.Image(label="YOLOv8-face output", height="calc(100vh - 495px)")
                gr.HTML("<div class='cap'><b>Fig. b</b> — <span class='y'>YOLOv8n-face</span> "
                        "(single-stage CNN detector, pretrained on WIDER FACE)</div>")
        section("1", "Parameters", "Demo only — the experiment uses the fixed values in config.py")
        p_live = params_block(enh_live)
        cam.stream(live, [cam, enh_live, *p_live], [out_h, out_y, board],
                   stream_every=0.2, concurrency_limit=1, show_progress="hidden")
        enh_live.change(formula, enh_live, form_live)
        cap_btn.click(capture, cond, cap_msg)
        # any change of method or parameter is applied at once to the last frame
        for comp in [enh_live, *p_live]:
            comp.change(redo_last, [enh_live, *p_live], [out_h, out_y, board],
                        show_progress="hidden", trigger_mode="always_last")

    with gr.Tab("Single image analysis"):
        section("1", "Input image", "Both detectors × four pre-processing methods")
        with gr.Row():
            with gr.Column(scale=5):
                pic = gr.Image(sources=["upload", "webcam"], type="filepath", label="Input image", height=330)
            with gr.Column(scale=6):
                p_pic = params_block()
                btn = gr.Button("Run detection", variant="primary")
                note = gr.Markdown(elem_classes="note")
        section("2", "Detection results", "Left: Haar Cascade · Right: YOLOv8-face · "
                "green = ground truth, blue = TP, red = FP")
        gal = gr.Gallery(columns=2, height="auto", object_fit="contain", show_label=False)
        tbl = gr.Dataframe(label="Table 1 — Detected faces and processing time per method")
        btn.click(photo, [pic, *p_pic], [gal, tbl, note])

    with gr.Tab("Quantitative results"):
        section("1", "Experiment", "All images × 4 pre-processing methods × 2 detectors")
        status = gr.HTML(dataset_status())
        with gr.Row(elem_classes="btnrow"):
            run_btn = gr.Button("Run experiment", variant="primary")
            rbtn = gr.Button("Refresh", variant="secondary")
        run_log = gr.Textbox(label="Console output", lines=8, max_lines=14, autoscroll=True)
        section("2", "Results", "Micro-averaged over all images · IoU ≥ 0.50")
        s1 = gr.Dataframe(label="Table 2 — Precision, recall, F1, MAE and time per detector × method")
        s2 = gr.Dataframe(label="Table 3 — The same metrics per imaging condition")
        section("3", "Figures")
        figs = gr.Gallery(columns=2, height="auto", object_fit="contain", show_label=False)
        run_btn.click(run_experiment_ui, None, [run_log, s1, s2, figs]).then(dataset_status, None, status)
        rbtn.click(load_results, None, [s1, s2, figs]).then(dataset_status, None, status)
        app.load(load_results, None, [s1, s2, figs])

if __name__ == "__main__":
    app.launch(inbrowser=True, theme=THEME, css=CSS, head=FORCE_LIGHT, allowed_paths=[str(config.RESULTS_DIR)])
