"""Proje_Raporu.docx dosyasini deney ciktilarindan uretir.

Metrikler results/report_per_image.csv ve report_summary.csv dosyalarindan okunur.
Yer gercegi olmadigi icin kesinlik, duyarlilik, F1 ve MAE hucreleri bos birakilir.
"""
import os
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import config  # noqa: E402
import detectors  # noqa: E402
import enhance  # noqa: E402
import visualize as vz  # noqa: E402

RES = config.RESULTS_DIR
FIG = RES / "report_figures"
plt.rcParams["font.family"] = "DejaVu Sans"

ENH = [
    ("original", "Original"),
    ("brightness_contrast", "Brightness/contrast"),
    ("hist_eq", "Histogram equalization"),
    ("clahe", "CLAHE"),
]
NAVY = RGBColor(0x0F, 0x27, 0x42)


def fmt(x, nd=2):
    if x is None or (isinstance(x, float) and pd.isna(x)) or pd.isna(x):
        return "—"
    return f"{float(x):.{nd}f}"


def fmt_int(x):
    return str(int(round(float(x))))


def set_run_font(run, size=12, bold=False, italic=False, color=None, name="Times New Roman"):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    if color is not None:
        run.font.color.rgb = color


def shade(cell, hex_color):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), hex_color)
    shd.set(qn("w:val"), "clear")
    tcPr.append(shd)


def set_cell_border(cell):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = OxmlElement("w:tcBorders")
    for edge in ("top", "left", "bottom", "right"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "4")
        el.set(qn("w:color"), "BFCBD9")
        tcBorders.append(el)
    tcPr.append(tcBorders)


def add_p(doc, text, size=12, bold=False, italic=False, align="justify",
          before=0, after=8, spacing=1.5):
    p = doc.add_paragraph()
    p.alignment = {
        "justify": WD_ALIGN_PARAGRAPH.JUSTIFY,
        "center": WD_ALIGN_PARAGRAPH.CENTER,
        "left": WD_ALIGN_PARAGRAPH.LEFT,
    }[align]
    pf = p.paragraph_format
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.line_spacing = spacing
    run = p.add_run(text)
    set_run_font(run, size=size, bold=bold, italic=italic)
    return p


def add_h(doc, text, level=1):
    p = doc.add_heading(text, level=level)
    size = {1: 16, 2: 13, 3: 12}[level]
    for run in p.runs:
        set_run_font(run, size=size, bold=True, color=NAVY)
    p.paragraph_format.space_before = Pt(16 if level == 1 else 12)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.15
    return p


def add_fig(doc, path, caption, width=15.8):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    p.add_run().add_picture(str(path), width=Cm(width))
    add_p(doc, caption, size=11, italic=True, align="justify", before=0, after=10, spacing=1.15)


def add_table(doc, headers, rows, col_widths=None, font=8, header_fill="0F2742"):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.alignment = WD_ALIGN_PARAGRAPH.CENTER
    table.autofit = True
    for j, h in enumerate(headers):
        cell = table.rows[0].cells[j]
        cell.text = ""
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.space_before = Pt(0)
        run = p.add_run(h)
        set_run_font(run, size=font, bold=True, color=RGBColor(255, 255, 255))
        shade(cell, header_fill)
        set_cell_border(cell)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = table.rows[i + 1].cells[j]
            cell.text = ""
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.space_before = Pt(0)
            run = p.add_run(str(val))
            set_run_font(run, size=font, bold=False)
            if i % 2 == 1:
                shade(cell, "F4F7FB")
            set_cell_border(cell)
    if col_widths:
        for row in table.rows:
            for j, w in enumerate(col_widths):
                row.cells[j].width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return table


def lookup(summary, det, enh, col, subset="ozgun_temsilciler"):
    m = summary[(summary.subset == subset) & (summary.detector == det) & (summary.enhancement == enh)]
    return m.iloc[0][col]


def draw_flowchart(path):
    fig, ax = plt.subplots(figsize=(8.6, 11.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 22)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    def box(x, y, w, h, text, fc, ec, fs=8.4):
        rect = plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=1.2,
                             joinstyle="round", zorder=2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                zorder=3, color="#102033")

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#334155", lw=1.05), zorder=1)

    box(1.8, 20.35, 6.4, 0.95, "Start\nInput: dataset/images and dataset/labels", "#D9EAD3", "#6AA84F")
    arrow(5, 20.35, 5, 19.55)
    box(1.4, 18.15, 7.2, 1.3, "1. Audit\nUnique images (SHA-256), filename category,\nground-truth .txt present?", "#E8EEF7", "#4F81BD")
    arrow(5, 18.15, 5, 17.25)
    box(1.2, 15.85, 7.6, 1.3, "2. Pre-processing — fixed parameters for every image\nY channel of YCrCb only", "#FFF2CC", "#BF9000")
    for x in (1.35, 3.7, 6.05, 8.4):
        arrow(5, 15.85, x, 15.05)
    box(0.25, 13.55, 2.2, 1.4, "Original\nimage", "#FFF2CC", "#BF9000")
    box(2.6, 13.55, 2.2, 1.4, "Brightness/\ncontrast\ng = αf + β", "#FFF2CC", "#BF9000", 8)
    box(4.95, 13.55, 2.2, 1.4, "Histogram\nequalization\n(manual, NumPy)", "#FFF2CC", "#BF9000", 8)
    box(7.3, 13.55, 2.45, 1.4, "CLAHE\n(OpenCV)\n8×8, clip 2.0", "#FFF2CC", "#BF9000", 8)
    for x in (1.35, 3.7, 6.05, 8.5):
        arrow(x, 13.55, 5, 12.75)
    box(1.15, 11.85, 7.7, 0.8, "Every enhanced image is passed to both detectors", "#FFF8E6", "#BF9000", 8.5)
    arrow(3.2, 11.85, 2.5, 11.05)
    arrow(6.8, 11.85, 7.5, 11.05)
    box(0.3, 9.55, 4.4, 1.4, "Haar Cascade (Viola–Jones)\nscaleFactor=1.1   minNeighbors=5\nminSize=30×30", "#F4CCCC", "#CC0000", 8)
    box(5.3, 9.55, 4.4, 1.4, "YOLOv8n-face\nconf=0.25   NMS IoU=0.45\nimgsz 640, device CPU", "#CFE2F3", "#3D85C6", 8)
    arrow(2.5, 9.55, 5, 8.75)
    arrow(7.5, 9.55, 5, 8.75)
    box(1.3, 7.25, 7.4, 1.4, "3. Matching — only if a label file exists\nGreedy one-to-one, highest IoU first\nIoU ≥ 0.50 → TP; otherwise FP or FN", "#E8EEF7", "#4F81BD", 8)
    arrow(5, 7.25, 5, 6.45)
    box(1.3, 4.95, 7.4, 1.4, "4. Metrics\nPrecision, recall, F1, |count error|, MAE\nInference time and pre-processing + inference", "#E8EEF7", "#4F81BD", 8)
    arrow(5, 4.95, 5, 4.15)
    box(1.15, 2.7, 7.7, 1.35, "5. Research question\nDoes a visually better image always\ngive higher face-detection accuracy?", "#FFF2CC", "#BF9000", 8.2)
    arrow(5, 2.7, 5, 1.9)
    box(1.7, 0.7, 6.6, 1.1, "Report and figures\nMetrics are computed from the saved labels", "#D9EAD3", "#6AA84F")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def redraw_charts(summary):
    u = summary[summary.subset == "ozgun_temsilciler"].copy()
    order = [k for k, _ in ENH]
    labels = [name for _, name in ENH]

    def grouped(col, ylabel, title, fname):
        piv = u.pivot(index="enhancement", columns="detector", values=col).loc[order]
        piv.index = labels
        piv = piv.rename(columns={"haar": "Haar Cascade", "yolo": "YOLOv8-face"})
        ax = piv.plot(kind="bar", figsize=(9.2, 4.8), rot=12, color=["#c0392b", "#1f6fb5"])
        ax.set_ylabel(ylabel)
        ax.set_xlabel("")
        ax.set_title(title)
        ax.legend(frameon=False)
        for c in ax.containers:
            ax.bar_label(c, fmt="%.2f", fontsize=8)
        plt.tight_layout()
        plt.savefig(FIG / fname, dpi=130)
        plt.close()

    grouped("mean_pred_count", "Mean detection count",
            "Mean face-box count", "pred_count_by_method.png")
    grouped("mean_detector_ms", "Mean inference time (ms)",
            "Detector call only (file read and enhancement excluded)", "detector_time_by_method.png")
    grouped("mean_end_to_end_ms", "Enhancement + inference (ms)",
            "Pre-processing plus detector call", "e2e_time_by_method.png")

    q = u[u.detector == "haar"].set_index("enhancement").loc[order]
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    x = range(len(labels))
    w = 0.25
    ax.bar([i - w for i in x], q.mean_brightness, width=w, label="Mean brightness", color="#1f6fb5")
    ax.bar(list(x), q.mean_rms_contrast, width=w, label="RMS contrast", color="#c0392b")
    ax.bar([i + w for i in x], q.mean_entropy, width=w, label="Entropy (bits)", color="#1e8449")
    ax.set_xticks(list(x), labels, rotation=12)
    n_img = int(q["images"].iloc[0]) if "images" in q.columns else len(q)
    ax.set_title(f"Image statistics after pre-processing ({n_img} images)")
    ax.legend(frameon=False)
    ax.set_ylabel("Value")
    fig.tight_layout()
    fig.savefig(FIG / "image_quality_by_enhancement.png", dpi=130)
    plt.close(fig)


def example_names(uniq):
    base = uniq[uniq.enhancement == "original"]
    wide = base.pivot(index="image", columns="detector", values="pred_count")
    wide["gap"] = (wide["haar"] - wide["yolo"]).abs()
    names = []
    for name in wide.sort_values(["gap", "haar"], ascending=False).index:
        if (config.IMAGES_DIR / name).exists():
            names.append(name)
        if len(names) == 3:
            break
    return names


def redraw_detections(names):
    haar = detectors.HaarFaceDetector()
    yolo = detectors.YoloFaceDetector()
    colors = {"haar": (60, 60, 230), "yolo": (230, 140, 40)}
    titles = {"haar": "Haar Cascade", "yolo": "YOLOv8-face"}
    used = []
    for name in names:
        img = cv2.imread(str(config.IMAGES_DIR / name))
        if img is None:
            print(f"  atlandi, okunamadi: {name}")
            continue
        used.append(name)
        panels, ptitles = [], []
        for key, label in ENH:
            panels.append(enhance.apply(key, img))
            ptitles.append(label)
        fig, axes = plt.subplots(1, 4, figsize=(14.5, 4.3))
        for ax, im, title in zip(axes, panels, ptitles):
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            ax.set_title(title, fontsize=10)
            ax.axis("off")
        fig.suptitle(f"Pre-processing comparison — {name}", fontsize=12)
        fig.tight_layout()
        fig.savefig(FIG / f"enh_compare_{name}.png", dpi=120)
        plt.close(fig)
        for key, label in (("original", "Original"), ("clahe", "CLAHE")):
            e = enhance.apply(key, img)
            tiles = []
            for det in (haar, yolo):
                preds, ms = det.detect(e)
                tiles.append(vz.draw(
                    e, None, preds, None,
                    f"{titles[det.name]} | {len(preds)} boxes | {ms:.0f} ms",
                    pred_color=colors[det.name]))
            fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
            for ax, im, det in zip(axes, tiles, (haar, yolo)):
                ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                ax.set_title(titles[det.name], fontsize=11)
                ax.axis("off")
            fig.suptitle(f"{label} — {name}", fontsize=12)
            fig.tight_layout()
            fig.savefig(FIG / f"det_{key}_{name}.png", dpi=120)
            plt.close(fig)
    return used


def configure(doc):
    sec = doc.sections[0]
    sec.page_width = Cm(21.0)
    sec.page_height = Cm(29.7)
    sec.left_margin = Cm(2.0)
    sec.right_margin = Cm(2.0)
    sec.top_margin = Cm(1.8)
    sec.bottom_margin = Cm(1.8)
    sec.header_distance = Cm(0.6)
    sec.footer_distance = Cm(0.6)
    header = sec.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = header.add_run("Digital Image Processing  ·  Project Report")
    set_run_font(run, size=9, italic=True, color=RGBColor(0x3B, 0x4A, 0x5C))
    footer = sec.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = footer.add_run("Face detection under challenging imaging  ·  Haar Cascade and YOLOv8-face")
    set_run_font(run, size=9, color=RGBColor(0x3B, 0x4A, 0x5C))
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(12)
    style._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")


def build():
    audit = pd.read_json(RES / "dataset_audit.json", typ="series")
    summary = pd.read_csv(RES / "report_summary.csv")
    per = pd.read_csv(RES / "report_per_image.csv")
    settings = pd.read_json(RES / "report_settings.json", typ="series")
    uniq = per[per.unique_rep].copy()

    FIG.mkdir(parents=True, exist_ok=True)
    print("Akış şeması ve grafikler yazılıyor...")
    draw_flowchart(FIG / "akis_semasi.png")
    redraw_charts(summary)
    examples = example_names(uniq)
    if os.environ.get("REPORT_SKIP_DET") == "1":
        print("Tespit görselleri yeniden üretilmedi.")
    else:
        print("Örnek tespit görselleri yenileniyor...")
        examples = redraw_detections(examples)

    def S(det, enh, col):
        return lookup(summary, det, enh, col)

    doc = Document()
    configure(doc)

    add_p(doc, "Digital Image Processing", size=14, bold=True, align="center", before=18, after=2, spacing=1.0)
    add_p(doc, "PROJECT REPORT", size=20, bold=True, align="center", before=0, after=8, spacing=1.0)
    add_p(doc, "Face Detection and People Counting under Challenging Imaging Conditions",
          size=14, bold=True, align="center", before=4, after=2, spacing=1.0)
    add_p(doc, "Classical Haar Cascade (Viola–Jones) compared with YOLOv8-face\nunder four pre-processing conditions",
          size=12, italic=True, align="center", before=0, after=10, spacing=1.15)
    add_p(doc, "Research question. Does a visually better image always give higher face-detection accuracy?",
          size=12, align="center", before=6, after=10, spacing=1.15)
    add_p(doc, "Group identifiers in README are still placeholders (No1, No2). Student numbers should be filled in before submission. No personal names were invented for this report.",
          size=11, align="center", before=8, after=2, spacing=1.15)
    add_p(doc, "23 November 2026", size=12, align="center", before=2, after=6, spacing=1.0)
    add_p(doc, "Machine: " + str(settings["machine"]) + ". Device: CPU. Python " + str(settings["python"]) + ".",
          size=10, italic=True, align="center", before=8, after=4, spacing=1.15)

    add_h(doc, "Abstract", 1)
    add_p(doc,
          "Two face detectors were compared under identical fixed parameters: the OpenCV Haar Cascade (Viola–Jones) and YOLOv8n-face trained on WIDER FACE. Every image was passed through four conditions: the original frame, brightness/contrast adjustment, a hand-written histogram equalization, and CLAHE. The aim is to measure how pre-processing changes image statistics and detection output, and whether a visually stronger image also raises detection accuracy.")
    n_img = int(S("haar", "original", "images"))
    gt_faces = int(uniq.gt_count.sum() / 8)  # each image is repeated over 4 enhancements x 2 detectors
    add_p(doc,
          f"The dataset contains {int(audit['n_files'])} image files. SHA-256 shows that all of them are unique. Each image has a YOLO label file in dataset/labels ({int(audit['n_label_files'])} files). Every one of those files is empty, so the reference face count is zero. Two filenames use the normal_ prefix. No file uses low_, backlight_, small_, or occlusion_. The other thirteen names do not carry a condition prefix.")
    add_p(doc,
          "Matching used a one-to-one rule with IoU at least 0.50. With no reference box, no prediction can match. For every detector and every enhancement, TP = 0 and FN = 0, and FP equals the number of boxes. Precision is therefore 0. Recall is undefined when TP + FN = 0; evaluate.py writes 0 in that case, and the tables keep that value. F1 is 0. MAE is the mean of |predicted count − 0|, so it equals the mean box count. Two times are reported. detector_ms is the detector call only. end_to_end_ms is pre-processing plus that call. Disk reading and model loading are excluded.")
    add_p(doc,
          f"On the original images, Haar produced a mean of {fmt(S('haar','original','mean_pred_count'), 3)} boxes, with TP = {fmt_int(S('haar','original','TP'))}, FP = {fmt_int(S('haar','original','FP'))}, FN = {fmt_int(S('haar','original','FN'))}, precision = {fmt(S('haar','original','precision'), 3)}, recall = {fmt(S('haar','original','recall'), 3)}, F1 = {fmt(S('haar','original','F1'), 3)}, and MAE = {fmt(S('haar','original','MAE'), 3)}. YOLOv8-face produced a mean of {fmt(S('yolo','original','mean_pred_count'), 3)} boxes, FP = {fmt_int(S('yolo','original','FP'))}, F1 = {fmt(S('yolo','original','F1'), 3)}, and MAE = {fmt(S('yolo','original','MAE'), 3)}. CLAHE raises entropy to {fmt(S('haar','clahe','mean_entropy'), 3)} bits and raises the Haar FP total from {fmt_int(S('haar','original','FP'))} to {fmt_int(S('haar','clahe','FP'))}. F1 stays at 0. Image statistics change while the accuracy score does not, because there is no reference box to match. The total number of reference faces is {gt_faces}.")

    add_h(doc, "1. Introduction", 1)
    add_p(doc,
          "Face detection is the problem of placing a box on every face in an image. People counting then uses the number of those boxes. The classical Viola–Jones method uses Haar-like features and a cascade of AdaBoost classifiers. A single-stage convolutional network learns the same task end to end. The second detector in this project is YOLOv8n-face, run through the Ultralytics framework and fine-tuned for faces.")
    add_p(doc,
          "The image-processing part of the course is the point and histogram operations applied before the detector. Brightness and contrast form a linear point operation. Histogram equalization remaps grey levels through the cumulative distribution. CLAHE applies the same idea inside tiles and clips the histogram, which limits noise amplification. The research question is whether these operations raise detection accuracy when they make the image easier to look at.")
    add_p(doc,
          "Answering that question requires ground truth. A box counts as correct only when its IoU with a reference box is at least 0.50. Label files exist for this run and matching was executed. The files are empty, so the reference set is empty. The report copies TP, FP, FN, precision, recall, F1, and MAE from the CSV produced against that empty reference.")

    add_h(doc, "2. Dataset audit", 1)
    add_p(doc,
          "Images are in dataset/images and labels are in dataset/labels. A label line is in YOLO form: class, normalized center x, center y, width, and height. Every image has a .txt file with the same stem. All fifteen files exist, and none of them contains a line. The loader treats an empty file as zero reference boxes.")
    add_p(doc,
          f"There are {int(audit['n_files'])} images and {int(audit['n_unique_sha256'])} unique SHA-256 digests. The threshold of fifteen unique images is met. The category is the filename token before the first underscore. The expected prefixes are normal, low, backlight, small, and occlusion. Resolutions are not uniform: files range from 768×1024 to 2048×1536. There are no duplicate groups.")

    add_table(doc,
              ["Category", "Expected prefix", "Files", "Five-image requirement"],
              [
                  ["Normal light", "normal_", "2", "Not met"],
                  ["Low light", "low_", "0", "Not met"],
                  ["Backlight / small face / occlusion", "backlight_, small_, occlusion_", "0", "Not met"],
                  ["No condition prefix (IMG_ and WhatsApp)", "—", "13", "Not assigned to a category"],
              ],
              font=8)
    add_p(doc, "Table 1. Lighting and difficulty distribution read from filenames. The requirement is at least five images in each category.",
          size=11, italic=True, before=0, after=8, spacing=1.15)
    add_p(doc,
          "Warning. Fifteen label files were saved and each of them contains zero faces. The TP, FP, FN, precision, recall, F1, and MAE values below are relative to that empty reference. Faces that are visible in a photograph do not change these numbers. Matching uses only the boxes written in the txt files.")
    warn = doc.add_table(rows=1, cols=1)
    cell = warn.rows[0].cells[0]
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(0)
    run = p.add_run("Label files: 15. Reference face boxes: 0. The five-image quota per category was not met. Metrics were computed against an empty reference: TP = 0, FN = 0, FP = number of boxes.")
    set_run_font(run, size=11, bold=True)
    shade(cell, "FDEBD0")
    set_cell_border(cell)
    add_p(doc, "", size=6, after=4, spacing=1.0)

    add_h(doc, "3. Method", 1)
    add_h(doc, "3.1 Pipeline", 2)
    add_p(doc,
          "The pipeline is the same for every image and is bound to the constants in config.py. The image is read. The selected enhancement is applied. The same result is given to both Haar and YOLOv8-face. If a label file exists, boxes are matched greedily and one to one. Parameters are not changed from image to image.")
    add_fig(doc, FIG / "akis_semasi.png",
            "Figure 1. Experiment flow. All four enhancements, with the same fixed parameters, are sent to both detectors. IoU matching produces TP, FP, and FN from the saved label file.")

    add_h(doc, "3.2 Brightness and contrast (hand-written point operation)", 2)
    add_p(doc,
          "A colour image is converted to YCrCb. The operation is applied only to the Y channel. Cr and Cb stay unchanged, so hue does not shift. For a grey value f the output is")
    add_p(doc, "g(x, y) = clip( α · f(x, y) + β ,  0,  255 )",
          align="center", italic=True, before=2, after=6, spacing=1.15)
    add_p(doc,
          "In the experiment, α = 1.5 and β = 30. Because α is greater than one, contrast is stretched. Because β is positive, brightness increases. Values outside 0 and 255 are clipped. The implementation is NumPy code in brightness_contrast_gray (src/enhance.py). It does not call a ready-made brightness function. Clipping can saturate bright regions and pile the histogram at the two ends. Section 5 measures that effect as a drop in entropy.")

    add_h(doc, "3.3 Histogram equalization (hand-written transform)", 2)
    add_p(doc,
          "Histogram equalization is also applied on the Y channel, and it does not call cv2.equalizeHist. The steps in hist_eq_gray_manual are as follows.")
    add_p(doc, "1. Histogram. For k = 0, 1, …, 255, h(k) is the number of pixels with value k. Counts come from np.unique.",
          align="left", after=2, spacing=1.15)
    add_p(doc, "2. Cumulative distribution. cdf(k) = h(0) + h(1) + … + h(k).",
          align="left", after=2, spacing=1.15)
    add_p(doc, "3. Limits. N is the number of pixels and cdf_min is the first non-zero cumulative value. If the image has a single grey level, N = cdf_min and the image is returned unchanged.",
          align="left", after=2, spacing=1.15)
    add_p(doc, "4. Look-up table.",
          align="left", after=2, spacing=1.15)
    add_p(doc, "T(k) = round( (cdf(k) − cdf_min) / (N − cdf_min) · 255 )",
          align="center", italic=True, before=0, after=6, spacing=1.15)
    add_p(doc, "5. Every pixel value k is replaced by T(k). The table is clipped to 0 and 255.",
          align="left", after=8, spacing=1.15)
    add_p(doc,
          f"At the start of the experiment, IMG_2805.JPEG was converted to grey and the manual mapping was compared with cv2.equalizeHist. The maximum absolute pixel difference was {int(settings['manual_hist_eq_max_abs_diff_vs_opencv'])}. That check is not the experiment. The experiment equalizes the Y channel, not the full grey image, and it does not call cv2.equalizeHist.")

    add_h(doc, "3.4 CLAHE", 2)
    add_p(doc,
          "CLAHE was not written by hand in this project. OpenCV’s cv2.createCLAHE is used. The clip limit is 2.0 and the tile grid is 8×8. Inside each tile the histogram is clipped, the clipped mass is redistributed, the tile is equalized, and neighbouring tiles are blended with bilinear interpolation. The aim is to stop global equalization from amplifying noise in flat regions. The operation is again applied only to Y. Brightness/contrast and histogram equalization are project code. CLAHE is a library function.")

    add_h(doc, "3.5 Detectors", 2)
    add_p(doc,
          "HaarFaceDetector loads haarcascade_frontalface_default.xml from the OpenCV 4.x distribution. scaleFactor = 1.1, minNeighbors = 5, and minSize = (30, 30) pixels. scaleFactor is the step of the image pyramid. minNeighbors is the number of overlapping detections required before a window is kept as a face. A larger value reduces false boxes and can increase misses. The cascade does not return a confidence score. The code stores 1.0 for every box. A colour input is converted to grey inside the timed call, so that conversion is part of the inference time.")
    add_p(doc,
          "YoloFaceDetector opens models/yolov8n-face.pt through the Ultralytics YOLO class. The confidence threshold is 0.25, the NMS IoU threshold is 0.45, the network input is 640 pixels, and the device is cpu. The constructor runs one warm-up prediction on a black 640×640 image before the timed loop, so the first image does not include weight loading. The model is a YOLOv8n variant trained for faces on WIDER FACE. That training was not repeated in this project.")

    add_h(doc, "3.6 Matching and accuracy measures", 2)
    add_p(doc,
          "The protocol is in src/evaluate.py and it ran for this experiment. IoU is the area of intersection divided by the area of union of boxes A and B. Predictions and reference boxes are paired from the highest IoU downward, and a pair is kept only when IoU is at least 0.50. Each prediction and each reference is used at most once. A matched prediction is a TP, an unmatched prediction is an FP, and an unmatched reference is an FN. If the reference list is empty there is no matched pair: TP = 0, FN = 0, and FP equals the number of predictions.")
    add_p(doc, "Precision = TP / (TP + FP),    Recall = TP / (TP + FN),    F1 = 2 · Precision · Recall / (Precision + Recall).",
          align="center", italic=True, before=2, after=6, spacing=1.15)
    add_p(doc,
          "If a denominator is zero, the measure is stored as 0. That rule also covers recall: when TP + FN = 0, recall is written as 0. Precision and recall are not the mean of per-image ratios. They are computed from the summed TP, FP, and FN (micro average). The absolute count error of one image is |predicted count − reference count|. MAE is the arithmetic mean of that error. Because the reference count is 0, the error of an image equals its predicted count.")

    add_h(doc, "3.7 Definition of the reported times", 2)
    add_p(doc,
          "The two times are reported together and are not interchangeable. detector_ms is the wall-clock time of the detect call. For Haar it includes the grey conversion and the cascade search. For YOLOv8-face it is the model.predict call. enhance_ms is pre-processing only. end_to_end_ms is the sum of the two. cv2.imread, loading the model, and the warm-up pass are excluded. Timing uses time.perf_counter on one thread and on the CPU. No randomness is used (SEED = 0).")

    add_h(doc, "4. Experimental setup", 1)
    add_p(doc,
          f"run_report_eval.py applied four enhancements and two detectors to each of the {n_img} images, using the values in config.py. The number of condition passes is {n_img} × 4 × 2 = {n_img * 8}. Library versions: OpenCV {settings['opencv']}, NumPy {settings['numpy']}, PyTorch {settings['torch']}, Ultralytics {settings['ultralytics']}, pandas {settings['pandas']}. Machine: {settings['machine']}.")
    add_p(doc,
          "Image statistics are computed on the grey equivalent of the Y channel. Mean brightness is the pixel mean, RMS contrast is the standard deviation, and entropy is −Σ p log2 p over the non-zero histogram probabilities. These three numbers are an objective stand-in for “looking better.” High contrast or high entropy does not by itself mean a preferred image. Histogram equalization can reveal detail and still look unnatural.")

    add_h(doc, "5. Results", 1)
    add_h(doc, "5.1 Image statistics", 2)
    add_p(doc,
          f"Table 2 is the mean over {n_img} images. On the original frames, mean brightness is {fmt(S('haar','original','mean_brightness'), 2)}, RMS contrast is {fmt(S('haar','original','mean_rms_contrast'), 2)}, and entropy is {fmt(S('haar','original','mean_entropy'), 3)} bits. Brightness/contrast raises brightness to {fmt(S('haar','brightness_contrast','mean_brightness'), 2)}, sets contrast to {fmt(S('haar','brightness_contrast','mean_rms_contrast'), 2)}, and lowers entropy to {fmt(S('haar','brightness_contrast','mean_entropy'), 3)} bits. Histogram equalization reaches the highest contrast, {fmt(S('haar','hist_eq','mean_rms_contrast'), 2)}. CLAHE reaches the highest entropy, {fmt(S('haar','clahe','mean_entropy'), 3)} bits, and keeps brightness closer to the original at {fmt(S('haar','clahe','mean_brightness'), 2)}.")

    qrows = []
    for key, label in ENH:
        qrows.append([
            label,
            fmt(S("haar", key, "mean_brightness"), 2),
            fmt(S("haar", key, "mean_rms_contrast"), 2),
            fmt(S("haar", key, "mean_entropy"), 3),
        ])
    add_table(doc, ["Pre-processing", "Mean brightness", "RMS contrast", "Entropy (bits)"], qrows, font=10)
    add_p(doc, f"Table 2. Mean image statistics after pre-processing on {n_img} images. The statistic belongs to the image. It is read from the Haar rows because both detectors receive the same enhanced image.",
          size=11, italic=True, spacing=1.15)
    add_fig(doc, FIG / "image_quality_by_enhancement.png",
            "Figure 2. Plot of Table 2. Brightness/contrast raises brightness and lowers entropy. Histogram equalization raises contrast. CLAHE raises entropy the most.")

    add_h(doc, "5.2 Detection counts and accuracy", 2)
    add_p(doc,
          f"Table 3 reports micro-averaged accuracy on {n_img} images. The standard deviation uses the population definition (ddof = 0). Every row has TP = 0 and FN = 0. FP equals the box total. On the original images Haar produced {fmt_int(S('haar','original','sum_pred_count'))} boxes and YOLOv8-face produced {fmt_int(S('yolo','original','sum_pred_count'))}. CLAHE raises the Haar box total to {fmt_int(S('haar','clahe','sum_pred_count'))}. Precision and F1 stay at 0. The YOLOv8-face box totals are {fmt_int(S('yolo','original','sum_pred_count'))} (original), {fmt_int(S('yolo','brightness_contrast','sum_pred_count'))} (brightness/contrast), {fmt_int(S('yolo','hist_eq','sum_pred_count'))} (histogram equalization), and {fmt_int(S('yolo','clahe','sum_pred_count'))} (CLAHE).")

    crows = []
    for key, label in ENH:
        for det, dlabel in (("haar", "Haar Cascade"), ("yolo", "YOLOv8-face")):
            crows.append([
                dlabel, label, fmt_int(S(det, key, "images")),
                fmt_int(S(det, key, "sum_pred_count")),
                fmt(S(det, key, "mean_pred_count"), 3),
                fmt(S(det, key, "std_pred_count"), 3),
                fmt_int(S(det, key, "TP")),
                fmt_int(S(det, key, "FP")),
                fmt_int(S(det, key, "FN")),
                fmt(S(det, key, "precision"), 3),
                fmt(S(det, key, "recall"), 3),
                fmt(S(det, key, "F1"), 3),
                fmt(S(det, key, "MAE"), 3),
            ])
    add_table(doc,
              ["Detector", "Pre-processing", "N", "Box total", "Mean", "SD",
               "TP", "FP", "FN", "Precision", "Recall", "F1", "MAE"],
              crows, font=7)
    add_p(doc, f"Table 3. Micro average on {n_img} images at IoU ≥ 0.50. Because the reference box count is 0, TP and FN are zero and FP equals the box total. Recall is stored as 0 because TP + FN = 0. MAE = mean |prediction − 0|.",
          size=11, italic=True, spacing=1.15)
    add_fig(doc, FIG / "pred_count_by_method.png",
            "Figure 3. Mean box counts from Table 3. These counts are also the FP counts. Haar reaches its highest mean under CLAHE. F1 is 0 in all four conditions. A higher box count is not a higher precision.")

    # per image table
    wide_rows = []
    images = sorted(uniq.image.unique())
    for name in images:
        stem = Path(name).stem.replace("WhatsApp Image 2026-11-23 at ", "WA ")
        row = [stem]
        sub = uniq[uniq.image == name]
        for key, _ in ENH:
            for det in ("haar", "yolo"):
                val = sub[(sub.enhancement == key) & (sub.detector == det)].pred_count.iloc[0]
                row.append(fmt_int(val))
        wide_rows.append(row)
    add_table(doc,
              ["Image", "O Haar", "O YOLO", "B Haar", "B YOLO", "H Haar", "H YOLO", "C Haar", "C YOLO"],
              wide_rows, font=8)
    add_p(doc, "Table 4. Box count per image. Columns: O original, B brightness/contrast, H histogram equalization, C CLAHE. WA shortens the prefix WhatsApp Image 2026-11-23 at. Because the reference is 0, the absolute count error of each cell equals that cell’s box count.",
          size=11, italic=True, spacing=1.15)
    orig = uniq[uniq.enhancement == "original"].pivot(index="image", columns="detector", values="pred_count")
    gap_mean = float((orig["haar"] - orig["yolo"]).abs().mean())
    gap_name = (orig["haar"] - orig["yolo"]).abs().sort_values(ascending=False).index[0]
    add_p(doc,
          f"On the original images the mean absolute difference |Haar − YOLOv8-face| is {fmt(gap_mean, 3)} boxes. The largest gap is {gap_name}: Haar {fmt_int(orig.loc[gap_name, 'haar'])}, YOLOv8-face {fmt_int(orig.loc[gap_name, 'yolo'])}. The empty reference does not split that gap into misses and false boxes. Both detectors’ boxes are FP.")

    add_h(doc, "5.3 Timing", 2)
    add_p(doc,
          f"Table 5 separates inference time from the sum of pre-processing and inference. On the original images the Haar call averages {fmt(S('haar','original','mean_detector_ms'), 2)} ms and the YOLOv8-face call averages {fmt(S('yolo','original','mean_detector_ms'), 2)} ms. After CLAHE, Haar rises to {fmt(S('haar','clahe','mean_detector_ms'), 2)} ms. Enhancement times are shared by the two detector rows: original {fmt(S('haar','original','mean_enhance_ms'), 2)} ms, brightness/contrast {fmt(S('haar','brightness_contrast','mean_enhance_ms'), 2)} ms, histogram equalization {fmt(S('haar','hist_eq','mean_enhance_ms'), 2)} ms, and CLAHE {fmt(S('haar','clahe','mean_enhance_ms'), 2)} ms. For YOLOv8-face the histogram-equalization total is {fmt(S('yolo','hist_eq','mean_end_to_end_ms'), 2)} ms. Most of that increase is the enhancement, not the network.")

    trows = []
    for key, label in ENH:
        for det, dlabel in (("haar", "Haar Cascade"), ("yolo", "YOLOv8-face")):
            trows.append([
                dlabel, label,
                fmt(S(det, key, "mean_enhance_ms"), 2),
                fmt(S(det, key, "mean_detector_ms"), 2),
                fmt(S(det, key, "mean_end_to_end_ms"), 2),
            ])
    add_table(doc,
              ["Detector", "Pre-processing", "Enhancement (ms)", "Inference (ms)", "Enhancement + inference (ms)"],
              trows, font=9)
    add_p(doc, f"Table 5. Mean time on {n_img} images. The inference column is detector_ms. The last column is end_to_end_ms. Disk reading and model loading are excluded.",
          size=11, italic=True, spacing=1.15)
    add_fig(doc, FIG / "detector_time_by_method.png",
            "Figure 4. Mean time of the detector call only. At these resolutions Haar is several times slower than the YOLOv8-face call. CLAHE lengthens the Haar call.")
    add_fig(doc, FIG / "e2e_time_by_method.png",
            f"Figure 5. Sum of pre-processing and inference. Histogram equalization raises the total. On the YOLOv8-face side that enhancement takes {fmt(S('yolo','hist_eq','mean_enhance_ms'), 2)} ms.")

    add_h(doc, "5.4 Example images", 2)
    add_p(doc,
          "The figures below are the original frames on which the two detectors disagree most in box count. No green ground-truth box is drawn, because the reference count is zero. The red tone is Haar and the blue tone is YOLOv8-face. Against the empty label, every drawn box is an FP. The millisecond stamp belongs to the drawing pass. The averages are in Table 5.")
    fig_no = 6
    for name in examples:
        h_n = int(orig.loc[name, "haar"]) if name in orig.index else 0
        y_n = int(orig.loc[name, "yolo"]) if name in orig.index else 0
        add_fig(doc, FIG / f"enh_compare_{name}.png",
                f"Figure {fig_no}. Four pre-processing results for {name}.",
                width=16.2)
        fig_no += 1
        add_fig(doc, FIG / f"det_original_{name}.png",
                f"Figure {fig_no}. {name}, original image. Haar {h_n} boxes, YOLOv8-face {y_n} boxes. The reference count is 0, so TP = 0 and the boxes are FP.",
                width=15.5)
        fig_no += 1
        add_fig(doc, FIG / f"det_clahe_{name}.png",
                f"Figure {fig_no}. {name}, CLAHE. A change in box count does not change F1. F1 is 0 in this experiment.",
                width=15.5)
        fig_no += 1

    add_h(doc, "6. Discussion of the research question", 1)
    add_p(doc,
          "The question is whether a visually better image always gives higher face-detection accuracy. Appearance was tracked by brightness, RMS contrast, and entropy. Accuracy is precision, recall, and F1. In Table 3 all three are 0 for every enhancement.")
    add_p(doc,
          f"Brightness/contrast lowers entropy from {fmt(S('haar','original','mean_entropy'), 3)} bits to {fmt(S('haar','brightness_contrast','mean_entropy'), 3)} bits and produces the brightest images (mean brightness {fmt(S('haar','brightness_contrast','mean_brightness'), 2)}). The Haar FP total moves from {fmt_int(S('haar','original','FP'))} to {fmt_int(S('haar','brightness_contrast','FP'))}, and the YOLOv8-face FP total moves from {fmt_int(S('yolo','original','FP'))} to {fmt_int(S('yolo','brightness_contrast','FP'))}. F1 stays at 0 for both detectors.")
    add_p(doc,
          f"Histogram equalization reaches the highest contrast, {fmt(S('haar','hist_eq','mean_rms_contrast'), 2)}. Haar FP becomes {fmt_int(S('haar','hist_eq','FP'))} and YOLOv8-face FP becomes {fmt_int(S('yolo','hist_eq','FP'))}. CLAHE reaches the highest entropy, {fmt(S('haar','clahe','mean_entropy'), 3)} bits, and the highest Haar FP total, {fmt_int(S('haar','clahe','FP'))}. YOLOv8-face FP is {fmt_int(S('yolo','clahe','FP'))}. Entropy and box count can rise together. Accuracy does not: there is no reference box to match, so an extra box only increases FP.")
    add_p(doc,
          "On this dataset the claim is not supported. Image statistics change while F1 stays at 0. The result does not generalize to a ground truth in which faces were actually drawn. If the label files are rewritten with face boxes, the same fixed parameters should be run again. Until then, Table 3 is the numerical result of an empty reference.")

    add_h(doc, "7. Conclusion", 1)
    add_p(doc,
          f"Four enhancements and two detectors were run with fixed parameters on {n_img} unique images. On the original images the mean inference time is {fmt(S('haar','original','mean_detector_ms'), 2)} ms for Haar and {fmt(S('yolo','original','mean_detector_ms'), 2)} ms for YOLOv8-face. Haar produced {fmt_int(S('haar','original','FP'))} FP and YOLOv8-face produced {fmt_int(S('yolo','original','FP'))} FP. CLAHE raises the Haar FP count to {fmt_int(S('haar','clahe','FP'))}. In every condition TP = 0, FN = 0, precision = 0, and F1 = 0. MAE equals the mean box count.")
    add_p(doc,
          "The threshold of fifteen unique images is met. The threshold of five images in each category is not: two files use the normal_ prefix, low light has 0, and backlight / small face / occlusion has 0. All fifteen label files are empty. F1 does not rise when the visual statistics change. This is the result of scoring against an empty reference.")

    add_h(doc, "8. AI assistance and external sources", 1)
    add_p(doc,
          "The repository README states that Anthropic Claude was used to help prepare the code structure and the documentation, and that the group members reviewed the code. That sentence is quoted from the README. This report does not independently verify that review.")
    add_p(doc,
          "This report, the evaluation script run_report_eval.py, and the document builder build_report.py were written with Grok in Cursor. The experimental numbers were not asked of the model. They were read from results/report_summary.csv and results/report_per_image.csv. Precision, recall, F1, and MAE are the values in those files.")
    add_p(doc,
          "The hand-written pieces are brightness/contrast and histogram equalization. CLAHE, the Haar cascade, the YOLO prediction, and the remaining image utilities (colour conversion, file reading, drawing) are library code. An English flowchart is also stored as flowchart.png. Figure 1 is a redraw of the same flow for this experiment.")
    add_table(doc,
              ["Component", "Version or file", "Source"],
              [
                  ["Haar cascade", "haarcascade_frontalface_default.xml", "OpenCV 4.x, opencv/opencv"],
                  ["YOLOv8n-face weights", "yolov8n-face.pt", "akanametov/yolo-face, trained on WIDER FACE"],
                  ["YOLO framework", "Ultralytics " + str(settings["ultralytics"]), "github.com/ultralytics/ultralytics"],
                  ["CLAHE", "cv2.createCLAHE", "OpenCV; Zuiderveld, 1994"],
                  ["Label files", "dataset/labels, 15 empty .txt files", "Saved with label_faces.py"],
              ],
              font=8)
    add_p(doc, "Table 6. External components used in the experiment. The weights were not retrained in this project.",
          size=11, italic=True, spacing=1.15)
    add_p(doc,
          "Who took the photographs, and under what consent, cannot be verified from the code. The README states that the images are of group members, that the subjects are adults, that the images are used only for face detection, and that no identity recognition is performed. The figures in this report are frames from this folder.")

    add_h(doc, "References", 1)
    refs = [
        "Gonzalez, R. C. and Woods, R. E. Digital Image Processing. Chapters on histogram equalization and point operations.",
        "Zuiderveld, K. (1994). Contrast limited adaptive histogram equalization. Graphics Gems IV. Academic Press.",
        "Viola, P. and Jones, M. (2001). Rapid object detection using a boosted cascade of simple features. Proceedings of CVPR.",
        "Yang, S., Luo, P., Loy, C. C., and Tang, X. (2016). WIDER FACE: A face detection benchmark. Proceedings of CVPR.",
        "Jocher, G., Chaurasia, A., and Qiu, J. Ultralytics YOLO. https://github.com/ultralytics/ultralytics",
        "akanametov. YOLOv8-face. https://github.com/akanametov/yolo-face",
        "OpenCV. Haar cascade files. https://github.com/opencv/opencv/tree/4.x/data/haarcascades",
        "makesense.ai. Export of object-detection labels in YOLO format.",
    ]
    for ref in refs:
        add_p(doc, ref, size=11, align="left", after=4, spacing=1.15)

    add_h(doc, "Appendix. Output files", 1)
    add_p(doc,
          "results/dataset_audit.json stores the dataset audit. results/report_per_image.csv stores one row per image, enhancement, and detector. results/report_summary.csv stores the aggregates. results/report_settings.json stores parameters and library versions. Figures are in results/report_figures. The same scripts can be run again after the label files are rewritten.")

    out = ROOT / "Proje_Raporu.docx"
    doc.save(out)
    print(f"Yazıldı: {out}")


if __name__ == "__main__":
    build()
