"""
label_faces.py - fare ile yuz kutusu cizip YOLO etiket kaydeder.

    python label_faces.py

Her fotograf dataset/images altindan acilir. Sol tusla surukleyerek
yuzun cevresine dikdortgen cizilir. Kayit dataset/labels/<ayni ad>.txt
dosyasina su satir biciminde yazilir:

    0  x_center  y_center  width  height

Degerler 0 ile 1 arasindadir. Sinif her zaman 0 (yuz).

Tuslar
    surukle     kutu ciz
    S           kaydet (kutu yoksa bos etiket: bu karede yuz yok)
    D / N       kaydet ve sonraki fotograf
    A / P       kaydet ve onceki fotograf
    Z           son kutuyu sil
    C           tum kutulari sil
    Q / Esc     kaydet ve cik
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import config  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_W, MAX_H = 1400, 860
MIN_BOX = 8  # ekran pikseli; kazara tiklamayi yok say
CLASS_ID = 0


def list_images():
    return sorted(
        p for p in config.IMAGES_DIR.iterdir()
        if p.suffix.lower() in IMG_EXT
    )


def to_yolo(box, width, height):
    x1, y1, x2, y2 = box
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    xc = (x1 + x2) / 2.0 / width
    yc = (y1 + y2) / 2.0 / height
    return (
        min(1.0, max(0.0, xc)),
        min(1.0, max(0.0, yc)),
        min(1.0, max(0.0, bw / width)),
        min(1.0, max(0.0, bh / height)),
    )


def from_yolo(line, width, height):
    parts = line.split()
    if len(parts) < 5:
        return None
    _, xc, yc, bw, bh = (float(v) for v in parts[:5])
    x1 = (xc - bw / 2.0) * width
    y1 = (yc - bh / 2.0) * height
    x2 = (xc + bw / 2.0) * width
    y2 = (yc + bh / 2.0) * height
    return [x1, y1, x2, y2]


def load_boxes(txt_path, width, height):
    if not txt_path.exists():
        return [], False
    boxes = []
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        box = from_yolo(line, width, height)
        if box is not None:
            boxes.append(box)
    return boxes, True


def save_boxes(txt_path, boxes, width, height):
    lines = []
    for box in boxes:
        xc, yc, bw, bh = to_yolo(box, width, height)
        if bw <= 0 or bh <= 0:
            continue
        lines.append(f"{CLASS_ID} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def fit(img):
    h, w = img.shape[:2]
    scale = min(MAX_W / w, MAX_H / h, 1.0)
    if scale == 1.0:
        return img.copy(), 1.0
    view = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return view, scale


def screen_box(box, scale):
    x1, y1, x2, y2 = box
    return int(round(x1 * scale)), int(round(y1 * scale)), int(round(x2 * scale)), int(round(y2 * scale))


def original_box(x1, y1, x2, y2, scale, width, height):
    xa, xb = sorted((x1 / scale, x2 / scale))
    ya, yb = sorted((y1 / scale, y2 / scale))
    xa, xb = max(0.0, xa), min(float(width), xb)
    ya, yb = max(0.0, ya), min(float(height), yb)
    return [xa, ya, xb, yb]


class Session:
    def __init__(self):
        self.boxes = []
        self.drag = None
        self.changed = False
        self.scale = 1.0
        self.width = 1
        self.height = 1

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag = (x, y, x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drag is not None:
            self.drag = (self.drag[0], self.drag[1], x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.drag is not None:
            x1, y1, x2, y2 = self.drag
            self.drag = None
            if abs(x2 - x1) < MIN_BOX or abs(y2 - y1) < MIN_BOX:
                return
            self.boxes.append(original_box(x1, y1, x2, y2, self.scale, self.width, self.height))
            self.changed = True


def render(base, session, title):
    view, _ = fit(base)
    # fit() kopya dondurur; session.scale bu cagriyla ayni olmali
    for box in session.boxes:
        x1, y1, x2, y2 = screen_box(box, session.scale)
        cv2.rectangle(view, (x1, y1), (x2, y2), (0, 200, 0), 2)
    if session.drag is not None:
        x1, y1, x2, y2 = session.drag
        cv2.rectangle(view, (x1, y1), (x2, y2), (0, 220, 255), 2)
    bar = f"{title}   kutu={len(session.boxes)}   S kaydet  D sonraki  A onceki  Z sil  Q cik"
    cv2.rectangle(view, (0, 0), (view.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(view, bar[:120], (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return view


def main():
    images = list_images()
    if not images:
        print(f"Goruntu yok: {config.IMAGES_DIR}")
        return
    config.LABELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"{len(images)} fotograf. Etiketler: {config.LABELS_DIR}")
    print("Surukle: kutu | S: kaydet | D/N: sonraki | A/P: onceki | Z: geri al | C: temizle | Q: cik")

    session = Session()
    index = 0
    window = "Yuz etiketi"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, session.on_mouse)

    def load_index(i):
        path = images[i]
        img = cv2.imread(str(path))
        if img is None:
            print(f"Okunamadi: {path.name}")
            return None
        session.height, session.width = img.shape[:2]
        _, session.scale = fit(img)
        txt = config.LABELS_DIR / f"{path.stem}.txt"
        session.boxes, existed = load_boxes(txt, session.width, session.height)
        session.drag = None
        session.changed = False
        note = "mevcut etiket acildi" if existed else "yeni"
        print(f"[{i + 1}/{len(images)}] {path.name}  {session.width}x{session.height}  {note}  kutu={len(session.boxes)}")
        return img

    img = load_index(index)
    while img is None and index < len(images) - 1:
        index += 1
        img = load_index(index)
    if img is None:
        return

    while True:
        path = images[index]
        title = f"{index + 1}/{len(images)}  {path.name}"
        cv2.imshow(window, render(img, session, title))
        key = cv2.waitKey(20) & 0xFF
        if key == 255:
            continue

        def persist(allow_empty):
            txt = config.LABELS_DIR / f"{path.stem}.txt"
            has_work = bool(session.boxes) or session.changed or txt.exists()
            if not has_work and not allow_empty:
                print(f"  kaydedilmedi (kutu yok): {path.name}. Bos etiket icin S.")
                return False
            n = save_boxes(txt, session.boxes, session.width, session.height)
            session.changed = False
            print(f"  kaydedildi {txt.name}  ({n} yuz)")
            return True

        if key in (ord("q"), 27):
            persist(allow_empty=False)
            break
        if key in (ord("z"), 8):
            if session.boxes:
                session.boxes.pop()
                session.changed = True
            continue
        if key == ord("c"):
            session.boxes = []
            session.changed = True
            continue
        if key == ord("s"):
            persist(allow_empty=True)
            continue

        step = 0
        if key in (ord("d"), ord("n")):
            step = 1
        elif key in (ord("a"), ord("p")):
            step = -1
        if step == 0:
            continue
        persist(allow_empty=False)
        nxt = index + step
        if not 0 <= nxt < len(images):
            print("Bu yonun sonunda fotograf kalmadi.")
            continue
        loaded = load_index(nxt)
        if loaded is None:
            continue
        index = nxt
        img = loaded

    cv2.destroyAllWindows()
    print("Bitti.")


if __name__ == "__main__":
    main()
