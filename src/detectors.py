"""
detectors.py - the two face detectors with the same interface.

    boxes, ms = detector.detect(img_bgr)
    boxes : list of [x1, y1, x2, y2, score]   (pixel coordinates)
    ms    : detector time in milliseconds for this image
            (= the detector call only; reading the file and enhancement are NOT included)

HaarFaceDetector  -> classical Viola-Jones cascade          [MEMBER 1]
YoloFaceDetector  -> YOLOv8n-face, pretrained on WIDER FACE  [MEMBER 2]
"""
import time

import cv2
import numpy as np

import config


class HaarFaceDetector:
    name = "haar"

    def __init__(self, scale_factor=None, min_neighbors=None, min_size=None):
        self.scale_factor = scale_factor or config.HAAR_SCALE_FACTOR
        self.min_neighbors = min_neighbors or config.HAAR_MIN_NEIGHBORS
        self.min_size = tuple(min_size or config.HAAR_MIN_SIZE)
        self.cascade = cv2.CascadeClassifier(str(config.HAAR_XML))
        if self.cascade.empty():
            raise FileNotFoundError(f"Haar cascade not found: {config.HAAR_XML}")

    def params(self):
        return {"scaleFactor": self.scale_factor,
                "minNeighbors": self.min_neighbors,
                "minSize": self.min_size}

    def detect(self, img_bgr):
        t0 = time.perf_counter()
        gray = img_bgr if img_bgr.ndim == 2 else cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        rects = self.cascade.detectMultiScale(
            gray, scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors, minSize=self.min_size)
        ms = (time.perf_counter() - t0) * 1000
        # Haar gives no confidence score -> use 1.0
        boxes = [[float(x), float(y), float(x + w), float(y + h), 1.0] for (x, y, w, h) in rects]
        return boxes, ms


class YoloFaceDetector:
    name = "yolo"

    def __init__(self, conf=None, nms_iou=None, imgsz=None, device=None):
        from ultralytics import YOLO          # imported here so Haar-only use needs no torch
        self.conf = conf or config.YOLO_CONF
        self.nms_iou = nms_iou or config.YOLO_NMS_IOU
        self.imgsz = imgsz or config.YOLO_IMGSZ
        self.device = device or config.DEVICE
        if not config.YOLO_WEIGHTS.exists():
            raise FileNotFoundError(f"YOLO weights not found: {config.YOLO_WEIGHTS}")
        self.model = YOLO(str(config.YOLO_WEIGHTS))
        # warm-up run so the first timed image is not slowed by model loading
        self.model.predict(np.zeros((self.imgsz, self.imgsz, 3), np.uint8),
                           device=self.device, verbose=False)

    def params(self):
        return {"conf": self.conf, "nms_iou": self.nms_iou,
                "imgsz": self.imgsz, "device": self.device}

    def detect(self, img_bgr):
        if img_bgr.ndim == 2:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        t0 = time.perf_counter()
        res = self.model.predict(img_bgr, conf=self.conf, iou=self.nms_iou,
                                 imgsz=self.imgsz, device=self.device, verbose=False)[0]
        ms = (time.perf_counter() - t0) * 1000
        xyxy = res.boxes.xyxy.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy()
        boxes = [[*map(float, b), float(s)] for b, s in zip(xyxy, scores)]
        return boxes, ms


def build(name, **kw):
    if name == "haar":
        return HaarFaceDetector(**kw)
    if name == "yolo":
        return YoloFaceDetector(**kw)
    raise ValueError(name)
