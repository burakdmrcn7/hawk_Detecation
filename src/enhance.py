"""
enhance.py - the three image enhancement operations.          [MEMBER 1]

1) Brightness / contrast adjustment   -> written by hand with NumPy
2) Histogram equalization             -> written by hand with NumPy (NO cv2.equalizeHist)
3) CLAHE (adaptive hist. equalization) -> OpenCV's cv2.createCLAHE

Colour images: every operation is applied to the luminance (Y) channel of the
YCrCb colour space only, so brightness changes but colours do not shift.
YOLO receives the colour result; Haar converts it to grayscale itself.
"""
import cv2
import numpy as np

import config


# ------------------------------------------------------------------ helpers
def _on_luminance(img_bgr, fn):
    """Apply a grayscale function `fn` to the Y channel of a BGR image."""
    if img_bgr.ndim == 2:                       # already grayscale
        return fn(img_bgr)
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = fn(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


# ------------------------------------------------------------------ 1) brightness / contrast
def brightness_contrast_gray(gray, alpha=None, beta=None):
    """Point operation  g(x) = alpha * f(x) + beta, clipped to [0, 255].
    alpha > 1 stretches contrast, beta > 0 makes the image brighter."""
    alpha = config.BC_ALPHA if alpha is None else alpha
    beta = config.BC_BETA if beta is None else beta
    out = alpha * gray.astype(np.float32) + beta
    return np.clip(out, 0, 255).astype(np.uint8)


def brightness_contrast(img, alpha=None, beta=None):
    return _on_luminance(img, lambda g: brightness_contrast_gray(g, alpha, beta))


# ------------------------------------------------------------------ 2) histogram equalization (manual)
def hist_eq_gray_manual(gray):
    """Histogram equalization implemented from scratch.

    Steps:
      1. histogram  h(k) = number of pixels with value k   (k = 0..255)
      2. cumulative distribution  cdf(k) = h(0) + ... + h(k)
      3. mapping  T(k) = round( (cdf(k) - cdf_min) / (N - cdf_min) * 255 )
         N = total pixels, cdf_min = first non-zero cdf value
      4. replace every pixel value k by T(k)   (look-up table)
    """
    gray = np.asarray(gray, dtype=np.uint8)
    hist = np.zeros(256, dtype=np.int64)
    for value, count in zip(*np.unique(gray, return_counts=True)):   # step 1
        hist[value] = count

    cdf = np.cumsum(hist)                                             # step 2
    n_pixels = gray.size
    cdf_min = cdf[np.nonzero(cdf)[0][0]]
    if n_pixels == cdf_min:                    # flat image (one grey level) -> unchanged
        return gray.copy()

    lut = np.round((cdf - cdf_min) / (n_pixels - cdf_min) * 255)      # step 3
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return lut[gray]                                                  # step 4


def hist_eq(img):
    return _on_luminance(img, hist_eq_gray_manual)


# ------------------------------------------------------------------ 3) CLAHE
def clahe(img, clip=None, tile=None):
    """Contrast Limited Adaptive Histogram Equalization: equalizes small tiles
    separately and clips the histogram so noise is not over-amplified."""
    clip = config.CLAHE_CLIP if clip is None else clip
    tile = config.CLAHE_TILE if tile is None else tile
    op = cv2.createCLAHE(clipLimit=clip, tileGridSize=tuple(tile))
    return _on_luminance(img, op.apply)


# ------------------------------------------------------------------ dispatcher
def apply(name, img):
    if name == "original":
        return img.copy()
    if name == "brightness_contrast":
        return brightness_contrast(img)
    if name == "hist_eq":
        return hist_eq(img)
    if name == "clahe":
        return clahe(img)
    raise ValueError(f"unknown enhancement: {name}")


def verify_manual_hist_eq(gray):
    """Compare our manual equalization with OpenCV's (used only as a check,
    never in the experiments). Returns the max absolute pixel difference."""
    ours = hist_eq_gray_manual(gray)
    ref = cv2.equalizeHist(gray)
    return int(np.abs(ours.astype(int) - ref.astype(int)).max())
