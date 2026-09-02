"""OpenCV preprocessing pipeline that runs *before* the I-JEPA backbone sees a leaf
photo. This does not touch the model -- it is classical image processing meant to
make disease symptoms (chlorotic/necrotic spots, rust pustules, mildew film) more
visually distinct and to damp down the exact background clutter (soil, glare,
other leaves) the original project's design doc calls out as a failure mode for
naive CNNs.

Steps:
  1. CLAHE (contrast-limited adaptive histogram equalization) on the L channel in
     LAB space -- boosts local contrast in each neighborhood without blowing out
     global exposure, so subtle lesions stop being washed out by bright leaf sheen.
  2. Light edge-preserving denoise -- removes sensor/JPEG noise that would otherwise
     show up as spurious high-frequency patch-embedding variation.
  3. Soft background suppression -- an HSV leaf-colored mask (green through
     yellow-brown, to keep diseased tissue) is used to gently darken/desaturate
     whatever falls outside it, rather than pretending backgrounds are irrelevant.
     Soft (alpha-blended), not a hard cutout, so imperfect segmentation never
     destroys real leaf-edge detail.
  4. Unsharp-mask sharpening -- restores lesion-edge and venation detail lost to
     steps 1-2.

Usage:
    from enhance_cv import enhance_pil
    enhanced = enhance_pil(pil_image)

CLI:
    python src/enhance_cv.py --input leaf.jpg --output leaf_enhanced.jpg
"""
import argparse

import cv2
import numpy as np
from PIL import Image


def _clahe_lab(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def _denoise(bgr: np.ndarray) -> np.ndarray:
    return cv2.bilateralFilter(bgr, d=5, sigmaColor=40, sigmaSpace=40)


def _soft_background_suppress(bgr: np.ndarray, strength: float = 0.35) -> np.ndarray:
    h, w = bgr.shape[:2]
    k = max(9, (min(h, w) // 20) | 1)  # odd kernel, scaled to image size

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # leaf tissue: green (healthy) through yellow/brown (chlorosis, necrosis, rust) -- wide hue band
    lower = np.array([10, 25, 25])
    upper = np.array([95, 255, 255])
    mask = cv2.inRange(hsv, lower, upper).astype(np.float32) / 255.0
    mask = cv2.GaussianBlur(mask, (k, k), 0)  # soft edges, avoid a hard cutout
    mask = mask[..., None]

    # Recede the background by softening it (heavy blur) rather than flattening it to
    # grey -- a flat brightness multiply turns bright, low-saturation bokeh highlights
    # into ugly uniform grey patches. Blurring keeps each pixel's own color/character
    # while still visually de-emphasizing it relative to the sharp, contrast-boosted leaf.
    softened = cv2.GaussianBlur(bgr, (0, 0), sigmaX=max(3, k / 6))
    softened = (softened.astype(np.float32) * (1 - strength * 0.4)).astype(np.float32)

    blended = bgr.astype(np.float32) * mask + softened * (1 - mask)
    return np.clip(blended, 0, 255).astype(np.uint8)


def _unsharp(bgr: np.ndarray, amount: float = 0.6) -> np.ndarray:
    blurred = cv2.GaussianBlur(bgr, (0, 0), sigmaX=2.0)
    sharpened = cv2.addWeighted(bgr, 1 + amount, blurred, -amount, 0)
    return sharpened


def _resize_max_dim(bgr: np.ndarray, max_dim: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    if max(h, w) <= max_dim:
        return bgr
    scale = max_dim / max(h, w)
    return cv2.resize(bgr, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)


def enhance_bgr(bgr: np.ndarray, suppress_background: bool = True, working_size: int = 512) -> np.ndarray:
    # Downscale first: the model input is 224x224 regardless, full-resolution source
    # photos (some 4000px+) make bilateral filtering slow for no downstream benefit,
    # and working at a size closer to the model's actually sharpens mask proportions.
    out = _resize_max_dim(bgr, working_size)
    out = _clahe_lab(out)
    out = _denoise(out)
    if suppress_background:
        out = _soft_background_suppress(out)
    out = _unsharp(out)
    return out


def enhance_pil(img: Image.Image, suppress_background: bool = True, working_size: int = 512) -> Image.Image:
    bgr = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    out_bgr = enhance_bgr(bgr, suppress_background=suppress_background, working_size=working_size)
    return Image.fromarray(cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--no_background_suppress", action="store_true")
    args = ap.parse_args()

    img = Image.open(args.input)
    enhanced = enhance_pil(img, suppress_background=not args.no_background_suppress)
    enhanced.save(args.output, quality=95)
    print(f"Saved enhanced image -> {args.output}")


if __name__ == "__main__":
    main()
