"""Lesion-localization lens -- logic extracted from Jiang et al., "Real-Time
Detection of Apple Leaf Diseases Using Deep Learning Approach Based on Improved
Convolutional Neural Networks" (IEEE Access, 2019), Research/Real-Time_Detection_
of_Apple_Leaf_Diseases_Using_D.pdf.

That paper trains a dedicated SSD/VGG/Inception object detector to find disease
LESIONS as located, multi-scale, possibly-multiple objects on a leaf (78.80% mAP,
apple-only, 5 disease classes) -- a fundamentally different, richer output than a
single whole-image label. Their own failure analysis (Figure 13) names three
specific problems worth solving generally, not just for apple:
  1. Small lesions get missed at a single feature scale (motivates their Inception
     multi-scale module + Rainbow cross-scale concatenation).
  2. Background gets misidentified as disease when color/texture happens to match
     (their dark-background-as-Mosaic, orange-background-as-Rust failures).
  3. Multiple distinct lesions, of different sizes, can co-occur on one leaf.

Reimplementing their specific 2018 CNN would be a dead end -- SSD/VGG/Inception is
obsolete next to a frozen ViT-H/14 already trained here, and their model only
covers 5 apple diseases. Instead, this module extracts the LOGIC (multi-scale,
background-masked, multi-instance) as a species-agnostic, purely post-hoc analysis
on top of the disease head we already have: no new labels, no new training data,
works for every species/condition in our existing vocabulary already.

Pipeline:
  1. Run the disease head's patch-relevance computation (see predict.py) at several
     center-crop scales of the SAME image -- a small object only fills enough
     patches to register at a tight crop; a large one only registers cleanly at the
     full frame. Each scale's relevance map is placed back into a common
     full-image coordinate frame (not stretched over it) and fused by taking the
     max across scales at every pixel.
  2. Mask the fused map to the segmented leaf silhouette (leaf_geometry.py) --
     directly targets failure mode 2: background can't register as a lesion if
     it isn't leaf tissue.
  3. Threshold + connected-component label the masked map into discrete lesion
     instances -- directly targets failure mode 3: this recovers instance count,
     each instance's location, and its size, not just one global heatmap.

Adding a future paper's logic as another lens: give it its own small module with
one clear entry point returning a fixed-length feature vector (mirroring this
file's `lesion_summary_features` and leaf_geometry.py's `leaf_signature`), add a
panel to visualize_lenses.py, and optionally concatenate its feature vector
alongside the I-JEPA embedding in train_fused.py-style fusion training. Nothing
else in the pipeline needs to change to add a new research-derived lens this way.
"""
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from scipy import ndimage

SCALES = (1.0, 0.7, 0.5)  # full frame, then progressively tighter center crops
RELEVANCE_THRESHOLD = 0.75  # fraction of the fused map's own max, per-image-relative
MIN_LESION_AREA_PX = 40      # at 224x224 (~50k px), below this a "blob" is almost certainly noise

N_LESION_FEATURES = 4
LESION_FEATURE_NAMES = ["n_lesions", "total_area_fraction", "largest_area_fraction", "mean_peak_score"]


def _patch_relevance_at_scale(backbone, head, pil_img: Image.Image, class_idx: int, scale: float,
                               device: str, image_size: int, mean, std) -> np.ndarray:
    """Relevance map for one crop scale, placed into a full image_size x image_size
    canvas at the crop's real position (not stretched across the whole canvas) --
    what makes this genuinely multi-SCALE rather than just multi-resolution."""
    w, h = pil_img.size
    if scale < 1.0:
        cw, ch = int(w * scale), int(h * scale)
        left, top = (w - cw) // 2, (h - ch) // 2
        crop = pil_img.crop((left, top, left + cw, top + ch))
    else:
        crop = pil_img
        left, top, cw, ch = 0, 0, w, h

    resized = T.Compose([T.Resize(int(image_size * 1.14)), T.CenterCrop(image_size)])(crop)
    tensor = T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])(resized).unsqueeze(0).to(device)

    with torch.no_grad():
        patch_embeds = backbone(pixel_values=tensor).last_hidden_state[0]
        patch_logits = head(patch_embeds)[:, class_idx]
    grid = int(np.sqrt(patch_embeds.shape[0]))
    m = patch_logits.reshape(grid, grid).float()
    m = (m - m.min()) / (m.max() - m.min() + 1e-8)
    m = F.interpolate(m.unsqueeze(0).unsqueeze(0), size=(image_size, image_size),
                       mode="bilinear", align_corners=False).squeeze().cpu().numpy()

    # Place this crop's map back at its real position in full-image coordinates.
    canvas = np.zeros((image_size, image_size), dtype=np.float32)
    left_px = int(left / w * image_size)
    top_px = int(top / h * image_size)
    cw_px = int(cw / w * image_size)
    ch_px = int(ch / h * image_size)
    placed = np.array(Image.fromarray((m * 255).astype(np.uint8)).resize((max(cw_px, 1), max(ch_px, 1)))) / 255.0
    canvas[top_px:top_px + placed.shape[0], left_px:left_px + placed.shape[1]] = placed
    return canvas


def multiscale_relevance(backbone, head, pil_img: Image.Image, class_idx: int, device: str,
                          image_size: int, mean, std, scales=SCALES) -> np.ndarray:
    maps = [_patch_relevance_at_scale(backbone, head, pil_img, class_idx, s, device, image_size, mean, std)
            for s in scales]
    return np.maximum.reduce(maps)


def detect_lesion_instances(relevance_map: np.ndarray, leaf_mask: np.ndarray = None,
                             threshold: float = RELEVANCE_THRESHOLD, min_area: int = MIN_LESION_AREA_PX):
    """Connected-component instances above threshold, masked to real leaf tissue if
    a segmentation mask is given (leaf_mask must be the same H x W as relevance_map).
    Returns a list of {bbox: (y0,y1,x0,x1), area_fraction, peak_score}."""
    thresh_val = relevance_map.max() * threshold
    binary = relevance_map >= thresh_val
    if leaf_mask is not None:
        if leaf_mask.shape != relevance_map.shape:
            leaf_mask = np.array(Image.fromarray(leaf_mask).resize(
                (relevance_map.shape[1], relevance_map.shape[0]), Image.NEAREST))
        binary = binary & (leaf_mask > 0)
    # A whole-image classifier's patch logits are a much smoother, coarser signal
    # than a purpose-trained detector's -- speckle noise at the threshold boundary
    # is expected. Opening removes isolated speckle without softening real blobs.
    binary = ndimage.binary_opening(binary, structure=np.ones((3, 3)))

    labeled, n = ndimage.label(binary)
    total_px = relevance_map.size
    instances = []
    for i in range(1, n + 1):
        ys, xs = np.where(labeled == i)
        if len(ys) < min_area:
            continue
        instances.append({
            "bbox": (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())),
            "area_fraction": float(len(ys) / total_px),
            "peak_score": float(relevance_map[ys, xs].max()),
        })
    instances.sort(key=lambda d: -d["area_fraction"])
    return instances


def lesion_summary_features(instances: list) -> np.ndarray:
    """Fixed-length feature vector summarizing the detected instances -- fusable
    onto the I-JEPA embedding the same way leaf_geometry.py's descriptor is."""
    if not instances:
        return np.zeros(N_LESION_FEATURES, dtype=np.float32)
    n = len(instances)
    total_area = sum(d["area_fraction"] for d in instances)
    largest_area = instances[0]["area_fraction"]
    mean_peak = float(np.mean([d["peak_score"] for d in instances]))
    return np.array([n, total_area, largest_area, mean_peak], dtype=np.float32)
