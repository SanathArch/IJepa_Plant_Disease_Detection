"""Classical-CV leaf "point cloud" geometry, per ijepa_Leaf_PointCloud_Strategy.md.

Builds a compact, rotation/scale/translation-invariant geometric signature for a
leaf photo: segment -> contour ("the raw 2D point cloud") -> normalize/resample ->
landmarks -> Elliptical Fourier Descriptors (the standard geometric-morphometrics
tool for this exact problem) + simple scale-invariant shape scalars + a vein-network
topology summary (Frangi vesselness filter -> skeleton -> branch/endpoint stats).

This produces a single fixed-length vector per leaf. It's used two ways:
  1. Standalone: per-species reference clusters + nearest-centroid matching
     (species_reference.py), for a cheap, offline, interpretable species signal.
  2. Fused: concatenated onto the I-JEPA embedding as extra input dimensions before
     the classifier heads, so the learned model reasons over both signals jointly
     (see extract_geometry.py / train_fused.py).

Any image where segmentation fails (blown-out background, no clear leaf silhouette,
degenerate contour) returns `None` -- callers must handle that as a missing/invalid
reading, never silently substitute zeros as if they meant something.
"""
import cv2
import numpy as np
import pyefd
from scipy.ndimage import convolve
from skimage.filters import frangi
from skimage.morphology import skeletonize

EFD_ORDER = 10
N_SHAPE_SCALARS = 3   # circularity, solidity, aspect_ratio
N_VEIN_FEATURES = 4   # vein_density, branch_density, endpoint_density, branch/endpoint ratio
GEOM_DIM = EFD_ORDER * 4 + N_SHAPE_SCALARS + N_VEIN_FEATURES  # 47

WORKING_SIZE = 384
N_RESAMPLE_POINTS = 150


def _segment_leaf(bgr: np.ndarray):
    h, w = bgr.shape[:2]
    scale = WORKING_SIZE / max(h, w)
    if scale < 1.0:
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue_mask = cv2.inRange(hsv, (10, 25, 25), (95, 255, 255))

    sat = hsv[:, :, 1]
    _, sat_mask = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    mask = cv2.bitwise_or(hue_mask, sat_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, None, None
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    frame_area = bgr.shape[0] * bgr.shape[1]
    if area < 400 or area < 0.02 * frame_area:
        return None, None, None

    # A leaf that fills most (even nearly all) of the frame is a perfectly normal,
    # common macro-photography composition -- reject on raw area would wrongly kill
    # those. What actually indicates segmentation failed onto the rectangular frame
    # border itself (rather than a real leaf silhouette) is a contour whose extent
    # (area / bounding-box area) is suspiciously close to 1.0 -- real leaf outlines
    # have serration/lobing/petiole notches that a bare rectangle doesn't.
    x, y, bw, bh = cv2.boundingRect(contour)
    extent = area / max(bw * bh, 1)
    touches_all_edges = (x <= 1 and y <= 1 and x + bw >= bgr.shape[1] - 1 and y + bh >= bgr.shape[0] - 1)
    # Tight macro crops (leaf fills the whole frame, edges clipped by the photo
    # border on some/all sides) are a normal, common composition in this data and
    # still have real, informative boundary curvature wherever it's visible -- only
    # a near-perfect rectangle (extent ~1.0) indicates the mask degenerated onto the
    # frame border itself rather than tracing any real leaf edge.
    if touches_all_edges and extent > 0.999:
        return None, None, None

    # clean_mask (filled) is for masking the vein-detection step to the leaf region;
    # the ORIGINAL contour (not one re-derived from this filled mask) is what carries
    # the real, fine boundary shape and must be used for every shape/EFD computation --
    # re-extracting a contour from an already-filled near-frame-sized mask rounds it
    # toward a bare rectangle and throws the actual leaf outline away.
    clean_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(clean_mask, [contour], -1, 255, thickness=cv2.FILLED)
    return clean_mask, contour.reshape(-1, 2), bgr


def _resample_contour(contour: np.ndarray, n_points: int = N_RESAMPLE_POINTS):
    pts = contour.astype(np.float64)
    closed = np.vstack([pts, pts[:1]])
    seg_len = np.sqrt((np.diff(closed, axis=0) ** 2).sum(axis=1))
    cum = np.concatenate([[0], np.cumsum(seg_len)])
    total = cum[-1]
    if total < 1e-6:
        return None
    targets = np.linspace(0, total, n_points, endpoint=False)
    x = np.interp(targets, cum, closed[:, 0])
    y = np.interp(targets, cum, closed[:, 1])
    return np.stack([x, y], axis=1)


def _landmarks_and_shape(contour: np.ndarray, resampled: np.ndarray):
    centered = resampled - resampled.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, np.argmax(eigvals)]
    perp = eigvecs[:, np.argmin(eigvals)]

    proj = centered @ principal
    perp_proj = centered @ perp
    length = float(proj.max() - proj.min())
    width = float(perp_proj.max() - perp_proj.min())
    aspect_ratio = width / (length + 1e-6)

    area = cv2.contourArea(contour.astype(np.float32))
    perimeter = cv2.arcLength(contour.astype(np.float32), True)
    circularity = float(4 * np.pi * area / (perimeter ** 2 + 1e-6))

    hull = cv2.convexHull(contour.astype(np.float32))
    hull_area = cv2.contourArea(hull)
    solidity = float(area / (hull_area + 1e-6))

    return np.array([circularity, solidity, aspect_ratio], dtype=np.float64)


def _vein_features(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray).astype(np.float64) / 255.0

    # Veins can appear lighter or darker than surrounding tissue depending on lighting
    # (backlit translucent leaves vs. directly lit ones) -- take both ridge polarities.
    ridge_dark = frangi(enhanced, sigmas=range(1, 4), black_ridges=True)
    ridge_light = frangi(enhanced, sigmas=range(1, 4), black_ridges=False)
    ridge = np.maximum(ridge_dark, ridge_light)
    ridge[mask == 0] = 0

    leaf_vals = ridge[mask > 0]
    if leaf_vals.size == 0 or leaf_vals.max() <= 0:
        return np.zeros(N_VEIN_FEATURES, dtype=np.float64)
    thresh_val = leaf_vals.mean() + leaf_vals.std()
    vein_mask = ridge > thresh_val

    skel = skeletonize(vein_mask)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    neighbor_count = convolve(skel.astype(np.uint8), kernel, mode="constant")

    branch_points = int(((neighbor_count >= 3) & skel).sum())
    endpoints = int(((neighbor_count == 1) & skel).sum())
    leaf_area = int((mask > 0).sum())
    skel_px = int(skel.sum())

    vein_density = skel_px / max(leaf_area, 1)
    branch_density = branch_points / max(leaf_area, 1) * 1000
    endpoint_density = endpoints / max(leaf_area, 1) * 1000
    branch_endpoint_ratio = branch_points / max(endpoints, 1)

    return np.array([vein_density, branch_density, endpoint_density, branch_endpoint_ratio], dtype=np.float64)


def leaf_signature(pil_img) -> np.ndarray | None:
    """Returns a GEOM_DIM-length float32 vector, or None if the leaf couldn't be
    reliably segmented from this photo (caller must treat that as missing data,
    not silently zero-fill it)."""
    try:
        bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
        mask, contour, resized_bgr = _segment_leaf(bgr)
        if mask is None or len(contour) < 20:
            return None

        resampled = _resample_contour(contour)
        if resampled is None:
            return None

        efd = pyefd.elliptic_fourier_descriptors(resampled, order=EFD_ORDER, normalize=True).flatten()
        shape_scalars = _landmarks_and_shape(contour, resampled)

        gray = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2GRAY)
        vein_feats = _vein_features(gray, mask)

        vec = np.concatenate([efd, shape_scalars, vein_feats]).astype(np.float32)
        if not np.all(np.isfinite(vec)):
            return None
        return vec
    except Exception:
        return None
