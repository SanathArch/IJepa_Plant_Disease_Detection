"""Standalone, offline-capable leaf analysis: disease + species, two complementary
angles per ijepa_Leaf_PointCloud_Strategy.md. Self-contained: everything this script
needs (backbone weights, trained heads, vocabs, geometric reference, enhancement
code) lives next to it in this folder -- no network access, no dependency on the
rest of the project.

Pipeline: OpenCV vision-enhancement -> frozen I-JEPA ViT-H/14 backbone (loaded from
the local backbone/ folder) -> trained disease head + trained species head. Alongside
that learned signal, also runs the classical-CV leaf-geometry species match (contour
-> Elliptical Fourier Descriptors + vein topology -> nearest reference-cluster
species) as a second, independent, offline opinion on species -- cheap and
interpretable, though measurably weaker than the learned model at the current data
scale (see README.md for the head-to-head numbers). Agreement raises confidence;
disagreement is shown, not silently resolved.

Also runs the lesion-localization lens (lesion_localization.py, logic extracted from
apple-disease detection research and generalized to every species/condition in the
vocab): multi-scale relevance fused and masked to leaf tissue, split into discrete
lesion instances (white boxes on the heatmap) rather than one undifferentiated blob.

For every input image, writes an annotated result image (prediction + confidence +
disease relevance heatmap + lesion boxes) into the SAME folder as the input, named
"<original>_prediction.jpg". If given a folder, also writes "_predictions_summary.csv".

Usage:
    python predict.py "C:\\path\\to\\leaf.jpg"
    python predict.py "C:\\path\\to\\folder_of_leaves"
    python predict.py "C:\\path\\to\\folder" --recursive
"""
import argparse
import csv
import json
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from transformers import IJepaModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import leaf_geometry as lg
from enhance_cv import enhance_pil
from leaf_geometry import leaf_signature
from lesion_localization import detect_lesion_instances, lesion_summary_features, multiscale_relevance

HERE = os.path.dirname(os.path.abspath(__file__))
BACKBONE_DIR = os.path.join(HERE, "backbone")
DISEASE_HEAD_PATH = os.path.join(HERE, "condition_head.pt")
DISEASE_VOCAB_PATH = os.path.join(HERE, "condition_vocab.json")
SPECIES_HEAD_PATH = os.path.join(HERE, "species_head_baseline.pt")
SPECIES_VOCAB_PATH = os.path.join(HERE, "species_vocab.json")
GEOM_REFERENCE_PATH = os.path.join(HERE, "species_geometry_reference.json")
DISEASE_THRESHOLDS_PATH = os.path.join(HERE, "adaptive_thresholds", "disease.json")
SPECIES_THRESHOLDS_PATH = os.path.join(HERE, "adaptive_thresholds", "species.json")
RECURRENCE_PATH = os.path.join(HERE, "recurrence_memory.json")

IJEPA_IMAGE_SIZE = 224
IJEPA_MEAN = [0.5, 0.5, 0.5]
IJEPA_STD = [0.5, 0.5, 0.5]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Mirrors species_reference.py's match rule: standardized (diagonal) distance to
# each species' running mean/variance, with a match threshold that ramps looser as
# a cluster's sample count grows (a 3-sample cluster must be matched more strictly
# than a 300-sample one -- its centroid/spread are less trustworthy yet).
GEOM_DIM = 47
MIN_VAR = 1e-4
THRESHOLD_BASE = GEOM_DIM * 2.5
THRESHOLD_TAU = 20.0

# Continual-learning Stage 1 (evidence-based auto-accept thresholds) and Stage 3
# (immune-memory unknown-routing / recurrence confidence) -- see
# Crop_Tracker_Continual_Learning_Strategy.md and src/continual/ in the main project.
DEFAULT_AUTO_ACCEPT_THRESHOLD = 0.90
UNKNOWN_MAX_PROB_FLOOR = 0.15
RECURRENCE_HIGH = 0.90
RECURRENCE_LOW = 0.40
RECURRENCE_TAU = 10.0


def load_json(path: str, default):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def stage1_route(label: str, prob: float, schedule: dict):
    threshold = schedule.get(label, {}).get("auto_accept_threshold", DEFAULT_AUTO_ACCEPT_THRESHOLD)
    return prob >= threshold, threshold


def stage3_route(probs_by_class: dict, recurrence: dict):
    top_class = max(probs_by_class, key=probs_by_class.get)
    top_prob = probs_by_class[top_class]
    if top_prob < UNKNOWN_MAX_PROB_FLOOR:
        return {"status": "unknown", "label": None, "confidence": top_prob}
    n = recurrence.get(top_class, 0)
    scale = n / (n + RECURRENCE_TAU)
    required = RECURRENCE_HIGH - (RECURRENCE_HIGH - RECURRENCE_LOW) * scale
    status = "accepted" if top_prob >= required else "needs_confirmation"
    return {"status": status, "label": top_class, "confidence": top_prob,
            "recurrence_count": n, "required_confidence": required}


class LinearProbeHead(nn.Module):
    def __init__(self, hidden_dim: int, num_labels: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels),
        )

    def forward(self, x):
        return self.net(x)


def match_species(x: np.ndarray, reference: dict):
    def cluster_variance(cluster):
        n = max(cluster["n"], 1)
        m2 = np.array(cluster["m2"], dtype=np.float64)
        var = m2 / n if n < 2 else m2 / (n - 1)
        return np.maximum(var, MIN_VAR)

    def threshold(n):
        scale = n / (n + THRESHOLD_TAU)
        return THRESHOLD_BASE * (0.25 + 0.75 * scale)

    distances = {}
    for species, cluster in reference.items():
        if cluster["n"] == 0:
            continue
        mean = np.array(cluster["mean"], dtype=np.float64)
        var = cluster_variance(cluster)
        distances[species] = float(np.sum((x - mean) ** 2 / var))
    if not distances:
        return None, float("inf"), False
    best_species, best_dist = min(distances.items(), key=lambda kv: kv[1])
    within = best_dist <= threshold(reference[best_species]["n"])
    return best_species, best_dist, within


def find_images(path: str, recursive: bool):
    if os.path.isfile(path):
        return [path]
    files = []
    if recursive:
        for dirpath, _dirs, filenames in os.walk(path):
            for fn in filenames:
                if os.path.splitext(fn)[1].lower() in IMAGE_EXTS and "_prediction" not in fn:
                    files.append(os.path.join(dirpath, fn))
    else:
        for fn in os.listdir(path):
            full = os.path.join(path, fn)
            if os.path.isfile(full) and os.path.splitext(fn)[1].lower() in IMAGE_EXTS and "_prediction" not in fn:
                files.append(full)
    return sorted(files)


def build_transform():
    return T.Compose([
        T.Resize(int(IJEPA_IMAGE_SIZE * 1.14)),
        T.CenterCrop(IJEPA_IMAGE_SIZE),
    ])


def to_normalized_tensor(img: Image.Image) -> torch.Tensor:
    t = T.Compose([T.ToTensor(), T.Normalize(mean=IJEPA_MEAN, std=IJEPA_STD)])
    return t(img)


def run_one(image_path, backbone, disease_head, disease_vocab, species_head, species_vocab,
            geom_reference, disease_schedule, species_schedule, recurrence, device, topk):
    raw_img = Image.open(image_path).convert("RGB")
    enhanced = enhance_pil(raw_img)
    resized = build_transform()(enhanced)
    pixel_values = to_normalized_tensor(resized).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = backbone(pixel_values=pixel_values)
        patch_embeds = outputs.last_hidden_state[0]
        global_embed = patch_embeds.mean(dim=0, keepdim=True)

        disease_probs = torch.sigmoid(disease_head(global_embed)[0])
        disease_patch_logits = disease_head(patch_embeds)
        species_probs = torch.sigmoid(species_head(global_embed)[0])

    top_disease_probs, top_disease_idx = disease_probs.topk(min(topk, len(disease_vocab)))
    disease_predictions = [(disease_vocab[i], p.item()) for p, i in zip(top_disease_probs, top_disease_idx)]

    top_species_probs, top_species_idx = species_probs.topk(min(topk, len(species_vocab)))
    ijepa_species_predictions = [(species_vocab[i], p.item()) for p, i in zip(top_species_probs, top_species_idx)]
    ijepa_species = ijepa_species_predictions[0][0]

    geom_vec = leaf_signature(enhanced)
    if geom_vec is not None:
        geom_species, geom_dist, _within = match_species(geom_vec, geom_reference)
    else:
        geom_species, geom_dist = None, None
    species_agree = geom_species == ijepa_species if geom_species else None

    disease_stage1_accept, _ = stage1_route(disease_predictions[0][0], disease_predictions[0][1], disease_schedule)
    species_auto_accept, _ = stage1_route(ijepa_species, ijepa_species_predictions[0][1], species_schedule)

    disease_probs_by_class = {lbl: p.item() for lbl, p in zip(disease_vocab, disease_probs)}
    disease_routing = stage3_route(disease_probs_by_class, recurrence)
    disease_auto_accept = disease_stage1_accept and disease_routing["status"] == "accepted"

    top_class_idx = top_disease_idx[0].item()
    grid = int(np.sqrt(patch_embeds.shape[0]))
    m = disease_patch_logits[:, top_class_idx].reshape(grid, grid).float()
    m = (m - m.min()) / (m.max() - m.min() + 1e-8)
    m = F.interpolate(m.unsqueeze(0).unsqueeze(0), size=(IJEPA_IMAGE_SIZE, IJEPA_IMAGE_SIZE),
                       mode="bilinear", align_corners=False)
    relevance_map = m.squeeze().cpu().numpy()

    img_np = np.array(resized.resize((IJEPA_IMAGE_SIZE, IJEPA_IMAGE_SIZE))) / 255.0

    # Lesion-localization lens (logic extracted from apple-disease detection
    # research, generalized to every species/condition already in the vocab): fuse
    # relevance across a few center-crop scales, mask to leaf tissue, then split
    # into discrete connected-component instances instead of one undifferentiated
    # blob.
    lesion_relevance = multiscale_relevance(
        backbone, disease_head, enhanced, top_class_idx, device, IJEPA_IMAGE_SIZE, IJEPA_MEAN, IJEPA_STD)
    try:
        bgr = cv2.cvtColor(np.array(enhanced.convert("RGB")), cv2.COLOR_RGB2BGR)
        leaf_mask, _contour, _resized_bgr = lg._segment_leaf(bgr)
        if leaf_mask is not None:
            leaf_mask = cv2.resize(leaf_mask, (IJEPA_IMAGE_SIZE, IJEPA_IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
    except Exception:
        leaf_mask = None
    lesion_instances = detect_lesion_instances(lesion_relevance, leaf_mask)
    lesion_feats = lesion_summary_features(lesion_instances)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5.6))
    axes[0].imshow(img_np)
    species_mark = "" if species_auto_accept else "  [NEEDS REVIEW]"
    species_line = f"species: {ijepa_species} ({ijepa_species_predictions[0][1]*100:.0f}%){species_mark}"
    if geom_species:
        agree_mark = "match" if species_agree else "differs"
        species_line += f"\nshape-based guess: {geom_species} ({agree_mark})"
    axes[0].set_title(species_line, fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(img_np)
    axes[1].imshow(relevance_map, cmap="jet", alpha=0.5)
    for inst in lesion_instances:
        y0, y1, x0, x1 = inst["bbox"]
        axes[1].add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                             linewidth=1.2, edgecolor="white", facecolor="none"))
    if disease_routing["status"] == "unknown":
        disease_title = f"UNKNOWN -- doesn't resemble any known disease (max {disease_routing['confidence']*100:.0f}%)"
    else:
        mark = "" if disease_auto_accept else "  [NEEDS REVIEW]"
        disease_title = f"{disease_predictions[0][0]}  ({disease_predictions[0][1]*100:.1f}%){mark}"
    if lesion_instances:
        disease_title += f"\n{len(lesion_instances)} lesion(s), largest {lesion_instances[0]['area_fraction']*100:.1f}% of leaf"
    axes[1].set_title(disease_title, fontsize=10)
    axes[1].axis("off")

    lines = "  |  ".join(f"{lbl}: {p*100:.1f}%" for lbl, p in disease_predictions)
    fig.suptitle(lines, fontsize=9, y=0.02)
    plt.tight_layout(rect=[0, 0.04, 1, 1])

    stem, _ext = os.path.splitext(image_path)
    out_path = f"{stem}_prediction.jpg"
    plt.savefig(out_path, dpi=140)
    plt.close(fig)

    return (out_path, disease_predictions, ijepa_species_predictions, geom_species, species_agree,
            disease_auto_accept, species_auto_accept, disease_routing, lesion_feats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="image file or folder of images")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if not os.path.exists(args.path):
        print(f"ERROR: path does not exist: {args.path}")
        sys.exit(1)

    images = find_images(args.path, args.recursive)
    if not images:
        print(f"No images found under: {args.path}")
        sys.exit(1)
    print(f"Found {len(images)} image(s) to process.")

    with open(DISEASE_VOCAB_PATH, encoding="utf-8") as f:
        disease_vocab = json.load(f)
    disease_ckpt = torch.load(DISEASE_HEAD_PATH, map_location="cpu")
    disease_head = LinearProbeHead(disease_ckpt["hidden_dim"], len(disease_vocab), disease_ckpt["dropout"])
    disease_head.load_state_dict(disease_ckpt["model_state"])
    disease_head.eval().to(args.device)
    print(f"Loaded disease head ({len(disease_vocab)} classes, "
          f"val micro-F1={disease_ckpt.get('val_f1', float('nan')):.3f})")

    with open(SPECIES_VOCAB_PATH, encoding="utf-8") as f:
        species_vocab = json.load(f)
    species_ckpt = torch.load(SPECIES_HEAD_PATH, map_location="cpu")
    species_head = LinearProbeHead(species_ckpt["hidden_dim"], len(species_vocab), species_ckpt["dropout"])
    species_head.load_state_dict(species_ckpt["model_state"])
    species_head.eval().to(args.device)
    print(f"Loaded species head ({len(species_vocab)} classes, "
          f"val micro-F1={species_ckpt.get('val_f1', float('nan')):.3f})")

    with open(GEOM_REFERENCE_PATH, encoding="utf-8") as f:
        geom_reference = json.load(f)
    print(f"Loaded geometric species reference ({len(geom_reference)} species clusters)")

    disease_schedule = load_json(DISEASE_THRESHOLDS_PATH, {})
    species_schedule = load_json(SPECIES_THRESHOLDS_PATH, {})
    recurrence = load_json(RECURRENCE_PATH, {})
    print(f"Loaded adaptive thresholds ({len(disease_schedule)} disease, {len(species_schedule)} species classes) "
          f"and recurrence memory ({len(recurrence)} classes)")

    print(f"Loading I-JEPA backbone from local copy: {BACKBONE_DIR} ...")
    backbone = IJepaModel.from_pretrained(BACKBONE_DIR)
    backbone.eval().to(args.device)
    for p in backbone.parameters():
        p.requires_grad = False

    summary_rows = []
    for i, img_path in enumerate(images, 1):
        try:
            (out_path, disease_preds, species_preds, geom_species, agree,
             disease_auto_accept, species_auto_accept, disease_routing, lesion_feats) = run_one(
                img_path, backbone, disease_head, disease_vocab, species_head, species_vocab,
                geom_reference, disease_schedule, species_schedule, recurrence, args.device, args.topk,
            )
            top_disease, top_disease_conf = disease_preds[0]
            top_species, top_species_conf = species_preds[0]
            agree_str = "n/a" if agree is None else ("agree" if agree else "DISAGREE")
            if disease_routing["status"] == "unknown":
                review_str = "UNKNOWN:disease"
            else:
                review_str = " ".join(filter(None, [
                    None if disease_auto_accept else "REVIEW:disease",
                    None if species_auto_accept else "REVIEW:species",
                ])) or "auto-accepted"
            n_lesions = int(lesion_feats[0])
            print(f"[{i}/{len(images)}] {os.path.basename(img_path)} -> "
                  f"disease={top_disease} ({top_disease_conf*100:.0f}%)  "
                  f"species={top_species} ({top_species_conf*100:.0f}%, geometric={geom_species}, {agree_str})  "
                  f"lesions={n_lesions}  "
                  f"[{review_str}]  saved {os.path.basename(out_path)}")
            summary_rows.append({
                "image": img_path,
                "prediction_image": out_path,
                "disease": top_disease,
                "disease_confidence": round(top_disease_conf, 4),
                "disease_auto_accepted": disease_auto_accept,
                "disease_routing_status": disease_routing["status"],
                "n_lesions": n_lesions,
                "lesion_total_area_pct": round(float(lesion_feats[1]) * 100, 2),
                "lesion_largest_area_pct": round(float(lesion_feats[2]) * 100, 2),
                "species_ijepa": top_species,
                "species_ijepa_confidence": round(top_species_conf, 4),
                "species_auto_accepted": species_auto_accept,
                "species_geometric": geom_species,
                "species_agree": agree,
            })
        except Exception as e:
            print(f"[{i}/{len(images)}] FAILED on {img_path}: {e}")

    if os.path.isdir(args.path) and summary_rows:
        summary_path = os.path.join(args.path, "_predictions_summary.csv")
        fieldnames = list(summary_rows[0].keys())
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(summary_rows)
        print(f"\nWrote summary -> {summary_path}")

    print(f"\nDone. Processed {len(summary_rows)}/{len(images)} image(s).")


if __name__ == "__main__":
    main()
