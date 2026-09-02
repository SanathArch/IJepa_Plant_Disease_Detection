# Trained Leaf Analysis Model (self-contained)

Everything needed to run predictions lives in this folder -- no internet connection
or access to the rest of the project is required.

## Contents
- `backbone/` -- frozen I-JEPA ViT-H/14 weights (`facebook/ijepa_vith14_1k`), copied
  locally so inference works fully offline.
- `condition_head.pt` / `condition_vocab.json` -- disease classifier (21 classes,
  species-agnostic). Val micro-F1 = 0.884 (500-epoch retrain; 0.880 at 60 epochs --
  the gain past the ~epoch-25 plateau is small, since val loss overfits from there
  on and best-checkpoint selection is what actually captures the improvement).
- `species_head_baseline.pt` / `species_vocab.json` -- species classifier (14
  classes), trained on the I-JEPA embedding alone. Val micro-F1 = 0.936 (500-epoch
  retrain; 0.934 at 60 epochs).
- `species_geometry_reference.json` -- per-species reference clusters (centroid +
  spread) in classical-CV leaf-geometry feature space, per
  `ijepa_Leaf_PointCloud_Strategy.md`.
- `leaf_geometry.py` -- segmentation -> contour -> Elliptical Fourier Descriptors +
  vein-network topology. Produces the 47-dim geometric signature used above.
- `enhance_cv.py` -- OpenCV preprocessing (CLAHE contrast, denoise, soft background
  suppression, sharpening) applied to every image before the backbone sees it.
- `lesion_localization.py` -- multi-scale, leaf-masked, multi-instance lesion
  detection, logic extracted from apple-disease detection research and generalized
  to every species/condition already in the vocab (no new labels or training data
  needed -- it's a post-hoc analysis on top of the existing disease head). See
  "Lesion localization" below.
- `fused_condition_head.pt` / `fused_species_head.pt` -- experimental heads trained
  on the I-JEPA embedding **concatenated** with the geometric descriptor. Not used
  by `predict.py` by default -- see "Fusion experiment" below for why.
- `adaptive_thresholds/disease.json`, `adaptive_thresholds/species.json` -- per-class
  auto-accept confidence thresholds, derived from measured test-set F1 (Stage 1,
  continual learning strategy). A class the model is proven reliable on auto-accepts
  at lower confidence; a class with a thin or zero track record needs to clear a much
  higher bar before `predict.py` will accept it without flagging for review.
- `recurrence_memory.json` -- per-class confirmed-sample counts, seeded from the
  labeled training set. Drives Stage 3 (immune-memory) routing: predictions that
  don't resemble any known class at all are routed to `UNKNOWN` rather than forced
  into a guess, and predictions on thinly-confirmed classes need more confidence to
  be accepted than predictions on well-established ones.
- `predict.py` -- the inference script (see `../predict.bat` for the one-click launcher).

## Continual learning: review-routing, not just a raw prediction
Every prediction `predict.py` makes is routed, not just reported. Disease
predictions can land in one of three states:
- **auto-accepted** -- confidence clears both the class's evidence-based threshold
  (Stage 1) and its recurrence-confidence bar (Stage 3).
- **`[NEEDS REVIEW]`** -- recognized as a known class, but confidence didn't clear
  one of those bars (e.g. a rarely-confirmed class, or a track record that's still
  thin).
- **`UNKNOWN`** -- doesn't resemble any known class at all (max probability across
  every class is below 15%). Never forced into a guessed label.

Species predictions get the same Stage 1 evidence-based routing, shown alongside the
independent geometric cross-check described above. See
`Crop_Tracker_Continual_Learning_Strategy.md` and `src/continual/` in the main
project for the full mechanism (also includes a periodic consolidation/replay-buffer
retraining cycle and a fusion-graduation tracker, which operate on the training
pipeline rather than at inference time, so they aren't part of this standalone
package).

## Two signals on species, one on disease
`predict.py` reports disease from the learned I-JEPA head, and species from *two*
independent angles:
1. **Learned** (`species_head_baseline.pt`) -- the primary answer, 93.4% val F1.
2. **Geometric** (`species_geometry_reference.json`) -- a classical-CV nearest-cluster
   match on leaf outline shape + vein topology. Cheap, instant, fully offline, and
   interpretable (every number in a 47-dim EFD+vein vector maps to an actual shape
   property), but standalone accuracy on our current 14-species/64k-image data is
   only ~22% -- it is not a replacement for the learned model here. It's shown as a
   cross-check: when it agrees with the learned prediction, that's corroborating
   evidence; when it disagrees, the disagreement is surfaced rather than hidden.

## Fusion experiment: concatenating geometry into the I-JEPA representation
The point-cloud strategy doc frames the geometric signal as a fast, interpretable,
low-data-requirement *complement* to I-JEPA -- valuable especially for bootstrapping
a brand-new species from just a handful of examples, before there's enough data to
train a good learned head. At the request to actually fuse the two into one
representation and retrain, we built that (`train_fused.py`: concatenate
`[ijepa_embedding (1280), geom_descriptor (47), geom_valid_bit (1)]` -> 1328-dim
input) and compared it head-to-head against an I-JEPA-only baseline trained with the
identical recipe:

| head | val micro-F1 |
|---|---|
| disease, I-JEPA only | **0.880** |
| disease, fused | 0.876 |
| species, I-JEPA only | **0.934** |
| species, fused | 0.932 |

**Honest result: fusion didn't help, at our current data scale.** The differences are
within normal run-to-run noise, and I-JEPA-only was marginally ahead both times. The
likely reason: I-JEPA's 1280-dim self-supervised embedding already implicitly
encodes most of the shape/venation information the hand-crafted 47-dim descriptor
captures, so concatenating a cruder version of overlapping information doesn't add
signal once you already have a strong pretrained backbone and 51k+ training images.
This matches the strategy doc's own framing -- the geometric signal's real value is
cheap/offline/interpretable bootstrapping when learned-model data is scarce, not
squeezing out extra peak accuracy once a mature model already exists. `predict.py`
therefore ships the I-JEPA-only heads as primary and uses the geometric signal as a
decision-level cross-check (agreement/disagreement), not a feature-level fusion --
the fused checkpoints are kept in this folder for reference/reproducibility, not as
the default path.

## Lesion localization: where on the leaf, not just what
Extracted from Jiang et al.'s apple-leaf-disease detection research (a dedicated
SSD/VGG/Inception object detector, 78.80% mAP, apple-only, 5 disease classes). That
specific 2018 CNN is a dead end to reimplement -- obsolete next to the frozen
ViT-H/14 already in this package, and limited to apple. What generalizes is the
*logic* their own failure analysis points at:
1. Small lesions get missed at a single scale -> run disease-head relevance at
   several center-crop scales of the same image and fuse by max.
2. Background gets misidentified as disease -> mask the fused relevance to the
   segmented leaf silhouette (`leaf_geometry.py`) before anything downstream sees it.
3. Multiple distinct lesions can co-occur -> threshold + connected-component label
   the masked map into discrete instances, instead of one whole-leaf heatmap.

This runs species-agnostically on top of the *existing* disease head -- no new
labels, no new training data, works for every class already in the vocab. `predict.py`
draws each detected instance as a white box on the relevance heatmap and reports
lesion count + area in the title and CSV summary (`n_lesions`,
`lesion_total_area_pct`, `lesion_largest_area_pct`). To add a future paper's logic as
another lens the same way: one small module with a single entry point returning a
fixed-length feature vector, one visualization panel, and optionally concatenate its
feature vector alongside the I-JEPA embedding in fusion training -- nothing else in
the pipeline needs to change.

**Fusion tested too, same honest result as geometry.** Concatenating the 4-dim
lesion summary (`n_lesions`, `total_area_fraction`, `largest_area_fraction`,
`mean_peak_score`) onto the I-JEPA embedding and retraining head-to-head against a
fresh I-JEPA-only baseline, same 60-epoch recipe:

| head | val micro-F1 |
|---|---|
| disease, I-JEPA only | 0.878 |
| disease, + lesion fusion | 0.880 |
| species, I-JEPA only | 0.935 |
| species, + lesion fusion | 0.933 |

Both differences are within normal run-to-run noise -- no real gain either
direction, same conclusion as the geometry-fusion experiment above. `predict.py`
therefore keeps lesion localization as a visual/interpretive overlay (bounding
boxes + counts on the heatmap) rather than a feature-level fusion into the trained
head; the fused checkpoints (`lesion_fused_condition_head.pt`,
`lesion_fused_species_head.pt`) are kept for reference, not the default path.

## Usage
```
python predict.py "C:\path\to\leaf.jpg"
python predict.py "C:\path\to\folder_of_leaves"
python predict.py "C:\path\to\folder" --recursive
```
Or just double-click `../predict.bat` and drag an image/folder onto it.

Each input image gets a `<name>_prediction.jpg` written next to it (leaf photo +
disease relevance heatmap, with disease + both species opinions in the titles/
captions). Folders also get a `_predictions_summary.csv`.

## Caveats
- Species-condition combinations only observed for one species in the training data
  (citrus greening, tomato virus classes, black rot) haven't been validated for
  cross-species disease transfer -- that needs more species with those symptoms.
- The geometric segmentation (leaf_geometry.py) assumes there's one dominant leaf in
  frame; it will produce a low-confidence or garbage signature on photos with many
  overlapping leaves and no clear single subject. `predict.py` handles this
  gracefully (the geometric opinion is simply omitted), but it's worth knowing why
  the "shape-based guess" sometimes doesn't appear.
- All heads are linear/MLP probes on a *frozen* backbone, not fine-tuned -- fast and
  label-efficient, with a lower ceiling than full fine-tuning.
