---
license: agpl-3.0
library_name: ultralytics
pipeline_tag: object-detection
tags:
  - yolo
  - yolo11
  - ground-penetrating-radar
  - gpr
  - b-scan
  - hyperbola-detection
  - geophysics
---

# GPRadar hyperbola detector (YOLO11n)

Detects hyperbolic reflections from buried targets (utilities and voids) in ground-penetrating radar B-scan images. Part of [GPRadar](https://github.com/hectorvh/GPRadar), a GPR processing and detection prototype.

## Model

- Architecture: YOLO11n (Ultralytics), one class: `target`
- Input: greyscale B-scan image as 3 channels, 224 px
- Training run: `A_f100_seed2` (real data only, 100% of the real train labels, seed 2), fine-tuned from the COCO checkpoint
- File: `best.pt`

## Usage

```python
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

weights = hf_hub_download("hectorvh/gpradar-hyperbola-yolo", "best.pt")
model = YOLO(weights)
results = model.predict("bscan.png", imgsz=224, conf=0.25)
results[0].show()
```

## Training data

Real B-scans from Mojahid et al. (2025), *Intelligent recognition of subsurface utilities and voids using ground-penetrating radar and deep learning techniques: Data of GPR profiles images*, Mendeley Data, DOI [10.17632/ww7fd9t325.1](https://doi.org/10.17632/ww7fd9t325.1).

Only the 285 original images are used (the published augmented copies are excluded because they could not be reliably matched to originals for a leakage-free split). Cavity and utility images were labelled by hand with one class, `target`; intact images are negatives. Near-duplicates are grouped and kept in the same split. Split: 70% train / 30% test, stratified by class at group level; the 83 test images are locked and were used only once, after all training.

## Evaluation

Locked test set: 83 real images. Recall and false positives at confidence 0.25, IoU 0.5.

| Configuration | mAP@0.5 (mean ± std, 3 seeds) | Recall utility | Recall void | FP per image |
| --- | --- | --- | --- | --- |
| Real only, 100% labels (this model's configuration) | 0.756 ± 0.042 | 0.90 | 0.51 | 0.55 |

This checkpoint (seed 2) on its own: mAP@0.5 = TODO.

### Synthetic pre-training experiment

The project tested whether pre-training on synthetic gprMax B-scans (400 MHz field scenes) before fine-tuning on real data improves detection. It did not: at 25% and 100% of the real labels, the difference (synthetic + real minus real only) was -0.026 and -0.024 mAP@0.5, and both 95% bootstrap intervals include zero. A synthetic-only model reached 0.16 mAP@0.5 on real images. Full results: [GPRadar README](https://github.com/hectorvh/GPRadar#results-locked-test-set-83-real-images-3-seeds-per-setting).

## Limitations

- Small real dataset (156 training images, 83 test images); results have wide uncertainty.
- Voids are the hard case (recall about 0.5); real voids often appear as clusters of arches rather than a single hyperbola.
- The antenna frequency of the training data is not recorded. Performance on other antennas, soils or acquisition settings is unknown.
- Expects preprocessed B-scan images similar to the training set; it does not process raw radar files.
- A research prototype, not validated for field use or safety-critical utility locating.

## License and citation

Weights are released under AGPL-3.0, following Ultralytics. If you use the model, please also cite the dataset above.
