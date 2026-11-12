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
- File: `best.onnx`: the same weights for CPU inference without PyTorch. ONNX opset 12, fixed 1 x 3 x 224 x 224 input, output 1 x 5 x 1029 (cx, cy, w, h, score in pixels of the 224 px input). NMS is not in the graph. Parity with Ultralytics checked on 20 training images with `test/06f_test_onnx_parity.py`: with the same square input (`rect=False`), the boxes and scores are identical.

## Usage

```python
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

weights = hf_hub_download("hectorvh/gpradar-hyperbola-yolo", "best.pt")
model = YOLO(weights)
results = model.predict("bscan.png", imgsz=224, conf=0.25)
results[0].show()
```

ONNX with onnxruntime (CPU, no PyTorch). Letterbox to 224 x 224 like Ultralytics (keep aspect ratio, pad with grey 114, centre), then filter by score and run NMS yourself:

```python
import cv2
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

session = ort.InferenceSession(hf_hub_download("hectorvh/gpradar-hyperbola-yolo", "best.onnx"),
                               providers=["CPUExecutionProvider"])
grey = cv2.imread("bscan.png", cv2.IMREAD_GRAYSCALE)
h, w = grey.shape
r = min(224 / h, 224 / w)
nw, nh = round(w * r), round(h * r)
left, top = round((224 - nw) / 2 - 0.1), round((224 - nh) / 2 - 0.1)
canvas = np.full((224, 224), 114, np.uint8)
canvas[top:top + nh, left:left + nw] = cv2.resize(grey, (nw, nh), interpolation=cv2.INTER_LINEAR)
x = np.repeat(canvas[None, None], 3, axis=1).astype(np.float32) / 255.0

pred = session.run(None, {"images": x})[0][0].T          # (1029, 5): cx, cy, w, h, score
pred = pred[pred[:, 4] > 0.25]
xyxy = np.c_[pred[:, :2] - pred[:, 2:4] / 2, pred[:, :2] + pred[:, 2:4] / 2]
keep = cv2.dnn.NMSBoxes(np.c_[xyxy[:, :2], pred[:, 2:4]].tolist(), pred[:, 4].tolist(), 0.25, 0.5)
keep = np.asarray(keep, dtype=int).ravel()
boxes = (xyxy[keep] - [left, top, left, top]) / [nw / w, nh / h, nw / w, nh / h]
boxes = boxes.clip(0, [w, h, w, h])                         # xyxy in original pixels
```

Note: Ultralytics `predict` on `best.pt` pads only to a multiple of 32 (`rect=True` by default), while the ONNX model always sees a full 224 x 224 square. Scores can differ slightly between the two (up to about 0.01 on 20 training images; one borderline box near 0.25 changed). The locked-test numbers below were measured with Ultralytics on `best.pt`.

## Training data

Real B-scans from Mojahid et al. (2025), *Intelligent recognition of subsurface utilities and voids using ground-penetrating radar and deep learning techniques: Data of GPR profiles images*, Mendeley Data, DOI [10.17632/ww7fd9t325.1](https://doi.org/10.17632/ww7fd9t325.1).

Only the 285 original images are used (the published augmented copies are excluded because they could not be reliably matched to originals for a leakage-free split). Cavity and utility images were labelled by hand with one class, `target`; intact images are negatives. Near-duplicates are grouped and kept in the same split. Split: 70% train / 30% test, stratified by class at group level; the 83 test images are locked and were used only once, after all training.

## Evaluation

Locked test set: 83 real images. Recall and false positives at confidence 0.25, IoU 0.5.

| Configuration | mAP@0.5 (mean ± std, 3 seeds) | Recall utility | Recall void | FP per image |
| --- | --- | --- | --- | --- |
| Real only, 100% labels (this model's configuration) | 0.756 ± 0.042 | 0.90 | 0.51 | 0.55 |

This checkpoint (seed 2) on its own: mAP@0.5 = 0.786, mAP@0.5:0.95 = 0.438.

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
