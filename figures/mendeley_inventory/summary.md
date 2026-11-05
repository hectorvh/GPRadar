# Mendeley GPR dataset inventory

Citation: Mojahid et al. (2025), *Data in Brief*, DOI [10.17632/ww7fd9t325.1](https://doi.org/10.17632/ww7fd9t325.1).

## Counts

| Class | Originals (here) | Published originals | Augmented (here) | Published class total |
| --- | ---: | ---: | ---: | ---: |
| void | 79 | 79 | 553 | 553 |
| utility | 131 | 131 | 786 | 786 |
| intact | 75 | 75 | 900 | 900 |

Images in this download: **2524** (285 originals + 2239 augmented). The published 2,239 figure equals the augmented counts only (2239).

## Annotations

Images outside intact with no YOLO and no VOC file: **210** (all originals in `cavities` and `Utilities`).
YOLO/VOC box mismatches (IoU ≤ 0.95 or count differ): **9**.
Class ids and names actually present:
- `augmented_cavities`: voc:cavities=600, yolo:0=600
- `augmented_utilities`: voc:Utility=1007, yolo:0=1007

No non-image, non-annotation files under the root.

## Frequency

Step by step:
1. Folder and file names: no 200/400 MHz tokens.
2. EXIF / image info: no frequency fields.
3. Side text/CSV/XML/DZT: none outside annotation folders.
4. Burned-in text on 20 originals: tick marks along the top edge on several cavity and utility scans; no numeric axis labels and no '200 MHz' / '400 MHz' text in the frame

**frequency = unknown** for every image.
frequency_hint (dominant vertical period, pixels) on originals: n=285, min=6.22, median=24.08, max=224.00.
2-means centres: 26.78 and 224.00. Centres are separated, but see the note below.
The high centre sits at the image height for the one 224×224 utility original whose spectrum peak is the fundamental bin. That is not evidence of separate 200 MHz and 400 MHz groups.

## Image sizes

- 224×224: 2220
- 268×268: 75
- 336×336: 75
- 500×465: 2
- 489×470: 1
- 381×463: 1
- 411×459: 1
- 381×457: 1
- 612×465: 1
- 401×466: 1
- 638×530: 1
- 419×345: 1
- … and 144 other sizes

## Boxes

Boxes per labelled image: min=1, median=1.0, max=4, images with boxes=1339.
- void: n=600, width median=120.0 px, height median=109.0 px
- utility: n=1007, width median=71.0 px, height median=72.0 px

## Duplicates and augmentation groups

Exact MD5 duplicate groups: **0**.
Near-duplicate original pairs (dHash ≤ 6): **14**.
- intact/1 (2).JPG ↔ intact/1.JPG (d=4, intact/intact)
- intact/11 (2).JPG ↔ intact/11.JPG (d=6, intact/intact)
- intact/12 (2).JPG ↔ intact/12.JPG (d=4, intact/intact)
- intact/13.JPG ↔ intact/15 (2).JPG (d=6, intact/intact)
- intact/15 (2).JPG ↔ intact/18 (2).JPG (d=6, intact/intact)
- intact/15 (2).JPG ↔ intact/7 (2).JPG (d=6, intact/intact)
- intact/15 (2).JPG ↔ intact/8.JPG (d=5, intact/intact)
- intact/18 (2).JPG ↔ intact/18.JPG (d=6, intact/intact)
- intact/3 (2).JPG ↔ intact/4 (2).JPG (d=6, intact/intact)
- intact/8 (2).JPG ↔ intact/8.JPG (d=6, intact/intact)
- intact/8.JPG ↔ intact/9 (2).JPG (d=6, intact/intact)
- Utilities/181.jpg ↔ Utilities/183.jpg (d=4, utility/utility)
- Utilities/186.jpg ↔ Utilities/188.jpg (d=2, utility/utility)
- Utilities/289.jpg ↔ Utilities/301.jpg (d=3, utility/utility)
Augmented-to-original matches: 2239 (high=328, low=1911; high means distance ≤ 10 and margin ≥ 6).
These dHash thresholds do not work well on this set: most augmented images fall in the low-confidence bin (elastic deformation, noise and crops change the hash). Thresholds were not retuned.
High-confidence augmented images per original: min=1, median=1.0, max=4, originals with ≥1 high match=199.
Augmented images with another class also within distance ≤ 10: **373**.

## Licence

No licence, README or citation file is present inside the unzipped folder. The Mendeley record and the Data in Brief paper are outside this tree.

## Against the paper / README

- No `GPR_data` folder in this download. Top level is `cavities`, `Utilities`, `intact`, `augmented_cavities`, `augmented_utilities`, `augmented_intact`.
- Published 2,239 matches the three augmented folders. The 285 originals are present as well, so the tree holds 2,524 images.
- YOLO class id is 0 for both cavities and utilities. VOC names differ: `cavities` and `Utility`. Intact has no boxes.
- Originals in `cavities` and `Utilities` have no annotation files. Only the augmented copies are labelled.
- No per-image antenna frequency is recorded in the files.
