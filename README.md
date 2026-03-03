# Beech-Wood-Vision-System

Automated inspection system for detecting defects (knots, cracks, stains, rot)
in beech-wood samples using computer vision.

---

## Project structure

```
Beech-Wood-Vision-System/
├── data/
│   ├── raw/              # Step 1 & 3 – raw captured images
│   │   ├── healthy/      #   defect-free beech-wood samples
│   │   └── defective/    #   samples with defects
│   ├── clean/            # Step 4 – filtered & resized images
│   ├── labeled/          # Step 5 – annotation exports (bounding boxes / masks)
│   └── dataset/          # Step 6 – final train / val / test split
│       ├── train/{images,labels}/
│       ├── val/{images,labels}/
│       └── test/{images,labels}/
├── src/
│   ├── capture/
│   │   └── data_capture.py       # Step 1 – live camera capture
│   ├── preprocessing/
│   │   └── data_cleaning.py      # Step 4 – blur/brightness filtering + resize
│   └── data/
│       └── split_dataset.py      # Step 6 – train/val/test split
├── tests/                        # pytest unit & integration tests
├── dvc.yaml                      # DVC pipeline definition
├── params.yaml                   # Pipeline hyperparameters
└── requirements.txt
```

---

## Quick-start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For DVC (large-file management) install the appropriate remote back-end:

```bash
pip install dvc[gdrive]   # Google Drive
# or
pip install dvc[s3]       # AWS S3
```

---

## Data pipeline

### Step 1 – Data capture

Connect a camera and capture frames to a folder.
Press **SPACE** to save a frame; press **Q / ESC** to quit.

```bash
# Capture healthy samples
python src/capture/data_capture.py \
    --output data/raw/healthy \
    --camera 0

# Capture defective samples
python src/capture/data_capture.py \
    --output data/raw/defective \
    --trigger motion          # automatic trigger on motion
```

**Key options**

| Flag | Default | Description |
|---|---|---|
| `--camera` | `0` | Camera device index |
| `--output` | `data/raw` | Output folder |
| `--trigger` | `space` | `space` or `motion` |
| `--motion-threshold` | `0.5` | % changed pixels for motion trigger |
| `--width` / `--height` | `1920` / `1080` | Requested resolution |

---

### Step 2 – Environment standardisation

For reproducible results:

* Fix the camera to a rigid mount (tripod or arm) at a constant distance from the wood.
* Use diffuse, even lighting to avoid shadows that could be mistaken for cracks.
* Set a fixed white balance and exposure on the camera before starting capture.

---

### Step 3 – Data collection

Collect at least **two classes** of samples:

| Class | Description | Sub-folder |
|---|---|---|
| healthy | Clean, defect-free beech wood | `data/raw/healthy/` |
| defective | Knots (light/dark/rotten), cracks, false heartwood, rot, staining (blue stain) | `data/raw/defective/` |

Aim for diverse lighting angles, wood orientations and defect severities.

---

### Step 4 – Data cleaning

Filter blurry, too-dark or too-bright images and resize to the model input size.

```bash
python src/preprocessing/data_cleaning.py \
    --input  data/raw \
    --output data/clean \
    --size   640
```

**Key options**

| Flag | Default | Description |
|---|---|---|
| `--size` | `640` | Output resolution (square) |
| `--blur-threshold` | `100` | Laplacian variance below this → rejected |
| `--min-brightness` | `20` | Mean pixel below this → rejected |
| `--max-brightness` | `235` | Mean pixel above this → rejected |

---

### Step 5 – Labelling / Annotation

Import the cleaned images into an annotation tool and draw **bounding boxes** or
**polygon masks** around each defect, assigning the appropriate category.

Recommended tools:

* [CVAT](https://cvat.ai/) (self-hosted or cloud)
* [Label Studio](https://labelstud.io/)
* [Roboflow](https://roboflow.com/)

Export annotations in **YOLO format** (one `.txt` per image) and save them to
`data/labeled/`.

Defect categories to annotate:

| Label | Description |
|---|---|
| `knot_light` | Light / healthy knot |
| `knot_dark` | Dark / dry knot |
| `knot_rotten` | Rotten / loose knot |
| `crack` | Surface or structural crack |
| `false_heartwood` | False heartwood discolouration |
| `rot` | Wood rot / decay |
| `stain` | Blue stain or other discolouration |

---

### Step 6 – Versioning & dataset split

#### Train / val / test split

```bash
python src/data/split_dataset.py \
    --input  data/clean \
    --output data/dataset \
    --train  0.75 \
    --val    0.15 \
    --test   0.10 \
    --seed   42
```

This creates the YOLO-style directory layout under `data/dataset/`.

#### Large-file management with DVC

Image data is **not** stored in Git; only DVC pointer files (`*.dvc`) are committed.

```bash
# Initialise DVC (first time only)
dvc init

# Add a remote storage (e.g. Google Drive folder)
dvc remote add -d myremote gdrive://<FOLDER_ID>

# Track the data directories
dvc add data/raw data/clean data/labeled data/dataset

# Push data to remote
dvc push

# Collaborators pull data with
dvc pull
```

#### Run the full DVC pipeline

```bash
dvc repro
```

Parameters can be tuned in `params.yaml` without editing any Python files.

---

## Running tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Labelling guidelines

* **Do** label every visible defect, even minor ones.
* **Do** use tight bounding boxes that closely enclose the defect.
* **Don't** label the same defect twice.
* **Don't** label wood grain, natural colour variation or tool marks as defects.
