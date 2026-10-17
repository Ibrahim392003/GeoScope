# GeoScope

**Explainable land-use classification of satellite imagery.**
GeoScope is a web app that classifies Sentinel-2 satellite patches into 10 land-use
classes using the [EuroSAT](https://github.com/phelber/EuroSAT) dataset, benchmarks
three deep-learning architectures (ResNet-18, EfficientNet, ViT) in PyTorch, and
returns real-time predictions together with Grad-CAM heatmaps that show *which regions
of the image drove the decision*.

> Course project - Track A (Application Development). Group of 2.

## Project status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Problem definition | Done (see `report/`) |
| 2 | Data acquisition and preprocessing | Done (see `data_scripts/`) |
| 3 | Model training and benchmarking | Planned |
| 4 | Grad-CAM explainability | Planned |
| 5 | Web app deployment | Planned |

## Dataset

EuroSAT (RGB): 27,000 labelled 64x64 Sentinel-2 patches, 10 classes, 2,000-3,000
images per class, 34 European countries.
Helber et al., *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and
Land Cover Classification*, IEEE JSTARS, 2019.

Classes: AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture,
PermanentCrop, Residential, River, SeaLake.

The raw data is **not** stored in this repository (`data/` is git-ignored). Run the
download script below to fetch it.

## Repository structure

```
GeoScope/
|-- data_scripts/
|   |-- common.py           # paths, constants, helpers
|   |-- download_data.py    # data collection (torchvision) + download log
|   |-- clean_data.py       # corrupt files, duplicates, degenerate images, outliers
|   |-- preprocess.py       # label encoding, stratified split, normalization, transforms
|   `-- eda.py              # exploratory data analysis charts -> figures/
|-- figures/                # EDA charts used in the report
|-- report/                 # LaTeX technical report + auto-generated stats/logs
|-- requirements.txt
`-- README.md
```

## Reproducing Phase 2

```bash
pip install -r requirements.txt

python data_scripts/download_data.py   # 1. download EuroSAT into data/
python data_scripts/clean_data.py      # 2. audit data, write data/clean_manifest.csv
python data_scripts/preprocess.py      # 3. encode labels, split 70/15/15, compute stats
python data_scripts/eda.py             # 4. generate charts in figures/
```

Build the report:

```bash
cd report && pdflatex report.tex && pdflatex report.tex
```

### Cleaning policy

Raw images are never modified. `clean_data.py` writes a manifest recording what happened
to every file:

| Issue | Detection | Action |
|-------|-----------|--------|
| Corrupt / unreadable file | PIL `verify()` + decode | Removed |
| Wrong mode or size (not 64x64 RGB) | Header check | Removed |
| Exact duplicate | MD5 of decoded pixels | Keep first, remove rest |
| Same image under two labels | MD5 across classes | All copies removed |
| Blank no-data patch | Flat and pure black/white | Removed (flat water/forest patches are kept) |
| Near-duplicate | 64-bit dHash, Hamming distance <= 3 | Flagged only |
| Statistical outlier | 3 x IQR per class on brightness/contrast/sharpness | Flagged only |

### Feature strategy

- **Scaling:** resize 64x64 -> 224x224, pixel values to [0, 1]
- **Normalization:** ImageNet mean/std for pretrained backbones (EuroSAT train-set statistics are also computed)
- **Encoding:** class names -> integer ids (`data/label_map.json`)
- **Splits:** stratified 70/15/15, fixed seed 42 (`data/splits.csv`)
- **Augmentation (train only):** random horizontal/vertical flips

## Team

| Member | Responsibilities |
|--------|------------------|
| _Member 1_ | _e.g. data cleaning, report_ |
| _Member 2_ | _e.g. EDA, preprocessing, README_ |