# YOLO Cropper

Standalone local routine for turning raw class-folder images into cropped
class-folder images before PANN/PANC training.

Input:

```text
PetImages_short/
  Cat/
  Dog/
```

Output:

```text
runs/petimages-short-yolo/
  crops/
    Cat/
    Dog/
  crop_manifest.jsonl
  detections.json
```

## Setup With Local venv

Dependencies should stay isolated in this cropper directory.

```powershell
cd cropper\cropper-yolo
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install wheel
.\.venv\Scripts\python.exe -m pip install -e . --no-build-isolation
.\.venv\Scripts\cropper-yolo.exe download-model
```

The default model is `models/yolo11n.pt`. Model files and generated runs are
ignored by git.

## Crop A Dataset

```powershell
cd cropper\cropper-yolo
.\.venv\Scripts\cropper-yolo.exe crop --data C:\Users\vilex\Downloads\kagglecatsanddogs_5340\PetImages_short --out runs\petimages-short-yolo --allowed-labels cat,dog --match-source-class --conf 0.25 --expand 0.12 --output-size 256 --overwrite
```

Then train PANN/PANC on:

```text
cropper\cropper-yolo\runs\petimages-short-yolo\crops
```

## Notes

- Source folder names remain the class labels. For example, crops from `Cat/`
  are written under `crops/Cat/`.
- Top-level `Eval`, `Validation`, and `Test` folders are ignored by default so
  a train folder can contain a separate eval split.
- `--allowed-labels` filters YOLO detections.
- `--match-source-class` is useful for Cats/Dogs because YOLO labels are also
  `cat` and `dog`.
- `--background solid` fills the expanded area outside the detected box with a
  neutral color. This is a simple box-based background reduction, not a true
  segmentation mask.
- `--overwrite` replaces generated files under `--out`: `crops/`,
  `crop_manifest.jsonl`, and `detections.json`.
