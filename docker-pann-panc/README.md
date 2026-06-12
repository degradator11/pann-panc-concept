# Docker Wrapper

This folder packages the Rust `pann-panc` app into a small runtime image.

## Build

```powershell
docker-compose -f docker-pann-panc\docker-compose.yml build
```

## Smoke Run

Runs the bundled Iris benchmark:

```powershell
docker-compose -f docker-pann-panc\docker-compose.yml run --rm research-bench
```

## Train From A Mounted Dataset

Your host dataset should use one directory per class:

```text
PetImages_short\
  Cat\
  Dog\
```

Run:

```powershell
$env:TRAIN_DATA="C:\Users\vilex\Downloads\kagglecatsanddogs_5340\PetImages_short"
docker-compose -f docker-pann-panc\docker-compose.yml run --rm research-bench train-pann-image-folder --data /data --out /models/cats-dogs-pann.json --report-out /reports/cats-dogs-pann-train.json --image-size 64 --image-features rich-texture --image-resize center-crop --epochs 12 --intervals 8 --format json
```

Default mounts:

- `${TRAIN_DATA}` -> `/data` read-only
- `${MODELS_DIR:-../pann-panc/models}` -> `/models`
- `${REPORTS_DIR:-../pann-panc/reports}` -> `/reports`

Override `MODELS_DIR` or `REPORTS_DIR` if you want artifacts somewhere else.
