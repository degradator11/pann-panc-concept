# PANN/PANC Workspace

This repository is now split into a small production-style layout:

- `pann-panc/` - Rust PANN/PANC research application and documentation.
- `docker-pann-panc/` - Docker image and Compose wrapper for container runs.
- `.git/` - repository metadata, intentionally kept at this root level.
- `push-with-token.local.bat` - local push helper, intentionally kept at this root level and ignored by git.

## Run Locally

```powershell
cd pann-panc
cargo test
cargo run --release --bin research-bench -- pann-iris --format json
```

## Run With Docker

Build the image:

```powershell
docker-compose -f docker-pann-panc\docker-compose.yml build
```

Run the bundled Iris smoke benchmark:

```powershell
docker-compose -f docker-pann-panc\docker-compose.yml run --rm research-bench
```

Mount a local image-folder dataset and train a model:

```powershell
$env:TRAIN_DATA="C:\Users\vilex\Downloads\kagglecatsanddogs_5340\PetImages_short"
docker-compose -f docker-pann-panc\docker-compose.yml run --rm research-bench train-pann-image-folder --data /data --out /models/cats-dogs-pann.json --report-out /reports/cats-dogs-pann-train.json --image-size 64 --image-features rich-texture --image-resize center-crop --epochs 12 --intervals 8 --format json
```

Outputs are written to `pann-panc/models/` and `pann-panc/reports/` by default.

See `pann-panc/README.md` for full model/search/report artifact details.
