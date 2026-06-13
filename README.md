# PANN/PANC Workspace

This repository is split into a simple local-development layout:

- `pann-panc/` - Rust PANN/PANC research application and documentation.
- `cropper/cropper-yolo/` - standalone YOLO cropper routine for turning raw
  class-folder images into cropped class-folder images.
- `.git/` - repository metadata, intentionally kept at this root level.
- `push-with-token.local.bat` - local push helper, intentionally kept at this root level and ignored by git.

The current direction is **standalone local routines first**:

```text
class-folder images -> train/eval/predict commands -> JSON model/report artifacts
```

We are intentionally not maintaining Docker packaging right now. The next
research step is to prove the cropper/classifier pipeline locally. Once the
pipeline works well enough, we can package the proven routines into
containerized workers for deployment.

## Run Locally

```powershell
cd pann-panc
cargo test
cargo run --release --bin research-bench -- pann-iris --format json
```

Train an image-folder model:

```powershell
cd pann-panc
cargo run --release --bin research-bench -- train-pann-image-folder --data C:\path\to\train --out models\model.json --report-out reports\train-report.json --image-size 64 --image-features rich-texture --image-resize center-crop --epochs 12 --intervals 8 --format json
```

Evaluate that model:

```powershell
cd pann-panc
cargo run --release --bin research-bench -- eval-pann --model models\model.json --data C:\path\to\eval --report-out reports\eval-report.json --format json
```

See `pann-panc/README.md` for full model/search/report artifact details.

## Crop Images First

The first cropper routine is YOLO-based and isolated in its own Python venv:

```powershell
cd cropper\cropper-yolo
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install wheel
.\.venv\Scripts\python.exe -m pip install -e . --no-build-isolation
.\.venv\Scripts\cropper-yolo.exe download-model
.\.venv\Scripts\cropper-yolo.exe crop --data C:\path\to\raw-class-folders --out runs\my-crops --allowed-labels cat,dog --match-source-class --overwrite
```

Then train PANN/PANC on:

```text
cropper\cropper-yolo\runs\my-crops\crops
```
