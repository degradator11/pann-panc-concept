from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageFilter

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_MODEL_NAME = "yolo11n.pt"
DEFAULT_MODEL_PATH = Path("models") / DEFAULT_MODEL_NAME
LOCAL_ULTRALYTICS_CONFIG = Path(__file__).resolve().parents[2] / ".ultralytics"
LOCAL_ULTRALYTICS_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = str(LOCAL_ULTRALYTICS_CONFIG)

from ultralytics import YOLO


@dataclass
class CropRecord:
    source_path: str
    source_rel_path: str
    source_class: str
    crop_path: str
    crop_rel_path: str
    detection_label: str
    detection_class_id: int
    score: float
    box_xyxy: list[float]
    crop_box_xyxy: list[float]
    cropper: str
    model: str


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO class-folder cropper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download-model", help="download/cache a YOLO model")
    download.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    download.add_argument("--dest", default=str(DEFAULT_MODEL_PATH))

    crop = subparsers.add_parser("crop", help="crop class-folder images with YOLO detections")
    crop.add_argument("--data", required=True, help="input class-folder dataset")
    crop.add_argument("--out", required=True, help="output folder")
    crop.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="YOLO model path")
    crop.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="model to auto-download")
    crop.add_argument("--allowed-labels", default="", help="comma-separated YOLO labels")
    crop.add_argument("--exclude-classes", default="Eval,Validation,Test")
    crop.add_argument("--match-source-class", action="store_true")
    crop.add_argument("--conf", type=float, default=0.25)
    crop.add_argument("--iou", type=float, default=0.7)
    crop.add_argument("--imgsz", type=int, default=640)
    crop.add_argument("--device", default=None, help="YOLO device, for example cpu or 0")
    crop.add_argument("--expand", type=float, default=0.12)
    crop.add_argument("--output-size", type=int, default=256, help="0 keeps crop size")
    crop.add_argument("--max-crops-per-image", type=int, default=1)
    crop.add_argument("--selection", choices=["best", "largest", "all"], default="best")
    crop.add_argument("--background", choices=["keep", "solid", "blur"], default="keep")
    crop.add_argument("--pad-color", default="128,128,128")
    crop.add_argument("--quality", type=int, default=92)
    crop.add_argument("--overwrite", action="store_true", help="replace existing cropper outputs under --out")

    args = parser.parse_args()
    if args.command == "download-model":
        download_model(args.model_name, Path(args.dest))
    elif args.command == "crop":
        crop_dataset(args)


def download_model(model_name: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"model already exists: {dest}")
        return dest

    previous_cwd = Path.cwd()
    try:
        os.chdir(dest.parent)
        YOLO(model_name)
        downloaded = dest.parent / model_name
        if downloaded.exists() and downloaded.resolve() != dest.resolve():
            shutil.copy2(downloaded, dest)
    finally:
        os.chdir(previous_cwd)

    if not dest.exists():
        raise FileNotFoundError(f"YOLO did not create expected model file: {dest}")
    print(f"model ready: {dest}")
    return dest


def crop_dataset(args: argparse.Namespace) -> None:
    data_path = Path(args.data)
    out_path = Path(args.out)
    model_path = Path(args.model)
    if not data_path.exists():
        raise FileNotFoundError(f"input dataset not found: {data_path}")
    if not model_path.exists():
        download_model(args.model_name, model_path)

    crops_root = out_path / "crops"
    manifest_path = out_path / "crop_manifest.jsonl"
    detections_path = out_path / "detections.json"
    prepare_output(crops_root, manifest_path, detections_path, args.overwrite)
    crops_root.mkdir(parents=True, exist_ok=True)
    allowed_labels = parse_csv(args.allowed_labels)
    excluded_classes = parse_csv(args.exclude_classes)
    pad_color = parse_color(args.pad_color)
    model = YOLO(str(model_path))

    records: list[CropRecord] = []
    skipped_no_detection = 0
    skipped_unreadable = 0
    images_seen = 0
    images_with_detections = 0
    source_class_counts: Counter[str] = Counter()
    detection_label_counts: Counter[str] = Counter()

    for source_class_dir in class_dirs(data_path, excluded_classes):
        source_class = source_class_dir.name
        for image_path in image_files(source_class_dir):
            images_seen += 1
            source_class_counts[source_class] += 1
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as error:  # noqa: BLE001 - keep processing messy datasets.
                print(f"warning: skipped unreadable image {image_path}: {error}")
                skipped_unreadable += 1
                continue

            detections = predict_detections(
                model,
                image_path,
                allowed_labels,
                source_class if args.match_source_class else None,
                args,
            )
            if not detections:
                skipped_no_detection += 1
                continue

            images_with_detections += 1
            for crop_index, detection in enumerate(detections):
                detection_label_counts[detection["label"]] += 1
                crop, crop_box = build_crop(image, detection["box"], args, pad_color)
                if args.output_size > 0:
                    crop = crop.resize((args.output_size, args.output_size), Image.Resampling.LANCZOS)

                rel_source = image_path.relative_to(data_path)
                crop_rel = Path(source_class) / crop_name(image_path, crop_index)
                crop_path = crops_root / crop_rel
                crop_path.parent.mkdir(parents=True, exist_ok=True)
                crop.save(crop_path, quality=args.quality)

                records.append(
                    CropRecord(
                        source_path=str(image_path),
                        source_rel_path=rel_source.as_posix(),
                        source_class=source_class,
                        crop_path=str(crop_path),
                        crop_rel_path=crop_rel.as_posix(),
                        detection_label=detection["label"],
                        detection_class_id=detection["class_id"],
                        score=detection["score"],
                        box_xyxy=round_box(detection["box"]),
                        crop_box_xyxy=round_box(crop_box),
                        cropper="yolo",
                        model=str(model_path),
                    )
                )

    out_path.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for record in records:
            manifest.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    summary = {
        "version": 1,
        "cropper": "yolo",
        "data_path": str(data_path),
        "out_path": str(out_path),
        "crops_path": str(crops_root),
        "manifest_path": str(manifest_path),
        "model": str(model_path),
        "config": {
            "allowed_labels": sorted(allowed_labels),
            "match_source_class": args.match_source_class,
            "conf": args.conf,
            "iou": args.iou,
            "imgsz": args.imgsz,
            "expand": args.expand,
            "output_size": args.output_size,
            "max_crops_per_image": args.max_crops_per_image,
            "selection": args.selection,
            "background": args.background,
            "pad_color": pad_color,
            "overwrite": args.overwrite,
        },
        "totals": {
            "images_seen": images_seen,
            "images_with_detections": images_with_detections,
            "crops_written": len(records),
            "skipped_no_detection": skipped_no_detection,
            "skipped_unreadable": skipped_unreadable,
            "source_class_counts": dict(sorted(source_class_counts.items())),
            "detection_label_counts": dict(sorted(detection_label_counts.items())),
        },
    }
    detections_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["totals"], indent=2))
    print(f"crops: {crops_root}")
    print(f"manifest: {manifest_path}")
    print(f"detections: {detections_path}")


def prepare_output(crops_root: Path, manifest_path: Path, detections_path: Path, overwrite: bool) -> None:
    generated_paths = [crops_root, manifest_path, detections_path]
    existing = [path for path in generated_paths if path.exists()]
    if not existing:
        return

    if not overwrite:
        paths = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"output already contains cropper artifacts: {paths}; use --overwrite or a new --out")

    for path in existing:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def predict_detections(
    model: YOLO,
    image_path: Path,
    allowed_labels: set[str],
    source_class: str | None,
    args: argparse.Namespace,
) -> list[dict]:
    results = model.predict(
        source=str(image_path),
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        verbose=False,
    )
    result = results[0]
    detections = []
    source_label = source_class.lower() if source_class else None
    if result.boxes is None:
        return detections

    for box in result.boxes:
        class_id = int(box.cls.item())
        label = str(model.names[class_id]).lower()
        if allowed_labels and label not in allowed_labels:
            continue
        if source_label and label != source_label:
            continue
        xyxy = box.xyxy[0].tolist()
        score = float(box.conf.item())
        area = max(0.0, xyxy[2] - xyxy[0]) * max(0.0, xyxy[3] - xyxy[1])
        detections.append(
            {
                "class_id": class_id,
                "label": label,
                "score": score,
                "box": [float(value) for value in xyxy],
                "area": area,
                "rank_score": score * max(area, 1.0),
            }
        )

    if args.selection == "largest":
        detections.sort(key=lambda item: (item["area"], item["score"]), reverse=True)
    elif args.selection == "best":
        detections.sort(key=lambda item: (item["rank_score"], item["score"]), reverse=True)
    else:
        detections.sort(key=lambda item: item["score"], reverse=True)

    limit = max(1, args.max_crops_per_image)
    if args.selection == "all":
        return detections[:limit]
    return detections[:limit]


def build_crop(
    image: Image.Image,
    detection_box: list[float],
    args: argparse.Namespace,
    pad_color: tuple[int, int, int],
) -> tuple[Image.Image, list[float]]:
    width, height = image.size
    crop_box = expanded_square_box(detection_box, width, height, args.expand)
    crop = image.crop(tuple(round(value) for value in crop_box))
    crop = pad_to_square(crop, pad_color)

    if args.background == "solid":
        crop = apply_box_background(crop, detection_box, crop_box, pad_color)
    elif args.background == "blur":
        crop = apply_blur_background(crop, detection_box, crop_box)

    return crop, crop_box


def expanded_square_box(box: list[float], width: int, height: int, expand: float) -> list[float]:
    x1, y1, x2, y2 = box
    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)
    side = max(box_width, box_height) * (1.0 + max(0.0, expand) * 2.0)
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    crop_x1 = center_x - side / 2.0
    crop_y1 = center_y - side / 2.0
    crop_x2 = center_x + side / 2.0
    crop_y2 = center_y + side / 2.0

    if crop_x1 < 0:
        crop_x2 -= crop_x1
        crop_x1 = 0
    if crop_y1 < 0:
        crop_y2 -= crop_y1
        crop_y1 = 0
    if crop_x2 > width:
        crop_x1 -= crop_x2 - width
        crop_x2 = width
    if crop_y2 > height:
        crop_y1 -= crop_y2 - height
        crop_y2 = height

    return [
        max(0.0, crop_x1),
        max(0.0, crop_y1),
        min(float(width), crop_x2),
        min(float(height), crop_y2),
    ]


def pad_to_square(image: Image.Image, pad_color: tuple[int, int, int]) -> Image.Image:
    width, height = image.size
    side = max(width, height)
    if width == height:
        return image
    canvas = Image.new("RGB", (side, side), pad_color)
    canvas.paste(image, ((side - width) // 2, (side - height) // 2))
    return canvas


def apply_box_background(
    crop: Image.Image,
    detection_box: list[float],
    crop_box: list[float],
    pad_color: tuple[int, int, int],
) -> Image.Image:
    mask_box = detection_box_in_crop(detection_box, crop_box, crop.size)
    solid = Image.new("RGB", crop.size, pad_color)
    object_region = crop.crop(mask_box)
    solid.paste(object_region, mask_box)
    return solid


def apply_blur_background(
    crop: Image.Image,
    detection_box: list[float],
    crop_box: list[float],
) -> Image.Image:
    mask_box = detection_box_in_crop(detection_box, crop_box, crop.size)
    blurred = crop.filter(ImageFilter.GaussianBlur(radius=12))
    object_region = crop.crop(mask_box)
    blurred.paste(object_region, mask_box)
    return blurred


def detection_box_in_crop(
    detection_box: list[float],
    crop_box: list[float],
    crop_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    crop_width = max(1.0, crop_box[2] - crop_box[0])
    crop_height = max(1.0, crop_box[3] - crop_box[1])
    output_width, output_height = crop_size
    scale = min(output_width / crop_width, output_height / crop_height)
    used_width = crop_width * scale
    used_height = crop_height * scale
    offset_x = (output_width - used_width) / 2.0
    offset_y = (output_height - used_height) / 2.0
    x1 = offset_x + (detection_box[0] - crop_box[0]) * scale
    y1 = offset_y + (detection_box[1] - crop_box[1]) * scale
    x2 = offset_x + (detection_box[2] - crop_box[0]) * scale
    y2 = offset_y + (detection_box[3] - crop_box[1]) * scale
    return (
        max(0, round(x1)),
        max(0, round(y1)),
        min(output_width, round(x2)),
        min(output_height, round(y2)),
    )


def class_dirs(path: Path, excluded_classes: set[str]) -> list[Path]:
    return sorted(
        child
        for child in path.iterdir()
        if child.is_dir() and child.name.lower() not in excluded_classes
    )


def image_files(path: Path) -> Iterable[Path]:
    return (
        file
        for file in sorted(path.rglob("*"))
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    )


def crop_name(image_path: Path, crop_index: int) -> str:
    return f"{image_path.stem}__crop{crop_index:03d}.jpg"


def parse_csv(value: str) -> set[str]:
    return {part.strip().lower() for part in value.split(",") if part.strip()}


def parse_color(value: str) -> tuple[int, int, int]:
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 3 or any(part < 0 or part > 255 for part in parts):
        raise ValueError("--pad-color must be R,G,B with 0..255 values")
    return parts[0], parts[1], parts[2]


def round_box(box: list[float]) -> list[float]:
    return [round(value, 3) for value in box]


if __name__ == "__main__":
    main()
