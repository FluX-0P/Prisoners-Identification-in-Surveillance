#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
 encode_faces.py — Face Encoding Generator  (v2 — Dual-Mode)
 Project : Prisoners' Identification in a Surveillance Environment
 Author  : Shreyash Dehury
============================================================================

 PURPOSE
 -------
 Generate and manage a pickle-based database of 128-dimensional face
 encodings.  Two distinct operational modes are supported:

   Mode A — Single Append  (`--add-image`)
       Detect a face in one image, derive the identity label from the
       filename, and safely append the encoding to an existing (or new)
       `.pkl` database.

   Mode B — Batch Creation  (`--dataset`)
       Scan a directory tree of mugshot images, compute encodings for
       every detected face, and write a brand-new `.pkl` database
       (overwrites any file at the output path).

 DIRECTORY CONVENTION  (Batch Mode)
 --------------------
 known_faces/
 ├── Prisoner_A/
 │   ├── front.jpg
 │   ├── left_profile.jpg
 │   └── right_profile.jpg
 ├── Prisoner_B/
 │   └── mugshot.png
 └── loose_image.jpg          ← label derived from filename

 Each sub-folder name becomes the identity label.  Files placed directly
 inside the dataset root derive their label from the filename (stem).

 USAGE EXAMPLES
 --------------
   # Batch creation (overwrites output file)
   python encode_faces.py -d known_faces -o encodings.pkl

   # Single append (creates or safely appends)
   python encode_faces.py --add-image mugshots/Shreyash_Dehury.jpg -o encodings.pkl

   # Batch with extra options
   python encode_faces.py -d known_faces -o encodings.pkl -j 5 --upscale 2.5

============================================================================
"""

import os
import sys
import pickle
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import face_recognition
from mtcnn import MTCNN


# ── Supported image extensions ──────────────────────────────────────────────
VALID_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif",
    ".webp", ".pgm", ".ppm", ".pbm",
}

MIN_FACE_DIM = 250  # px — minimum for reliable MTCNN detection


# ════════════════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════════════════

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for dual-mode operation."""
    ap = argparse.ArgumentParser(
        description=(
            "Generate 128-d face encodings.  Use --add-image for single "
            "append or --dataset for batch creation."
        ),
    )

    # ── Mutually informative (not exclusive — we branch in main) ────────
    ap.add_argument(
        "-d", "--dataset",
        type=str,
        default="known_faces",
        help="Path to a folder of images (Mode B — Batch Creation).  Default: known_faces/",
    )
    ap.add_argument(
        "--add-image",
        type=str,
        default=None,
        help="Path to a single image file (Mode A — Single Append).",
    )
    ap.add_argument(
        "-o", "--output",
        type=str,
        default="encodings.pkl",
        help="Path to the .pkl file to create or update (default: encodings.pkl).",
    )

    # ── Shared tuning knobs ─────────────────────────────────────────────
    ap.add_argument(
        "-m", "--model",
        type=str,
        default="mtcnn",
        choices=["mtcnn", "hog", "cnn"],
        help=(
            "Face detection model.  'mtcnn' (default) handles pose "
            "variation best.  Legacy 'hog'/'cnn' options kept for "
            "compatibility."
        ),
    )
    ap.add_argument(
        "--min-confidence",
        type=float,
        default=0.85,
        help="Minimum MTCNN detection confidence (0.0–1.0).  Default: 0.85.",
    )
    ap.add_argument(
        "-j", "--jitters",
        type=int,
        default=1,
        help=(
            "Re-sampling count per face for encoding stability "
            "(default: 1; recommended for production: 5–10)."
        ),
    )
    ap.add_argument(
        "--upscale",
        type=float,
        default=0.0,
        help=(
            "Upscale factor for small images before face detection. "
            "0 = auto (upscales if the image is < 250 px on its "
            "shortest side).  Default: 0 (auto)."
        ),
    )

    args = ap.parse_args()

    return args


# ════════════════════════════════════════════════════════════════════════════
#  Shared helpers
# ════════════════════════════════════════════════════════════════════════════

def _prepare_image(img_path: str, upscale: float) -> np.ndarray | None:
    """
    Read an image from disk and return it as an RGB numpy array,
    optionally upscaled.  Returns *None* on read failure.
    """
    image_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return None

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    h, w = image_rgb.shape[:2]
    if upscale > 0:
        scale = upscale
    elif min(h, w) < MIN_FACE_DIM:
        scale = MIN_FACE_DIM / min(h, w)
    else:
        scale = 1.0

    if scale > 1.0:
        image_rgb = cv2.resize(
            image_rgb, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_LINEAR,
        )

    return image_rgb


def _detect_faces(
    image_rgb: np.ndarray,
    model: str,
    mtcnn_detector: MTCNN | None,
    min_confidence: float,
) -> list[tuple[int, int, int, int]]:
    """
    Return face bounding boxes as a list of (top, right, bottom, left)
    tuples — the format expected by `face_recognition.face_encodings`.
    """
    if model == "mtcnn":
        assert mtcnn_detector is not None
        detections = mtcnn_detector.detect_faces(image_rgb)
        boxes: list[tuple[int, int, int, int]] = []
        for face in detections:
            if face["confidence"] < min_confidence:
                continue
            x, y, bw, bh = face["box"]
            x, y = max(0, x), max(0, y)
            boxes.append((y, x + bw, y + bh, x))
        return boxes

    return face_recognition.face_locations(image_rgb, model=model)


def _load_pkl(path: str) -> dict:
    """
    Safely load an existing encodings `.pkl` file.

    Returns a dict with keys ``"encodings"`` and ``"names"`` (empty lists
    when the file does not exist or cannot be read).
    """
    if not os.path.isfile(path):
        return {"encodings": [], "names": []}

    try:
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        # Minimal validation
        if (
            isinstance(data, dict)
            and "encodings" in data
            and "names" in data
        ):
            return data
    except Exception as exc:
        print(f"[WARN] Could not parse existing pkl ({exc}); starting fresh.")

    return {"encodings": [], "names": []}


def _save_pkl(data: dict, path: str) -> None:
    """Atomically serialise *data* to a pickle file."""
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "wb") as fh:
            pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
        # Atomic rename — prevents a half-written file on crash
        if os.path.exists(path):
            os.replace(tmp_path, path)
        else:
            os.rename(tmp_path, path)
    except Exception:
        # Clean up temp file on failure
        if os.path.isfile(tmp_path):
            os.remove(tmp_path)
        raise

    size_kb = os.path.getsize(path) / 1024
    print(f"[INFO] Encodings saved → {path}  ({size_kb:.1f} KB)")


# ════════════════════════════════════════════════════════════════════════════
#  Mode A — Single Append
# ════════════════════════════════════════════════════════════════════════════

def single_append(args: argparse.Namespace) -> None:
    """
    Detect the face in a single image, derive the identity label from its
    filename, and append the encoding to the `.pkl` database.  If the
    database does not exist it is created.
    """
    img_path = args.add_image

    # ── Validate image path ─────────────────────────────────────────────
    if not os.path.isfile(img_path):
        print(f"[ERROR] Image not found: {img_path}")
        sys.exit(1)

    ext = Path(img_path).suffix.lower()
    if ext not in VALID_EXTENSIONS:
        print(f"[ERROR] Unsupported image format '{ext}'.")
        sys.exit(1)

    # ── Derive identity from filename ───────────────────────────────────
    name = Path(img_path).stem          # e.g. "Shreyash_Dehury"
    print(f"[INFO] Identity derived from filename: {name}")

    # ── Read & prepare ──────────────────────────────────────────────────
    image_rgb = _prepare_image(img_path, args.upscale)
    if image_rgb is None:
        print(f"[ERROR] Could not read image: {img_path}")
        sys.exit(1)

    # ── Detect ──────────────────────────────────────────────────────────
    use_mtcnn = (args.model == "mtcnn")
    mtcnn_detector = MTCNN() if use_mtcnn else None
    if use_mtcnn:
        print(f"[INFO] MTCNN detector initialised  (min_confidence={args.min_confidence})")

    boxes = _detect_faces(image_rgb, args.model, mtcnn_detector, args.min_confidence)

    if len(boxes) == 0:
        print("[ERROR] No face detected in the image — aborting to protect the database.")
        sys.exit(1)

    if len(boxes) > 1:
        print(f"[WARN] {len(boxes)} faces found; using the first (highest-confidence) detection.")
        boxes = [boxes[0]]

    # ── Encode ──────────────────────────────────────────────────────────
    encodings = face_recognition.face_encodings(
        image_rgb, known_face_locations=boxes, num_jitters=args.jitters,
    )

    if len(encodings) == 0:
        print("[ERROR] face_recognition could not compute an encoding — aborting.")
        sys.exit(1)

    # ── Load → Append → Save ────────────────────────────────────────────
    data = _load_pkl(args.output)
    prev_count = len(data["encodings"])

    data["encodings"].append(encodings[0])
    data["names"].append(name)

    _save_pkl(data, args.output)
    print(f"[INFO] Appended 1 encoding for '{name}'  "
          f"(database: {prev_count} → {len(data['encodings'])} entries).")


# ════════════════════════════════════════════════════════════════════════════
#  Mode B — Batch Creation
# ════════════════════════════════════════════════════════════════════════════

def discover_images(dataset_path: str) -> list[tuple[str, str]]:
    """
    Walk the dataset directory and return a list of (label, image_path).

    * Sub-folder images → label = sub-folder name.
    * Loose files in the dataset root → label = filename stem.
    """
    results: list[tuple[str, str]] = []
    dataset = Path(dataset_path).resolve()

    if not dataset.exists():
        print(f"[ERROR] Dataset directory not found: {dataset}")
        sys.exit(1)

    for root, _dirs, files in os.walk(dataset):
        for filename in files:
            ext = Path(filename).suffix.lower()
            if ext not in VALID_EXTENSIONS:
                continue

            filepath = os.path.join(root, filename)
            parent = os.path.basename(root)

            if parent == dataset.name:
                # Loose file in dataset root — use filename as label
                label = Path(filename).stem
            else:
                label = parent

            results.append((label, filepath))

    return results


def batch_creation(args: argparse.Namespace) -> None:
    """
    Scan the provided folder for images, compute encodings, and write a
    brand-new `.pkl` database (any existing file is overwritten).
    """
    dataset_path = args.dataset

    images = discover_images(dataset_path)
    if not images:
        print("[ERROR] No valid images found.  Expected structure:")
        print("          <dataset>/PersonName/image.jpg")
        print("        or loose images directly inside <dataset>/")
        sys.exit(1)

    # ── Initialise detector once ────────────────────────────────────────
    use_mtcnn = (args.model == "mtcnn")
    mtcnn_detector = MTCNN() if use_mtcnn else None
    if use_mtcnn:
        print(f"[INFO] MTCNN detector initialised  (min_confidence={args.min_confidence})")

    known_encodings: list[np.ndarray] = []
    known_names: list[str] = []
    skipped = 0
    total = len(images)
    print(f"[INFO] Processing {total} image(s) …\n")

    for idx, (name, img_path) in enumerate(images, start=1):
        print(
            f"  [{idx}/{total}]  {name:20s} ← {os.path.basename(img_path)}",
            end="", flush=True,
        )

        image_rgb = _prepare_image(img_path, args.upscale)
        if image_rgb is None:
            print("  [WARN] Could not read image — skipping.")
            skipped += 1
            continue

        boxes = _detect_faces(image_rgb, args.model, mtcnn_detector, args.min_confidence)

        if len(boxes) == 0:
            print("  [WARN] No face detected — skipping.")
            skipped += 1
            continue

        if len(boxes) > 1:
            print(f"  [WARN] {len(boxes)} faces detected; encoding ALL of them.")
        else:
            print()  # newline after compact progress line

        encodings = face_recognition.face_encodings(
            image_rgb, known_face_locations=boxes, num_jitters=args.jitters,
        )

        for enc in encodings:
            known_encodings.append(enc)
            known_names.append(name)

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n[INFO] Encoding complete.")
    print(f"       Total encodings : {len(known_encodings)}")
    print(f"       Unique persons  : {len(set(known_names))}")
    print(f"       Skipped images  : {skipped}")

    if len(known_encodings) == 0:
        print("[ERROR] No face encodings were generated.  Check your images.")
        sys.exit(1)

    data = {"encodings": known_encodings, "names": known_names}
    _save_pkl(data, args.output)


# ════════════════════════════════════════════════════════════════════════════
#  Main entry-point
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_arguments()

    # ── Determine mode ──────────────────────────────────────────────────
    if args.add_image:
        mode = "Single Append"
    else:
        mode = "Batch Creation"

    # ── Banner ──────────────────────────────────────────────────────────
    print("=" * 65)
    print(" Face Encoding Generator  (v2 — Dual-Mode)")
    print(" Prisoners' Identification in a Surveillance Environment")
    print("=" * 65)
    print(f" Mode       : {mode}")
    if args.add_image:
        print(f" Image      : {args.add_image}")
    else:
        print(f" Dataset    : {args.dataset}")
    print(f" Output     : {args.output}")
    print(f" Model      : {args.model.upper()}")
    print(f" Min conf.  : {args.min_confidence}")
    print(f" Jitters    : {args.jitters}")
    print("=" * 65 + "\n")

    start = time.time()

    if mode == "Single Append":
        single_append(args)
    else:
        batch_creation(args)

    elapsed = time.time() - start
    print(f"[INFO] Done in {elapsed:.2f}s.\n")


if __name__ == "__main__":
    main()
