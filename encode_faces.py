#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
 encode_faces.py — Face Encoding Generator
 Project : Prisoners' Identification in a Surveillance Environment
 Author  : Shreyash Dehury
============================================================================

 PURPOSE
 -------
 This utility script scans the `known_faces/` directory for mugshot images,
 computes a 128-dimensional face encoding for every detected face, and
 serialises the encodings + names into a compact pickle file for fast
 loading during real-time (or near-real-time) video processing.

 DIRECTORY CONVENTION
 --------------------
 known_faces/
 ├── Prisoner_A/
 │   ├── front.jpg
 │   ├── left_profile.jpg
 │   └── right_profile.jpg
 ├── Prisoner_B/
 │   └── mugshot.png
 └── ...

 Each sub-folder name becomes the identity label.  Multiple images per
 person improve robustness — especially across pose and lighting changes.

 CROSS-RESOLUTION NOTE
 ---------------------
 Mugshot images are typically captured at high resolution (≥ 1080p),
 whereas CCTV frames are much lower quality (480p or below).  The
 `face_recognition` library internally resizes images to a canonical
 size before computing HOG/CNN features, which provides a degree of
 resolution invariance.  However, for best results:

   • Use clear, front-facing mugshots (at least 250 × 250 px face area).
   • Avoid heavy JPEG compression artefacts.
   • Include a few lower-quality images per person if available, so the
     encoding set already spans the quality distribution.

 USAGE
 -----
   python encode_faces.py                          # defaults
   python encode_faces.py --dataset known_faces    # explicit path
   python encode_faces.py --output encodings.pkl   # custom output
   python encode_faces.py --model cnn              # GPU-accelerated

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


# ── Supported image extensions ──────────────────────────────────────────────
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".pgm", ".ppm", ".pbm"}


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments with sensible defaults."""
    ap = argparse.ArgumentParser(
        description="Generate 128-d face encodings from mugshot images."
    )
    ap.add_argument(
        "-d", "--dataset",
        type=str,
        default="known_faces",
        help="Path to the directory of known-face images (default: known_faces/)",
    )
    ap.add_argument(
        "-o", "--output",
        type=str,
        default="encodings.pkl",
        help="Output pickle file for serialised encodings (default: encodings.pkl)",
    )
    ap.add_argument(
        "-m", "--model",
        type=str,
        default="hog",
        choices=["hog", "cnn"],
        help=(
            "Face detection model to use.  'hog' is faster on CPU; "
            "'cnn' is more accurate but requires a CUDA-capable GPU "
            "(default: hog)"
        ),
    )
    ap.add_argument(
        "-j", "--jitters",
        type=int,
        default=1,
        help=(
            "Number of times to re-sample the face when computing "
            "encodings.  Higher values are more accurate but slower "
            "(default: 1; recommended for production: 5–10)"
        ),
    )
    ap.add_argument(
        "--upscale",
        type=float,
        default=0.0,
        help=(
            "Upscale factor for small images before face detection. "
            "Use 2.0-3.0 for tiny datasets like AT&T/ORL (92x112). "
            "0 = auto (upscales if the image is smaller than 250px "
            "on its shortest side).  Default: 0 (auto)."
        ),
    )
    return ap.parse_args()


def discover_images(dataset_path: str) -> list[tuple[str, str]]:
    """
    Walk the dataset directory and return a list of (label, image_path) tuples.

    The label is derived from the immediate parent folder name.  Files
    placed directly inside the dataset root are labelled "Unknown".

    Parameters
    ----------
    dataset_path : str
        Root directory containing sub-folders of face images.

    Returns
    -------
    list of (str, str)
        Each element is (person_name, absolute_image_path).
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
            # Label = parent folder name (one level above the image)
            label = os.path.basename(root)
            if label == os.path.basename(str(dataset)):
                label = "Unknown"
            results.append((label, filepath))

    return results


def encode_dataset(
    image_list: list[tuple[str, str]],
    model: str = "hog",
    num_jitters: int = 1,
    upscale: float = 0.0,
) -> dict:
    """
    Compute 128-d face encodings for every image in the list.

    Parameters
    ----------
    image_list : list of (str, str)
        (label, image_path) tuples from `discover_images`.
    model : str
        Face detection backend — "hog" (CPU) or "cnn" (GPU).
    num_jitters : int
        Re-sampling count per face for encoding stability.

    Returns
    -------
    dict
        {"encodings": [np.ndarray, ...], "names": [str, ...]}
    """
    known_encodings: list[np.ndarray] = []
    known_names: list[str] = []
    skipped = 0
    MIN_FACE_DIM = 250  # pixels — minimum for reliable HOG detection

    total = len(image_list)
    print(f"[INFO] Processing {total} image(s) …\n")

    for idx, (name, img_path) in enumerate(image_list, start=1):
        print(f"  [{idx}/{total}]  {name:20s} ← {os.path.basename(img_path)}", end="", flush=True)

        # Load image in RGB (face_recognition expects RGB order)
        image_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            print("  [WARN] Could not read image — skipping.")
            skipped += 1
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Auto-upscale small images for reliable face detection
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

        # Detect face locations, then compute encodings
        boxes = face_recognition.face_locations(image_rgb, model=model)

        if len(boxes) == 0:
            print("  [WARN] No face detected — skipping.")
            skipped += 1
            continue

        if len(boxes) > 1:
            print(f"  [WARN] {len(boxes)} faces detected; encoding ALL of them.")
        else:
            print()  # newline after compact progress

        encodings = face_recognition.face_encodings(
            image_rgb, known_face_locations=boxes, num_jitters=num_jitters
        )

        for enc in encodings:
            known_encodings.append(enc)
            known_names.append(name)

    print(f"\n[INFO] Encoding complete.")
    print(f"       Total encodings : {len(known_encodings)}")
    print(f"       Unique persons  : {len(set(known_names))}")
    print(f"       Skipped images  : {skipped}")

    return {"encodings": known_encodings, "names": known_names}


def save_encodings(data: dict, output_path: str) -> None:
    """Serialise the encoding dictionary to a pickle file."""
    with open(output_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_kb = os.path.getsize(output_path) / 1024
    print(f"[INFO] Encodings saved → {output_path}  ({size_kb:.1f} KB)")


# ── Main ────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_arguments()

    print("=" * 65)
    print(" Face Encoding Generator")
    print(" Prisoners' Identification in a Surveillance Environment")
    print("=" * 65)
    print(f" Dataset    : {args.dataset}")
    print(f" Output     : {args.output}")
    print(f" Model      : {args.model.upper()}")
    print(f" Jitters    : {args.jitters}")
    print("=" * 65 + "\n")

    start = time.time()

    images = discover_images(args.dataset)
    if not images:
        print("[ERROR] No valid images found.  Please populate the dataset folder.")
        print("        Expected structure:")
        print("          known_faces/PersonName/image.jpg")
        sys.exit(1)

    data = encode_dataset(images, model=args.model, num_jitters=args.jitters,
                           upscale=args.upscale)

    if len(data["encodings"]) == 0:
        print("[ERROR] No face encodings were generated.  Check your images.")
        sys.exit(1)

    save_encodings(data, args.output)

    elapsed = time.time() - start
    print(f"[INFO] Done in {elapsed:.2f}s.\n")


if __name__ == "__main__":
    main()
