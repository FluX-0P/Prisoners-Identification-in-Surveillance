#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
 process_image.py — Static Image Prisoner Identification
 Project : Prisoners' Identification in a Surveillance Environment
 Author  : Shreyash Dehury
============================================================================

 PURPOSE
 -------
 Identify known prisoners in a single static image (JPEG, PNG, etc.)
 using the same 128-d face-encoding database and matching logic as
 `main_tracker.py`.

 The script:
   1. Loads pre-computed encodings from `encodings.pkl`.
   2. Reads the input image and detects all faces.
   3. Computes 128-d encodings and matches each face against the DB
      using Euclidean distance + inverse-distance weighted majority vote.
   4. Draws RED (match) or GREEN (unknown) bounding boxes with labels.
   5. Saves the annotated image to disk and optionally displays it.

 The core `process_image()` function is importable so a web frontend
 (Flask / FastAPI / Streamlit) can call it directly.

 USAGE
 -----
   python process_image.py -i photo.jpg
   python process_image.py -i photo.jpg -t 0.45
   python process_image.py -i photo.jpg -o output/result.jpg --model cnn
   python process_image.py -i photo.jpg --no-display

============================================================================
"""

import os
import sys
import pickle
import argparse
from pathlib import Path
from dataclasses import dataclass, field

import cv2
import numpy as np
import face_recognition
from mtcnn import MTCNN


# ── Visual constants (identical to main_tracker.py) ─────────────────────────
COLOR_MATCH = (0, 0, 255)        # RED   — known prisoner
COLOR_UNKNOWN = (0, 255, 0)      # GREEN — unregistered face
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.65
FONT_THICKNESS = 2
BOX_THICKNESS = 2


# ── Data classes for structured results (importable by web frontends) ───────
@dataclass
class FaceResult:
    """Result of identification for a single detected face."""
    name: str
    confidence: float
    top: int
    right: int
    bottom: int
    left: int

    @property
    def is_known(self) -> bool:
        return self.name != "Unknown"

    def to_dict(self) -> dict:
        """Serialise to a plain dict (JSON-friendly)."""
        return {
            "name": self.name,
            "confidence": self.confidence,
            "is_known": self.is_known,
            "bbox": {
                "top": self.top,
                "right": self.right,
                "bottom": self.bottom,
                "left": self.left,
            },
        }


@dataclass
class ImageResult:
    """Aggregate result for one processed image."""
    faces: list[FaceResult] = field(default_factory=list)
    annotated_image: np.ndarray | None = None

    @property
    def total_faces(self) -> int:
        return len(self.faces)

    @property
    def known_count(self) -> int:
        return sum(1 for f in self.faces if f.is_known)

    @property
    def unknown_count(self) -> int:
        return self.total_faces - self.known_count

    def to_dict(self) -> dict:
        return {
            "total_faces": self.total_faces,
            "known_count": self.known_count,
            "unknown_count": self.unknown_count,
            "faces": [f.to_dict() for f in self.faces],
        }


# ── Environment helpers ─────────────────────────────────────────────────────
def _has_display() -> bool:
    """Return True if a GUI display is available."""
    if "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ:
        return False
    if os.name == "posix" and not os.environ.get("DISPLAY"):
        return False
    return True


def _has_cuda_gpu() -> bool:
    """Return True if dlib was built with CUDA and a GPU is present."""
    try:
        import dlib
        return dlib.DLIB_USE_CUDA and dlib.cuda.get_num_devices() > 0
    except Exception:
        return False


# ── Core functions (importable) ─────────────────────────────────────────────
def load_encodings(path: str) -> dict:
    """
    Load the face-encoding database produced by `encode_faces.py`.

    Parameters
    ----------
    path : str
        Path to the `.pkl` file.

    Returns
    -------
    dict  {"encodings": list[np.ndarray], "names": list[str]}
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Encoding file not found: {path}\n"
            "Run `python encode_faces.py` first."
        )
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def identify_faces(
    face_encodings: list[np.ndarray],
    known_encodings: list[np.ndarray],
    known_names: list[str],
    tolerance: float,
) -> list[tuple[str, float]]:
    """
    Match detected face encodings against the known database.

    Uses Euclidean distance comparison with inverse-distance weighted
    majority voting — identical logic to `main_tracker.py`.

    Parameters
    ----------
    face_encodings : list of np.ndarray
        128-d encodings of detected faces.
    known_encodings : list of np.ndarray
        Reference encodings from the database.
    known_names : list of str
        Labels corresponding to `known_encodings`.
    tolerance : float
        Maximum Euclidean distance to accept a match (0.0–1.0).

    Returns
    -------
    list of (name, confidence) tuples.
    """
    results: list[tuple[str, float]] = []

    for encoding in face_encodings:
        distances = face_recognition.face_distance(known_encodings, encoding)
        matches = distances <= tolerance

        name = "Unknown"
        confidence = 0.0

        if np.any(matches):
            # Inverse-distance weighted majority vote
            name_counts: dict[str, float] = {}
            for i, m in enumerate(matches):
                if m:
                    n = known_names[i]
                    weight = 1.0 / (distances[i] + 1e-6)
                    name_counts[n] = name_counts.get(n, 0.0) + weight

            name = max(name_counts, key=name_counts.get)
            best_dist = min(distances[matches])
            confidence = round(1.0 - best_dist, 3)

        results.append((name, confidence))

    return results


def draw_annotation(
    image: np.ndarray,
    top: int, right: int, bottom: int, left: int,
    name: str,
    confidence: float,
) -> None:
    """Draw a bounding box and label on the image (in-place)."""
    is_known = name != "Unknown"
    color = COLOR_MATCH if is_known else COLOR_UNKNOWN

    # Bounding box
    cv2.rectangle(image, (left, top), (right, bottom), color, BOX_THICKNESS)

    # Label text
    label = f"{name} ({confidence:.0%})" if is_known else "Unknown"

    # Text background
    (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)
    cv2.rectangle(
        image,
        (left, bottom),
        (left + tw + 10, bottom + th + 14),
        color,
        cv2.FILLED,
    )
    # Text
    cv2.putText(
        image,
        label,
        (left + 5, bottom + th + 8),
        FONT,
        FONT_SCALE,
        (255, 255, 255),
        FONT_THICKNESS,
    )


def process_image(
    image_path: str,
    encodings_path: str = "encodings.pkl",
    tolerance: float = 0.50,
    model: str = "auto",
) -> ImageResult:
    """
    Detect and identify faces in a single static image.

    This is the primary entry point for programmatic use (web frontends,
    notebooks, etc.).

    Parameters
    ----------
    image_path : str
        Path to the input image file.
    encodings_path : str
        Path to the encodings pickle (default: ``encodings.pkl``).
    tolerance : float
        Matching threshold (default: 0.50).
    model : str
        ``"hog"`` (CPU), ``"cnn"`` (GPU), or ``"auto"`` (detect GPU).

    Returns
    -------
    ImageResult
        Structured result containing per-face data and the annotated image.
    """
    # ── Resolve model ───────────────────────────────────────────────────
    if model == "auto":
        model = "cnn" if _has_cuda_gpu() else "hog"

    # ── Load encodings ──────────────────────────────────────────────────
    data = load_encodings(encodings_path)
    known_encodings = data["encodings"]
    known_names = data["names"]

    # ── Read image ──────────────────────────────────────────────────────
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # ── Face detection ──────────────────────────────────────────────────
    boxes = face_recognition.face_locations(image_rgb, model=model)

    # ── Face encoding ───────────────────────────────────────────────────
    encodings = face_recognition.face_encodings(
        image_rgb, known_face_locations=boxes
    )

    # ── Identification ──────────────────────────────────────────────────
    result = ImageResult()

    if encodings:
        matches = identify_faces(
            encodings, known_encodings, known_names, tolerance
        )
    else:
        matches = []

    # ── Build results + draw annotations ────────────────────────────────
    annotated = image_bgr.copy()

    for (top, right, bottom, left), (name, conf) in zip(boxes, matches):
        result.faces.append(
            FaceResult(
                name=name,
                confidence=conf,
                top=top,
                right=right,
                bottom=bottom,
                left=left,
            )
        )
        draw_annotation(annotated, top, right, bottom, left, name, conf)

    result.annotated_image = annotated
    return result


# ── CLI ─────────────────────────────────────────────────────────────────────
def parse_arguments() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Identify known prisoners in a static image.",
    )
    ap.add_argument(
        "-i", "--image",
        type=str,
        required=True,
        help="Path to the input image file (JPEG, PNG, etc.)",
    )
    ap.add_argument(
        "-e", "--encodings",
        type=str,
        default="encodings.pkl",
        help="Path to the encodings pickle file (default: encodings.pkl).",
    )
    ap.add_argument(
        "-t", "--tolerance",
        type=float,
        default=0.50,
        help="Matching tolerance (0.0–1.0).  Default: 0.50.",
    )
    ap.add_argument(
        "-m", "--model",
        type=str,
        default="mtcnn",
        choices=["mtcnn", "hog", "cnn", "auto"],
        help="Detection model: 'mtcnn' (default), 'hog', 'cnn', or 'auto'.",
    )
    ap.add_argument(
        "--min-confidence",
        type=float,
        default=0.85,
        help="Minimum MTCNN detection confidence (0.0–1.0).  Default: 0.85.",
    )
    ap.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help=(
            "Save path for the annotated image.  "
            "Default: output/<input_name>_identified.<ext>"
        ),
    )
    ap.add_argument(
        "--no-display",
        action="store_true",
        help="Skip the GUI display window (useful in headless / Colab).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_arguments()

    print("=" * 55)
    print(" Prisoners' Identification — Static Image")
    print(" Author: Shreyash Dehury")
    print("=" * 55)
    print(f"  Input       : {args.image}")
    print(f"  Encodings   : {args.encodings}")
    print(f"  Tolerance   : {args.tolerance}")
    print(f"  Model       : {args.model}")
    print(f"  Min conf.   : {args.min_confidence}")
    print("=" * 55 + "\n")

    # ── Process ─────────────────────────────────────────────────
    result = process_image(
        image_path=args.image,
        encodings_path=args.encodings,
        tolerance=args.tolerance,
        model=args.model,
        min_confidence=args.min_confidence,
    )

    # ── Console summary ─────────────────────────────────────────────────
    print(f"[INFO] Faces detected : {result.total_faces}")
    print(f"[INFO] Known matches  : {result.known_count}")
    print(f"[INFO] Unknown faces  : {result.unknown_count}")

    for i, face in enumerate(result.faces, start=1):
        status = f"{face.name} ({face.confidence:.0%})" if face.is_known else "Unknown"
        print(f"  Face #{i}  →  {status}")

    # ── Determine output path ───────────────────────────────────────────
    if args.output:
        out_path = args.output
    else:
        os.makedirs("output", exist_ok=True)
        stem = Path(args.image).stem
        ext = Path(args.image).suffix or ".jpg"
        out_path = os.path.join("output", f"{stem}_identified{ext}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, result.annotated_image)
    print(f"\n[INFO] Annotated image saved → {out_path}")

    # ── Display (unless headless or --no-display) ───────────────────────
    show = not args.no_display and _has_display()
    if show:
        win = "Prisoner Identification — Static Image"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        h, w = result.annotated_image.shape[:2]
        disp_h = min(h, 720)
        disp_w = int(w * (disp_h / h))
        cv2.resizeWindow(win, disp_w, disp_h)
        cv2.imshow(win, result.annotated_image)
        print("[INFO] Press any key to close the window.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print("[INFO] Done.\n")


if __name__ == "__main__":
    main()
