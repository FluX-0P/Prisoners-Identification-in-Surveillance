#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
 app.py — Streamlit Web Interface
 Project : Prisoners' Identification in a Surveillance Environment
 Author  : Shreyash Dehury
============================================================================

 Launch:
   streamlit run app.py

============================================================================
"""

import os
import sys
import glob
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import streamlit as st

# ── Page config (must be the first Streamlit call) ──────────────────────────
st.set_page_config(
    page_title="Prisoner Identification System",
    page_icon="🔍",
    layout="wide",
)

# ── Import project modules ──────────────────────────────────────────────────
from process_image import process_image, ImageResult  # noqa: E402


# ── Constants ───────────────────────────────────────────────────────────────
ENCODINGS_PATH = "encodings.pkl"
TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".tmp")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".pgm", ".ppm", ".pbm")
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
ALLOWED_EXTENSIONS = [
    "jpg", "jpeg", "png", "bmp", "tiff", "webp", "pgm", "ppm", "pbm",
    "mp4", "avi", "mov", "mkv",
]

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs("output", exist_ok=True)


# ── Helpers ─────────────────────────────────────────────────────────────────
def save_uploaded_file(uploaded_file) -> str:
    """Save a Streamlit UploadedFile to a temp path and return the path."""
    suffix = Path(uploaded_file.name).suffix
    dest = os.path.join(TEMP_DIR, f"upload_{int(time.time())}{suffix}")
    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dest


def get_latest_output_video() -> str | None:
    """Return the path to the most recently created tracked_*.mp4."""
    candidates = sorted(glob.glob("output/tracked_*.mp4"), key=os.path.getmtime)
    return candidates[-1] if candidates else None


def run_video_processing(video_path: str, tolerance: float) -> tuple[bool, str, str | None]:
    """
    Run main_tracker.py as a subprocess.

    Returns (success, console_output, output_video_path).
    """
    cmd = [
        sys.executable, "main_tracker.py",
        "-v", video_path,
        "-t", str(tolerance),
        "--save",
    ]

    # Set env so the tracker runs in headless mode with UTF-8 output
    env = os.environ.copy()
    env["COLAB_RELEASE_TAG"] = "streamlit"  # Trick headless detection
    env["PYTHONIOENCODING"] = "utf-8"       # Prevent cp1252 Unicode errors

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10-minute timeout
            env=env,
        )
        output = result.stdout + result.stderr
        success = result.returncode == 0

        out_video = get_latest_output_video() if success else None
        return success, output, out_video

    except subprocess.TimeoutExpired:
        return False, "ERROR: Processing timed out after 10 minutes.", None
    except Exception as e:
        return False, f"ERROR: {e}", None


# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ System Settings")

    tolerance = st.slider(
        "Recognition Tolerance",
        min_value=0.10,
        max_value=1.00,
        value=0.50,
        step=0.01,
        help=(
            "Controls how strict the face matching is.\n\n"
            "**Lower** (e.g. 0.40) → stricter, fewer false positives, "
            "may miss true matches.\n\n"
            "**Higher** (e.g. 0.60) → more lenient, captures more matches, "
            "but increases false acceptance risk.\n\n"
            "**0.50** is the default balanced trade-off for cross-resolution "
            "scenarios (high-res mugshots vs. low-res CCTV)."
        ),
    )

    st.divider()

    st.markdown(
        "**Current threshold:** `{:.2f}`\n\n"
        "Adjusting this value changes the **False Acceptance Rate (FAR)** "
        "and **False Rejection Rate (FRR)** of the system.".format(tolerance)
    )

    st.divider()

    # Check that encodings exist
    if os.path.exists(ENCODINGS_PATH):
        import pickle

        with open(ENCODINGS_PATH, "rb") as f:
            enc_data = pickle.load(f)
        n_enc = len(enc_data.get("encodings", []))
        n_ids = len(set(enc_data.get("names", [])))
        st.success(f"📋 Database: **{n_enc}** encodings, **{n_ids}** persons")
    else:
        st.error(
            "⚠️ `encodings.pkl` not found!\n\n"
            "Run `python encode_faces.py` first."
        )


# ── Main Page ───────────────────────────────────────────────────────────────
st.title("🔍 Prisoners' Identification in a Surveillance Environment")
st.caption("Author: Shreyash Dehury")

st.markdown(
    """
    This system identifies known prisoners in **surveillance footage** or
    **static images** by comparing detected faces against a pre-enrolled
    database of 128-dimensional face encodings.

    It is designed to work across **different resolutions** — matching
    high-resolution mugshots against low-quality CCTV frames using
    Euclidean distance comparison with inverse-distance weighted majority
    voting.

    ---

    **Upload an image or video below to begin identification.**
    """
)

uploaded_file = st.file_uploader(
    "Choose an image or video file",
    type=ALLOWED_EXTENSIONS,
    help="Supported: JPEG, PNG, BMP, WebP (images) — MP4, AVI, MOV, MKV (videos)",
)

if uploaded_file is not None:
    file_ext = Path(uploaded_file.name).suffix.lower()
    file_size_mb = uploaded_file.size / (1024 * 1024)

    st.info(
        f"📁 **{uploaded_file.name}**  —  "
        f"{file_size_mb:.1f} MB  —  "
        f"{'Image' if file_ext in IMAGE_EXTENSIONS else 'Video'}"
    )

    # ── IMAGE PROCESSING ────────────────────────────────────────────────
    if file_ext in IMAGE_EXTENSIONS:
        temp_path = save_uploaded_file(uploaded_file)

        with st.spinner("🔍 Detecting and identifying faces…"):
            try:
                result = process_image(
                    image_path=temp_path,
                    encodings_path=ENCODINGS_PATH,
                    tolerance=tolerance,
                )

                # Convert BGR → RGB for Streamlit display
                annotated_rgb = cv2.cvtColor(
                    result.annotated_image, cv2.COLOR_BGR2RGB
                )

                # ── Results ─────────────────────────────────────────────
                st.subheader("📊 Results")

                col1, col2, col3 = st.columns(3)
                col1.metric("Total Faces", result.total_faces)
                col2.metric("Known Matches", result.known_count)
                col3.metric("Unknown Faces", result.unknown_count)

                # ── Annotated image (capped width) ─────────────────────
                img_col, _ = st.columns([2, 1])
                with img_col:
                    st.image(
                        annotated_rgb,
                        caption="Annotated Image — RED = Known  |  GREEN = Unknown",
                        use_container_width=True,
                    )

                # ── Per-face details ────────────────────────────────────
                if result.faces:
                    st.subheader("🧑 Face Details")
                    for i, face in enumerate(result.faces, start=1):
                        if face.is_known:
                            st.markdown(
                                f"**Face #{i}:** 🔴 **{face.name}** "
                                f"— Confidence: `{face.confidence:.0%}`"
                            )
                        else:
                            st.markdown(f"**Face #{i}:** 🟢 Unknown")

                # ── Download annotated image ────────────────────────────
                _, img_bytes = cv2.imencode(".jpg", result.annotated_image)
                st.download_button(
                    label="⬇️ Download Annotated Image",
                    data=img_bytes.tobytes(),
                    file_name=f"{Path(uploaded_file.name).stem}_identified.jpg",
                    mime="image/jpeg",
                )

            except FileNotFoundError as e:
                st.error(f"❌ {e}")
            except ValueError as e:
                st.error(f"❌ {e}")
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)

    # ── VIDEO PROCESSING ────────────────────────────────────────────────
    elif file_ext in VIDEO_EXTENSIONS:
        temp_path = save_uploaded_file(uploaded_file)

        st.warning(
            "⏳ Video processing runs the full detection pipeline and may "
            "take several minutes depending on video length and hardware. "
            "The output video will be saved automatically."
        )

        with st.spinner("🎬 Processing video — please wait…"):
            success, console_output, out_video = run_video_processing(
                temp_path, tolerance
            )

        # Clean up uploaded temp
        if os.path.exists(temp_path):
            os.remove(temp_path)

        if success and out_video and os.path.exists(out_video):
            st.success("✅ Video processing complete!")

            # ── Show console summary ────────────────────────────────
            with st.expander("📋 Processing Log", expanded=False):
                st.code(console_output, language="text")

            # ── Video preview ───────────────────────────────────────
            st.subheader("🎬 Output Preview")
            st.video(out_video)

            # ── Download button ─────────────────────────────────────
            with open(out_video, "rb") as vf:
                video_bytes = vf.read()

            st.download_button(
                label="⬇️ Download Annotated Video",
                data=video_bytes,
                file_name=f"{Path(uploaded_file.name).stem}_tracked.mp4",
                mime="video/mp4",
            )
        else:
            st.error("❌ Video processing failed.")
            with st.expander("🔍 Error Details", expanded=True):
                st.code(console_output, language="text")

    else:
        st.error(f"❌ Unsupported file type: `{file_ext}`")

else:
    # ── Empty state ─────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="text-align: center; padding: 60px 0; opacity: 0.5;">
            <h3>Upload an image or video to get started</h3>
            <p>Drag and drop or click the uploader above</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Footer ──────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Prisoners' Identification in a Surveillance Environment  •  "
    "Shreyash Dehury  •  "
    "Powered by dlib / face_recognition / Streamlit"
)
