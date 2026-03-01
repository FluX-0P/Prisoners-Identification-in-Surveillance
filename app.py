#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
 app.py — Streamlit Web Interface  (v2 — Multi-Tab)
============================================================================

 Launch:
   streamlit run app.py

============================================================================
"""

import os
import sys
import glob
import pickle
import shutil
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path

import cv2
import streamlit as st

# ── Page config (must be the first Streamlit call) ──────────────────────────
st.set_page_config(
    page_title="Criminal Identification System",
    page_icon="🔍",
    layout="wide",
)

# ── Project root (directory where app.py lives) ────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
TEMP_DIR = PROJECT_ROOT / ".tmp"
OUTPUT_DIR = PROJECT_ROOT / "output"

TEMP_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════════════════════════

def _scan_files(pattern: str) -> list[str]:
    """Return a sorted list of files in PROJECT_ROOT matching *pattern*."""
    return sorted(
        str(p.relative_to(PROJECT_ROOT))
        for p in PROJECT_ROOT.glob(pattern)
        if p.is_file()
    )


def _subprocess_env() -> dict:
    """Return a copy of os.environ with headless + UTF-8 flags set."""
    env = os.environ.copy()
    env["COLAB_RELEASE_TAG"] = "streamlit"   # force headless detection
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _save_uploaded(uploaded_file, dest_dir: Path | None = None,
                   name_override: str | None = None) -> Path:
    """Persist a Streamlit UploadedFile to disk and return the path."""
    dest_dir = dest_dir or TEMP_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    fname = name_override or f"upload_{int(time.time())}{Path(uploaded_file.name).suffix}"
    dest = dest_dir / fname
    dest.write_bytes(uploaded_file.getbuffer())
    return dest


def _get_latest_output(glob_pattern: str) -> Path | None:
    """Return the newest file matching *glob_pattern* inside output/."""
    candidates = sorted(OUTPUT_DIR.glob(glob_pattern), key=os.path.getmtime)
    return candidates[-1] if candidates else None


def _run(cmd: list[str], timeout: int = 900) -> tuple[bool, str]:
    """Run a subprocess; return (success, combined stdout+stderr)."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=_subprocess_env(),
            cwd=str(PROJECT_ROOT),
        )
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, f"ERROR: Command timed out after {timeout}s."
    except Exception as exc:
        return False, f"ERROR: {exc}"


# ════════════════════════════════════════════════════════════════════════════
#  Sidebar — Global Settings
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("⚙️ Global Settings")

    tolerance = st.slider(
        "Recognition Tolerance",
        min_value=0.10,
        max_value=1.00,
        value=0.50,
        step=0.01,
        help=(
            "Controls how strict the face matching is.\n\n"
            "**Lower** (e.g. 0.40) → stricter, fewer false positives.\n\n"
            "**Higher** (e.g. 0.60) → more lenient, better recall.\n\n"
            "**0.50** is the default balanced trade-off."
        ),
    )

    enable_sr = st.toggle(
        "Enable Super-Resolution",
        value=False,
        help=(
            "Apply ESPCN ×4 up-scaling on low-resolution face crops "
            "(IPD < 60 px) before computing encodings.  Improves "
            "cross-resolution matching at the cost of speed."
        ),
    )

    st.divider()

    # ── Dynamic file scanners ───────────────────────────────────────────
    pkl_files = _scan_files("*.pkl")
    mp4_files = _scan_files("input_videos/*.mp4") or _scan_files("*.mp4")

    st.markdown(f"**Databases found:** {len(pkl_files)}")
    st.markdown(f"**Videos found:** {len(mp4_files)}")

    if pkl_files:
        # Quick peek at the selected default db
        default_pkl = pkl_files[0]
        try:
            with open(PROJECT_ROOT / default_pkl, "rb") as _f:
                _data = pickle.load(_f)
            _n = len(_data.get("encodings", []))
            _ids = len(set(_data.get("names", [])))
            st.success(f"📋 `{default_pkl}`: **{_n}** encodings, **{_ids}** persons")
        except Exception:
            pass
    else:
        st.warning("No `.pkl` database found — create one in **Tab 3**.")


# ════════════════════════════════════════════════════════════════════════════
#  Main Title
# ════════════════════════════════════════════════════════════════════════════

st.title("🔍 Criminal Identification Using Face Recognition")


# ════════════════════════════════════════════════════════════════════════════
#  Tabs
# ════════════════════════════════════════════════════════════════════════════

tab_video, tab_image, tab_db = st.tabs([
    "🎬 Video Surveillance",
    "🖼️ Image Analysis",
    "🗄️ Database Management",
])


# ────────────────────────────────────────────────────────────────────────────
#  Tab 3 — Database Management
# ────────────────────────────────────────────────────────────────────────────

with tab_db:
    st.subheader("Database Management")

    db_mode = st.radio(
        "Select operation",
        ["Create New Database (ZIP)", "Update Existing Database (Image)"],
        horizontal=True,
    )

    st.divider()

    # ── Mode A: Batch creation from ZIP ─────────────────────────────────
    if db_mode == "Create New Database (ZIP)":
        st.markdown(
            "Upload a **ZIP archive** whose internal structure mirrors the "
            "expected layout:\n"
            "```\n"
            "archive.zip\n"
            "├── PersonA/\n"
            "│   ├── img1.jpg\n"
            "│   └── img2.png\n"
            "├── PersonB/\n"
            "│   └── mugshot.jpg\n"
            "└── loose_image.jpg   ← label = filename\n"
            "```"
        )

        zip_file = st.file_uploader(
            "Upload ZIP archive", type=["zip"], key="zip_upload",
        )

        db_name = st.text_input(
            "New Database Name (without .pkl)",
            value="block_A_inmates",
            help="The output file will be `<name>.pkl`.",
        )

        if st.button("🚀 Create Database", key="btn_batch", disabled=zip_file is None):
            if not db_name.strip():
                st.error("Please enter a valid database name.")
            else:
                output_pkl = f"{db_name.strip()}.pkl"
                # Save & extract ZIP to a temp folder
                tmp_root = Path(tempfile.mkdtemp(dir=TEMP_DIR))
                try:
                    zip_path = _save_uploaded(zip_file, dest_dir=tmp_root,
                                              name_override="archive.zip")
                    extract_dir = tmp_root / "faces"
                    with zipfile.ZipFile(zip_path, "r") as zf:
                        zf.extractall(extract_dir)

                    # If the ZIP contains a single top-level folder, use it
                    children = list(extract_dir.iterdir())
                    if len(children) == 1 and children[0].is_dir():
                        dataset_dir = children[0]
                    else:
                        dataset_dir = extract_dir

                    with st.spinner("⏳ Running batch encoding — this may take a while…"):
                        cmd = [
                            sys.executable, "encode_faces.py",
                            "-d", str(dataset_dir),
                            "-o", str(PROJECT_ROOT / output_pkl),
                        ]
                        ok, log = _run(cmd)

                    with st.expander("📋 Encoding Log", expanded=not ok):
                        st.code(log, language="text")

                    if ok and (PROJECT_ROOT / output_pkl).exists():
                        st.success(f"✅ Database **{output_pkl}** created successfully!")
                        # Refresh sidebar pkl list on next rerun
                        st.rerun()
                    else:
                        st.error("❌ Batch encoding failed — see log above.")

                finally:
                    shutil.rmtree(tmp_root, ignore_errors=True)

    # ── Mode B: Single append ───────────────────────────────────────────
    else:
        st.markdown(
            "Upload a **single face image**.  The person's identity is "
            "automatically derived from the **filename** (e.g. "
            "`John_Doe.jpg` → *John_Doe*)."
        )

        # Re-scan so the dropdown is up to date
        pkl_files_tab1 = _scan_files("*.pkl")
        if not pkl_files_tab1:
            st.warning("No `.pkl` databases found — create one first.")
        else:
            selected_pkl = st.selectbox(
                "Target Database",
                pkl_files_tab1,
                key="append_pkl_select",
            )

            img_file = st.file_uploader(
                "Upload face image",
                type=["jpg", "jpeg", "png"],
                key="append_img_upload",
            )

            if st.button("➕ Append to Database", key="btn_append",
                         disabled=img_file is None):
                tmp_dir = Path(tempfile.mkdtemp(dir=TEMP_DIR))
                try:
                    # Preserve original filename (identity label)
                    img_path = _save_uploaded(
                        img_file, dest_dir=tmp_dir,
                        name_override=img_file.name,
                    )

                    with st.spinner("⏳ Detecting face and appending…"):
                        cmd = [
                            sys.executable, "encode_faces.py",
                            "--add-image", str(img_path),
                            "-o", str(PROJECT_ROOT / selected_pkl),
                        ]
                        ok, log = _run(cmd)

                    with st.expander("📋 Encoding Log", expanded=not ok):
                        st.code(log, language="text")

                    if ok:
                        person_name = Path(img_file.name).stem
                        st.success(
                            f"✅ **{person_name}** appended to `{selected_pkl}`!"
                        )
                        st.rerun()
                    else:
                        st.error("❌ Single-append failed — see log above.")
                finally:
                    shutil.rmtree(tmp_dir, ignore_errors=True)


# ────────────────────────────────────────────────────────────────────────────
#  Tab 2 — Image Analysis
# ────────────────────────────────────────────────────────────────────────────


with tab_image:
    st.subheader("Static Image Analysis")

    pkl_files_tab2 = _scan_files("*.pkl")
    if not pkl_files_tab2:
        st.warning("No `.pkl` databases found — create one in the **Database Management** tab.")
    else:
        selected_pkl_img = st.selectbox(
            "Encoding Database", pkl_files_tab2, key="img_pkl_select",
        )

        img_upload = st.file_uploader(
            "Upload a CCTV frame / image",
            type=["jpg", "jpeg", "png", "bmp", "tiff", "webp", "pgm"],
            key="img_analysis_upload",
        )

        if st.button("🔍 Analyse Image", key="btn_analyse_img",
                      disabled=img_upload is None):
            tmp_dir = Path(tempfile.mkdtemp(dir=TEMP_DIR))
            try:
                img_path = _save_uploaded(img_upload, dest_dir=tmp_dir,
                                          name_override=img_upload.name)

                stem = Path(img_upload.name).stem
                out_path = OUTPUT_DIR / f"{stem}_identified.jpg"

                cmd = [
                    sys.executable, "process_image.py",
                    "-i", str(img_path),
                    "-e", str(PROJECT_ROOT / selected_pkl_img),
                    "-t", str(tolerance),
                    "-o", str(out_path),
                    "--no-display",
                ]

                with st.spinner("🔍 Detecting and identifying faces…"):
                    ok, log = _run(cmd)

                with st.expander("📋 Processing Log", expanded=not ok):
                    st.code(log, language="text")

                if ok and out_path.exists():
                    st.success("✅ Analysis complete!")

                    annotated_bgr = cv2.imread(str(out_path))
                    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

                    # ── Metrics (parse from log) ────────────────────────
                    import re
                    m_total = re.search(r"Faces detected\s*:\s*(\d+)", log)
                    m_known = re.search(r"Known matches\s*:\s*(\d+)", log)
                    m_unk   = re.search(r"Unknown faces\s*:\s*(\d+)", log)

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Faces",   m_total.group(1) if m_total else "—")
                    col2.metric("Known Matches",  m_known.group(1) if m_known else "—")
                    col3.metric("Unknown Faces",  m_unk.group(1)   if m_unk   else "—")

                    img_col, _ = st.columns([2, 1])
                    with img_col:
                        st.image(
                            annotated_rgb,
                            caption="RED = Known  |  GREEN = Unknown",
                            use_container_width=True,
                        )

                    _, img_bytes = cv2.imencode(".jpg", annotated_bgr)
                    st.download_button(
                        "⬇️ Download Annotated Image",
                        data=img_bytes.tobytes(),
                        file_name=f"{stem}_identified.jpg",
                        mime="image/jpeg",
                    )
                else:
                    st.error("❌ Image analysis failed — see log above.")

            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)


# ────────────────────────────────────────────────────────────────────────────
#  Tab 1 — Video Surveillance
# ────────────────────────────────────────────────────────────────────────────

with tab_video:
    st.subheader("Video Surveillance Processing")

    pkl_files_tab3 = _scan_files("*.pkl")
    if not pkl_files_tab3:
        st.warning("No `.pkl` databases found — create one in the **Database Management** tab.")
    else:
        selected_pkl_vid = st.selectbox(
            "Encoding Database", pkl_files_tab3, key="vid_pkl_select",
        )

        vid_upload = st.file_uploader(
            "Upload CCTV video (.mp4)",
            type=["mp4"],
            key="vid_upload",
        )

        col_skip, col_scale = st.columns(2)
        with col_skip:
            frame_skip = st.slider(
                "Frame Skip",
                min_value=1,
                max_value=30,
                value=3,
                step=1,
                help="Process every Nth frame.  Higher = faster but may miss detections.",
            )
        with col_scale:
            downscale = st.slider(
                "Downscale Factor",
                min_value=0.25,
                max_value=2.00,
                value=1.00,
                step=0.05,
                help=(
                    "Scale factor applied to each frame before detection.  "
                    "Values < 1 speed up processing; values > 1 upscale for "
                    "better detection on low-res footage."
                ),
            )

        if st.button("🎬 Process Video", key="btn_process_vid",
                      disabled=vid_upload is None):
            tmp_dir = Path(tempfile.mkdtemp(dir=TEMP_DIR))
            try:
                vid_path = _save_uploaded(vid_upload, dest_dir=tmp_dir,
                                          name_override=vid_upload.name)

                stem = Path(vid_upload.name).stem
                log_csv = OUTPUT_DIR / f"{stem}_detections.csv"

                cmd = [
                    sys.executable, "main_tracker.py",
                    "-v", str(vid_path),
                    "-e", str(PROJECT_ROOT / selected_pkl_vid),
                    "-t", str(tolerance),
                    "-s", str(frame_skip),
                    "--scale", str(downscale),
                    "--save",
                    "--log", str(log_csv),
                ]
                if enable_sr:
                    cmd.append("--enhance")

                with st.spinner("🎬 Processing video — this may take several minutes…"):
                    ok, log = _run(cmd, timeout=1800)  # 30-min timeout

                with st.expander("📋 Processing Log", expanded=not ok):
                    st.code(log, language="text")

                out_video = _get_latest_output("tracked_*.mp4")

                if ok and out_video and out_video.exists():
                    st.success("✅ Video processing complete!")

                    st.subheader("🎬 Output Preview")
                    st.video(str(out_video))

                    # ── Downloads ───────────────────────────────────────
                    dl_col1, dl_col2 = st.columns(2)

                    with dl_col1:
                        with open(out_video, "rb") as vf:
                            st.download_button(
                                "⬇️ Download Annotated Video",
                                data=vf.read(),
                                file_name=f"{stem}_tracked.mp4",
                                mime="video/mp4",
                            )

                    with dl_col2:
                        if log_csv.exists():
                            with open(log_csv, "rb") as cf:
                                st.download_button(
                                    "⬇️ Download CSV Log",
                                    data=cf.read(),
                                    file_name=f"{stem}_detections.csv",
                                    mime="text/csv",
                                )
                        else:
                            st.info("No CSV log was generated.")
                else:
                    st.error("❌ Video processing failed — see log above.")

            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)


# ════════════════════════════════════════════════════════════════════════════
#  Footer
# ════════════════════════════════════════════════════════════════════════════

st.divider()
st.caption(
    "Powered by MTCNN / dlib / face_recognition / Streamlit"
)
