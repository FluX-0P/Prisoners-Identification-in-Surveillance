# Criminal Identification Using Face Recognition

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

A face-recognition pipeline that identifies known individuals in CCTV footage
or static images. The system matches detected faces against a pre-built
database of 128-dimensional encodings and annotates results with colour-coded
bounding boxes:

- 🔴 **RED box + name** — verified match (known individual)
- 🟢 **GREEN box + "Unknown"** — unregistered face

Ships with a **Streamlit web UI** for drag-and-drop image/video processing
and **CLI tools** for batch processing and automation.

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
  - [Option A — pip (all platforms)](#option-a--pip-all-platforms)
  - [Option B — Conda (recommended if dlib fails with pip)](#option-b--conda-recommended-if-dlib-fails-with-pip)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Web UI (Streamlit)](#web-ui-streamlit)
  - [CLI — Encode Faces](#cli--encode-faces)
  - [CLI — Video Tracker](#cli--video-tracker)
  - [CLI — Static Image](#cli--static-image)
- [Tolerance Tuning Guide](#tolerance-tuning-guide)
- [Tech Stack](#tech-stack)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **MTCNN face detection** — robust to pose variation and partial occlusion
- **Cross-resolution matching** — high-res mugshots vs. low-res CCTV frames
- **ESPCN ×4 Super-Resolution** — optional upscaling for tiny faces (< 60 px)
- **Inverse-distance weighted voting** — robust identification with confidence scores
- **Dual-mode encoding** — batch creation from a folder *or* single-image append
- **Web UI** — upload images/videos via a multi-tab Streamlit dashboard
- **CLI tools** — encode faces, process videos, analyse static images
- **CSV logging** — per-frame detection logs for FAR/FRR analysis
- **Auto GPU detection** — uses CNN model when CUDA is available, falls back to HOG
- **Headless mode** — works in SSH/Colab environments without a display

---

## Project Structure

```
├── app.py                 # Streamlit web interface (multi-tab)
├── encode_faces.py        # Build / update face-encoding database (.pkl)
├── main_tracker.py        # Identify faces in surveillance video
├── process_image.py       # Identify faces in a static image
├── ESPCN_x4.pb            # Super-Resolution model weights
├── requirements.txt       # Python dependencies
├── known_faces/           # Mugshot images (one sub-folder per person)
│   ├── PersonA/
│   │   ├── front.jpg
│   │   └── profile.jpg
│   └── PersonB/
│       └── mugshot.png
├── dataset/               # ORL/AT&T face dataset for experimentation
│   ├── s1/ … s40/
├── input_videos/          # CCTV footage to process
├── output/                # Annotated results & CSV logs
└── unknown_faces/         # Unrecognised face crops (runtime)
```

---

## Installation

### Prerequisites

- **Python 3.10+** (3.11 recommended)
- **CMake** — required to build dlib (pip path only)
- **C++ compiler** — Visual Studio Build Tools (Windows), Xcode CLI (macOS), or `build-essential` (Linux)

### Option A — pip (all platforms)

```bash
# 1. Clone the repo
git clone https://github.com/FluX-0P/Prisoners-Identification-in-Surveillance.git
cd Prisoners-Identification-in-Surveillance

# 2. Create a virtual environment
python -m venv .venv

# Activate it:
#   Windows  → .venv\Scripts\activate
#   macOS/Linux → source .venv/bin/activate

# 3. Install dependencies (dlib compiles from source — needs CMake + C++ toolchain)
pip install -r requirements.txt
```

> **Windows note:** Install
> [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
> (select "Desktop development with C++") before running pip.

### Option B — Conda (recommended if dlib fails with pip)

If `pip install dlib` fails due to CMake or compiler issues, **conda** can
install a pre-built dlib binary — no C++ toolchain required.

```bash
# 1. Install Miniconda / Anaconda if you haven't already
#    https://docs.conda.io/en/latest/miniconda.html

# 2. Create a new conda environment
conda create -n face-id python=3.11 -y
conda activate face-id

# 3. Install dlib from conda-forge (pre-built — no CMake needed)
conda install -c conda-forge dlib -y

# 4. Clone the repo
git clone https://github.com/FluX-0P/Prisoners-Identification-in-Surveillance.git
cd Prisoners-Identification-in-Surveillance

# 5. Install the remaining Python packages via pip
pip install -r requirements.txt
```

> **Why this works:** conda's `dlib` package is pre-compiled, so you skip
> the CMake/C++ build entirely. The rest of the dependencies (`face_recognition`,
> `opencv-python`, `mtcnn`, `streamlit`, etc.) install normally via pip.

> **GPU users:** If you have an NVIDIA GPU and want CUDA-accelerated dlib:
> ```bash
> conda install -c conda-forge dlib cudatoolkit -y
> ```

---

## Quick Start

```bash
# 1. Activate your environment (venv or conda)

# 2. Add mugshot images to known_faces/ (one sub-folder per person)

# 3. Generate face encodings
python encode_faces.py

# 4. Launch the web UI
streamlit run app.py
```

---

## Usage

### Web UI (Streamlit)

```bash
streamlit run app.py
```

The app has three tabs:

| Tab | Purpose |
|-----|---------|
| **Video Surveillance** | Upload a `.mp4`, configure frame-skip & scale, download annotated video + CSV |
| **Image Analysis** | Upload a CCTV frame, view annotated results with per-face details |
| **Database Management** | Create a new `.pkl` from a ZIP *or* append a single face image |

Global settings (tolerance, super-resolution toggle) are in the sidebar.

### CLI — Encode Faces

```bash
# Batch creation (default)
python encode_faces.py -d known_faces -o encodings.pkl

# Single append
python encode_faces.py --add-image mugshots/John_Doe.jpg -o encodings.pkl
```

| Flag | Default | Description |
|------|---------|-------------|
| `-d, --dataset` | `known_faces` | Path to mugshot directory |
| `--add-image` | — | Path to a single image (append mode) |
| `-o, --output` | `encodings.pkl` | Output pickle file |
| `-m, --model` | `mtcnn` | Detection model (`mtcnn`, `hog`, `cnn`) |
| `-j, --jitters` | `1` | Re-sampling count (higher = more accurate, slower) |
| `--upscale` | `0` (auto) | Upscale factor for small images |
| `--min-confidence` | `0.85` | Minimum MTCNN detection confidence |

### CLI — Video Tracker

```bash
python main_tracker.py -v input_videos/clip.mp4
python main_tracker.py -v input_videos/clip.mp4 -t 0.45 --skip 5 --save --enhance
```

| Flag | Default | Description |
|------|---------|-------------|
| `-v, --video` | *(required)* | Input video path |
| `-e, --encodings` | `encodings.pkl` | Encoding file |
| `-t, --tolerance` | `0.50` | Match threshold (0.0–1.0), lower = stricter |
| `-m, --model` | `mtcnn` | Detection model (`mtcnn`, `hog`, `cnn`, `auto`) |
| `-s, --skip` | `3` | Process every Nth frame |
| `--scale` | `1.0` | Frame resize factor before detection |
| `--save` | off | Save annotated video to `output/` |
| `--log` | off | Write per-frame CSV detections |
| `--enhance` | off | Enable ESPCN ×4 super-resolution for tiny faces |
| `--min-confidence` | `0.85` | Minimum MTCNN detection confidence |

Press **q** to quit during playback.

### CLI — Static Image

```bash
python process_image.py -i photo.jpg
python process_image.py -i photo.jpg -t 0.45 -o output/result.jpg
```

| Flag | Default | Description |
|------|---------|-------------|
| `-i, --image` | *(required)* | Input image path |
| `-t, --tolerance` | `0.50` | Match threshold |
| `-m, --model` | `mtcnn` | Detection model |
| `-o, --output` | auto | Output path for annotated image |
| `--no-display` | off | Skip GUI window (headless mode) |
| `--min-confidence` | `0.85` | Minimum MTCNN detection confidence |

---

## Tolerance Tuning Guide

| Tolerance | Strictness | Best For |
|:---------:|------------|----------|
| 0.40 | Very strict | Maximum-security, low FAR required |
| 0.45 | Strict | Good cameras, controlled lighting |
| **0.50** | **Balanced** | **General surveillance (default)** |
| 0.55 | Lenient | Low-quality cameras, varied angles |
| 0.60 | Very lenient | When missing matches is costly |

Generate a FAR analysis by sweeping tolerance values:

```bash
for tol in 0.40 0.45 0.50 0.55 0.60; do
  python main_tracker.py -v input_videos/test.mp4 -t $tol --log output/log_$tol.csv --save
done
```

---

## Tech Stack

| Component | Role |
|-----------|------|
| **Python 3.10+** | Core language |
| **OpenCV** | Video I/O, image processing, super-resolution (ESPCN) |
| **MTCNN** | Primary face detector (multi-task CNN) |
| **face_recognition** | 128-d face encoding (wraps dlib's ResNet-34) |
| **dlib** | HOG/CNN face detector, deep metric learning encoder |
| **NumPy** | Numerical operations |
| **Streamlit** | Web dashboard |
| **tqdm** | Progress bars for long-running jobs |

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m "Add my feature"`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request

---

## License

This project is licensed under the [MIT License](LICENSE).
