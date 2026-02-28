# Prisoners' Identification in a Surveillance Environment

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

A face-recognition pipeline that identifies known prisoners in CCTV footage
or static images. The system matches detected faces against a pre-built
database of 128-dimensional encodings and annotates results with colour-coded
bounding boxes:

- 🔴 **RED box + name** — verified match (known prisoner)
- 🟢 **GREEN box + "Unknown"** — unregistered face

It ships with a **Streamlit web UI** for drag-and-drop image/video processing
and also supports **CLI** usage for batch processing and automation.

> **Academic project** — B.Tech Final Year Project by Shreyash Dehury.

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
  - [Windows](#windows)
  - [macOS](#macos)
  - [Linux](#linux)
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

- **Cross-resolution matching** — high-res mugshots vs. low-res CCTV frames
- **Inverse-distance weighted voting** — robust identification with confidence scores
- **Web UI** — upload images/videos via a Streamlit dashboard
- **CLI tools** — encode faces, process videos, analyse static images
- **CSV logging** — per-frame detection logs for FAR/FRR analysis
- **Auto GPU detection** — uses CNN model when CUDA is available, falls back to HOG
- **Headless mode** — works in SSH/Colab environments without a display

---

## Project Structure

```
├── app.py                 # Streamlit web interface
├── encode_faces.py        # Step 1 — Build face-encoding database
├── main_tracker.py        # Step 2 — Identify faces in video
├── process_image.py       # Step 2 (alt) — Identify faces in a static image
├── requirements.txt       # Python dependencies
├── paper_outline.md       # Research paper structure
├── known_faces/           # Mugshot images (one sub-folder per person)
│   ├── PersonA/
│   │   ├── front.jpg
│   │   └── profile.jpg
│   └── PersonB/
│       └── mugshot.png
├── dataset/               # ORL/AT&T face dataset for experimentation
│   ├── s1/ … s40/
├── input_videos/          # CCTV footage to process (not tracked by Git)
├── output/                # Annotated results & CSV logs (not tracked by Git)
└── unknown_faces/         # Unrecognised face crops (runtime)
```

---

## Installation

### Prerequisites (all platforms)

- **Python 3.10+** (3.11 recommended)
- **CMake** — required to build dlib
- **C++ compiler** — platform-specific (see below)

### Windows

```bash
# 1. Install Visual Studio Build Tools (select "Desktop development with C++")
#    https://visualstudio.microsoft.com/visual-cpp-build-tools/

# 2. Clone the repo
git clone https://github.com/<your-username>/criminal-identification-using-fr.git
cd criminal-identification-using-fr

# 3. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt
```

### macOS

```bash
# 1. Install Xcode CLI tools and CMake
xcode-select --install
brew install cmake

# 2. Clone the repo
git clone https://github.com/<your-username>/criminal-identification-using-fr.git
cd criminal-identification-using-fr

# 3. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 4. Install dependencies (dlib compiles from source — takes a few minutes)
pip install -r requirements.txt
```

### Linux (Ubuntu / Debian)

```bash
# 1. Install build tools
sudo apt update
sudo apt install -y build-essential cmake python3-dev python3-venv

# 2. Clone the repo
git clone https://github.com/FluX-0P/Prisoners-Identification-in-Surveillance.git
cd Prisoners-Identification-in-Surveillance

# 3. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

> **Tip:** On any platform, if `pip install dlib` fails, ensure CMake is on
> your `PATH` and that you have a working C++ toolchain.

---

## Quick Start

```bash
# 1. Activate virtual environment (if not already active)
#    Windows:  .venv\Scripts\activate
#    macOS/Linux:  source .venv/bin/activate

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

Upload an image or video through the browser. The sidebar lets you adjust
the recognition tolerance in real time.

### CLI — Encode Faces

```bash
python encode_faces.py --dataset known_faces --output encodings.pkl
```

| Flag | Default | Description |
|------|---------|-------------|
| `-d, --dataset` | `known_faces` | Path to mugshot directory |
| `-o, --output` | `encodings.pkl` | Output pickle file |
| `-m, --model` | `hog` | Detection model (`hog` for CPU, `cnn` for GPU) |
| `-j, --jitters` | `1` | Re-sampling count (higher = more accurate, slower) |
| `--upscale` | `0` (auto) | Upscale factor for small images (e.g. `2.0` for tiny datasets) |

### CLI — Video Tracker

```bash
python main_tracker.py -v input_videos/clip.mp4
```

| Flag | Default | Description |
|------|---------|-------------|
| `-v, --video` | *(required)* | Input video path |
| `-e, --encodings` | `encodings.pkl` | Encoding file |
| `-t, --tolerance` | `0.50` | Match threshold (0.0–1.0), lower = stricter |
| `-m, --model` | `auto` | `hog` (CPU), `cnn` (GPU), or `auto` |
| `-s, --skip` | `3` | Process every Nth frame |
| `--scale` | `0.5` | Frame resize factor before detection |
| `--save` | off | Save annotated video to `output/` |
| `--log` | off | Write per-frame CSV detections |

Press **q** to quit during playback.

### CLI — Static Image

```bash
python process_image.py -i photo.jpg
```

| Flag | Default | Description |
|------|---------|-------------|
| `-i, --image` | *(required)* | Input image path |
| `-t, --tolerance` | `0.50` | Match threshold |
| `-m, --model` | `auto` | Detection model |
| `-o, --output` | auto | Output path for annotated image |
| `--no-display` | off | Skip GUI window (headless mode) |

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
| **OpenCV** | Video I/O, image processing, rendering |
| **face_recognition** | Face detection & 128-d encoding (wraps dlib) |
| **dlib** | HOG/CNN face detector, ResNet-34 encoder |
| **NumPy** | Numerical operations |
| **imutils** | Convenience wrappers for OpenCV |
| **Streamlit** | Web dashboard |
| **tqdm** | Progress bars for long-running jobs |

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m "Add my feature"`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request

Please ensure your code follows the existing style and includes docstrings.

---

## License

This project is licensed under the [MIT License](LICENSE).
