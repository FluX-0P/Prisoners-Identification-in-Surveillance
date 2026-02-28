# Prisoners' Identification in a Surveillance Environment
## Research Paper Outline

**Lead Author:** Shreyash Dehury

---

## Abstract
> A concise 200–250 word summary covering:
> - Problem statement (re-identification of known individuals in CCTV footage)
> - Proposed methodology (HOG/CNN face detection + 128-d dlib encoding + Euclidean distance matching)
> - Key contribution (cross-resolution matching pipeline with FAR-aware tolerance tuning)
> - Principal results (detection accuracy, FAR at chosen threshold, processing speed)

---

## 1. Introduction

### 1.1 Background & Motivation
- Growing need for automated surveillance in correctional facilities
- Manual monitoring is error-prone and does not scale
- Face recognition has matured, but **cross-resolution matching** (high-res mugshots vs. low-res CCTV) remains an open challenge

### 1.2 Problem Statement
- Given a database of prisoner mugshots and a live/recorded CCTV feed, automatically identify and annotate any known prisoner appearing in the footage in near real-time.

### 1.3 Objectives
1. Build a lightweight, deployable pipeline using open-source tools (Python, OpenCV, dlib/face_recognition)
2. Quantify system performance under varying resolution gaps
3. Analyse the tolerance–FAR trade-off and provide tuning guidelines for prison administrators

### 1.4 Scope & Limitations
- Single-camera, non-occluded frontal/near-frontal faces
- No re-identification across disjoint camera networks (future work)

---

## 2. Literature Review

### 2.1 Classical Face Recognition
- Eigenfaces (Turk & Pentland, 1991)
- Fisherfaces / LDA-based methods
- Local Binary Patterns (LBP)

### 2.2 Deep Learning Approaches
- DeepFace (Taigman et al., 2014)
- FaceNet (Schroff et al., 2015) — triplet loss & 128-d embeddings
- ArcFace (Deng et al., 2019) — additive angular margin loss

### 2.3 dlib's ResNet Encoder
- Architecture: ResNet-34 variant trained on ~3M face images
- Produces 128-dimensional embedding per face
- Achieves 99.38% accuracy on LFW benchmark
- Used in the `face_recognition` Python library (Geitgey, 2017)

### 2.4 Cross-Resolution Face Recognition
- Resolution mismatch degrades embedding quality
- Super-resolution pre-processing (SRGAN, ESRGAN) as mitigation
- Multi-scale training strategies
- Relevance to mugshot-vs-CCTV scenario

### 2.5 Surveillance-Specific Challenges
- Motion blur, low illumination, oblique angles
- Occlusion (masks, hats)
- Real-time throughput constraints on edge hardware

---

## 3. Methodology

### 3.1 System Architecture
```
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│  Mugshot DB  │────▶│ encode_faces │────▶│ encodings.pkl    │
│ (known_faces)│     │   .py        │     │ (128-d vectors)  │
└──────────────┘     └──────────────┘     └────────┬─────────┘
                                                   │
┌──────────────┐     ┌──────────────┐              │
│  CCTV Video  │────▶│ main_tracker │◀─────────────┘
│  (.mp4)      │     │   .py        │
└──────────────┘     └──────┬───────┘
                            │
                     ┌──────▼───────┐
                     │  Annotated   │
                     │  Video + CSV │
                     └──────────────┘
```

### 3.2 Enrolment Phase (encode_faces.py)
1. **Image ingestion**: Walk `known_faces/` directory; each sub-folder = one identity
2. **Face detection**: HOG (CPU) or CNN (GPU) detector via dlib
3. **Encoding extraction**: 128-d embedding per face using dlib's ResNet-34 variant
4. **Multi-image enrolment**: Multiple images per person (front, left, right profile) to capture intra-class variation
5. **Serialisation**: Save `{encodings, names}` dictionary to `.pkl` file

### 3.3 Identification Phase (main_tracker.py)
1. **Frame acquisition**: Read video frame-by-frame via OpenCV
2. **Frame skipping**: Process every N-th frame to maintain throughput; reuse cached detections on skipped frames
3. **Optional up-scaling**: Resize low-res frames before detection to recover small faces
4. **Face detection**: HOG/CNN on the (possibly scaled) frame
5. **Encoding computation**: Extract 128-d vector for each detected face
6. **Distance-based matching**:
   - Compute Euclidean distance from each query encoding to every enrolment encoding
   - Apply tolerance threshold τ
   - Majority-vote (weighted by inverse distance) among all matches below τ
7. **Annotation**: Draw RED box (match) or GREEN box (unknown) with label and confidence score

### 3.4 Cross-Resolution Handling
| Strategy | Description |
|----------|-------------|
| **Multi-scale enrolment** | Include both high-res and synthetically degraded images per person |
| **Frame up-scaling** | Resize low-res CCTV frames before face detection (configurable `--scale`) |
| **Encoding normalisation** | dlib's internal pipeline resizes faces to 150×150 before encoding, providing partial resolution invariance |
| **Tolerance calibration** | Lower τ compensates for noisier embeddings from low-res inputs |

### 3.5 Tolerance Tuning & FAR Analysis
- **False Acceptance Rate (FAR)**: Fraction of unknown faces incorrectly matched to a known identity
- **False Rejection Rate (FRR)**: Fraction of known faces that go unrecognised
- **Procedure**:
  1. Prepare a test video containing both known and unknown individuals (ground truth annotated)
  2. Sweep tolerance τ from 0.35 to 0.65 in steps of 0.02
  3. For each τ, record per-frame detections via `--log` CSV output
  4. Compute FAR and FRR from the CSV against ground truth
  5. Plot FAR vs. FRR (DET curve) and select the operational threshold

---

## 4. Experimental Setup

### 4.1 Hardware
- CPU: Intel Core i5/i7 or AMD Ryzen 5 (minimum)
- GPU: NVIDIA GTX 1050+ (optional, for CNN model)
- RAM: 8 GB minimum

### 4.2 Software Environment
- Python 3.10+
- Key libraries: OpenCV 4.8+, face_recognition 1.3, dlib 19.24, NumPy 1.24+, imutils 0.5

### 4.3 Dataset
- **Enrolment set**: N mugshot images across M identities (specify counts)
- **Test video(s)**: Recorded indoor CCTV footage at 480p/720p, X minutes duration
- **Ground truth**: Frame-level annotations of who appears and when

### 4.4 Evaluation Metrics
| Metric | Definition |
|--------|-----------|
| **Accuracy** | (TP + TN) / Total |
| **Precision** | TP / (TP + FP) |
| **Recall** | TP / (TP + FN) |
| **F1 Score** | 2 × (Precision × Recall) / (Precision + Recall) |
| **FAR** | FP / (FP + TN) |
| **FRR** | FN / (FN + TP) |
| **Throughput** | Frames processed per second |

---

## 5. Results & Discussion

### 5.1 Detection Accuracy
- Table: Accuracy, Precision, Recall, F1 at default tolerance (τ = 0.50)
- Confusion matrix (known vs. unknown)

### 5.2 FAR vs. Tolerance Sweep
- Plot: FAR and FRR as functions of τ
- Identify Equal Error Rate (EER) — the τ where FAR ≈ FRR
- Recommended operating point for prison surveillance

### 5.3 Cross-Resolution Impact
- Compare performance at original CCTV resolution vs. 1.5× and 2.0× up-scaling
- Table: Accuracy degradation when enrolment images are artificially down-sampled

### 5.4 Processing Speed
- Table: FPS on CPU (HOG) vs. GPU (CNN) at various frame-skip values
- Trade-off analysis: accuracy vs. throughput

### 5.5 Failure Cases
- Profile / extreme-angle faces
- Heavy occlusion
- Very low illumination
- Motion blur in fast-moving subjects

---

## 6. Conclusion
- Summarise key findings
- Restate the contribution: a practical, lightweight pipeline for prisoner identification in CCTV footage with quantified FAR characteristics
- Emphasise the importance of tolerance calibration per deployment site

---

## 7. Future Work
1. **Real-time live-stream processing** with multi-threaded frame capture
2. **Re-identification across multiple cameras** using appearance descriptors
3. **Anti-spoofing** (liveness detection) to prevent photo-based attacks
4. **Super-resolution pre-processing** (ESRGAN) for very low-res feeds
5. **Edge deployment** on Jetson Nano / Raspberry Pi with TensorRT optimisation
6. **Temporal smoothing** — track faces across frames to reduce flickering labels

---

## 8. References
> Use IEEE or APA format.  Key references to include:

1. Turk, M., & Pentland, A. (1991). Eigenfaces for recognition. *Journal of Cognitive Neuroscience*.
2. Schroff, F., Kalenichenko, D., & Philbin, J. (2015). FaceNet: A unified embedding for face recognition and clustering. *CVPR*.
3. Deng, J., Guo, J., Xue, N., & Zafeiriou, S. (2019). ArcFace: Additive angular margin loss for deep face recognition. *CVPR*.
4. King, D. E. (2009). Dlib-ml: A machine learning toolkit. *JMLR*.
5. Geitgey, A. (2017). face_recognition: The world's simplest facial recognition API. GitHub.
6. Taigman, Y., Yang, M., Ranzato, M., & Wolf, L. (2014). DeepFace: Closing the gap to human-level performance in face verification. *CVPR*.
7. Viola, P., & Jones, M. (2001). Rapid object detection using a boosted cascade of simple features. *CVPR*.
8. Ledig, C. et al. (2017). Photo-realistic single image super-resolution using a generative adversarial network (SRGAN). *CVPR*.

---

## Appendix A — How to Reproduce

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place mugshot images in known_faces/<PersonName>/
#    e.g., known_faces/John_Doe/front.jpg

# 3. Generate encodings
python encode_faces.py --dataset known_faces --output encodings.pkl

# 4. Run tracker on a test video
python main_tracker.py -v input_videos/test.mp4 --tolerance 0.50 --skip 2 --save --log output/detections.csv

# 5. Analyse FAR
#    Parse output/detections.csv against ground-truth annotations
```

## Appendix B — Tolerance Tuning Quick Reference

| Tolerance (τ) | Behaviour | Recommended Use Case |
|:-:|---|---|
| 0.40 | Very strict — low FAR, high FRR | Maximum-security environments |
| 0.45 | Strict | High-security with good quality cameras |
| **0.50** | **Balanced (default)** | **General prison surveillance** |
| 0.55 | Lenient — higher recall, higher FAR | When missing a match is more costly than a false alarm |
| 0.60 | Very lenient | Low-quality cameras; manual review expected |
