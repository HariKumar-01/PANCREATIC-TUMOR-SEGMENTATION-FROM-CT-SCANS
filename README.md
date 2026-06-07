<div align="center">

# 🔬 PancrAI
### Pancreatic Tumor Segmentation from CT Scans

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![MONAI](https://img.shields.io/badge/MONAI-1.3-00ADEF?style=for-the-badge)
![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=for-the-badge&logo=flask&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-F7DF1E?style=for-the-badge)

**Final Year B.Tech Project | IV/IV — AI & ML | Batch A10**

*SAGI RAMA KRISHNAM RAJU ENGINEERING COLLEGE (Autonomous), Bhimavaram*

*Guide: CH. Vinod Varma, Assistant Professor, Dept. of CSE*

</div>

---

## 🧬 The Problem

Pancreatic cancer is one of the deadliest malignancies, with:

| Statistic | Value |
|-----------|-------|
| 5-Year Survival Rate | **< 12%** |
| Cases Diagnosed at Stage III–IV | **~80%** |
| Dataset CT Scans Used | **281** annotated 3D volumes |

Tumors appear nearly iso-dense within complex abdominal anatomy, making manual 3D CT segmentation **slow, error-prone, and clinically infeasible** at scale.

---

## 💡 Our Solution — PancrAI

**PancrAI** is a fully automated, end-to-end Clinical Decision Support (CAD) system that:

- 🧠 **Segments** pancreatic tumors in full 3D CT volumes using a Swin-UNETR transformer
- 📐 **Measures** tumor volume (mL), RECIST longest-axis diameter, and anatomical region (Head / Body / Tail)
- 🌐 **Deploys** a Flask web application with multi-planar CT viewer and heatmap overlays
- 📄 **Generates** automated PDF radiology reports — ready for clinical review

---

## 🏆 Results vs Published Baselines

Benchmarked on the **Medical Segmentation Decathlon — Task07 Pancreas** dataset (MICCAI 2018):

| Model / Method | Pancreas Dice | Tumor Dice | Mean Dice |
|----------------|:------------:|:----------:|:---------:|
| 3D U-Net (Çiçek, 2016) | 0.612 | — | — |
| Attention U-Net (Oktay, 2018) | 0.641 | — | — |
| UNETR (Hatamizadeh, 2022) | 0.680 | 0.361 | 0.521 |
| Swin UNETR (Tang, 2022) | 0.700 | 0.380 | 0.540 |
| **PancrAI — Ours** | **0.732** ✅ | **0.422** ✅ | **0.577** ✅ |

> **PancrAI exceeds all published baselines** — +0.017 Pancreas Dice and +0.026 Tumor Dice over the Swin UNETR baseline, achieved through 8-flip TTA and 4× tumor class weighting in DiceCE loss.

---

## 🏗️ System Architecture

```
Input CT Scan (.nii.gz)
        │
        ▼
┌──────────────────────────────────────┐
│  Preprocessing Pipeline              │
│  • HU Windowing: −175 to +250        │
│  • Voxel Resampling: 1.5×1.5×2.0 mm │
│  • 10× MONAI Augmentations           │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  Swin-UNETR (62.2M Parameters)       │
│  • SSL Pretrained Swin Encoder       │
│  • Hierarchical Shifted Window Attn  │
│  • UNETR-style Skip Connections      │
│  • 8-flip Test-Time Augmentation     │
└──────────────────┬───────────────────┘
                   │
                   ▼
    3-Class Voxel Mask (BG / Pancreas / Tumor)
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
 Quantitative Metrics    Flask CAD App
 • Volume (mL)           • Multi-planar Viewer
 • RECIST Diameter        • Heatmap Overlay
 • Tumor Region           • PDF Report Generator
 • Tumor Burden %         • REST API /predict
```

---

## 📂 Repository Structure

```
PancrAI/
├── 📄 app.py                   Flask web application (REST API + CT viewer UI)
├── 🤖 infer.py                 Inference pipeline with 8-flip TTA
├── ⚙️  config.json              Model configuration & hyperparameters
├── 🧠 best_model.pth           Trained SwinUNETR checkpoint (epoch 115)
├── 📓 PancrAI_Colab_Ser.ipynb  Full training notebook (Google Colab)
├── 📁 templates/               Flask Jinja2 HTML templates
│   ├── index.html              Upload interface
│   └── result.html             Multi-planar results viewer
└── 📖 README.md
```

---

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.9+  |  CUDA 11.8+ (optional, CPU works)  |  ~4 GB RAM minimum
```

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/PancrAI.git
cd PancrAI
pip install torch torchvision monai flask nibabel numpy scipy
```

### 2. Run the Web App

```bash
python app.py
# Open your browser at http://localhost:5000
```

### 3. Upload & Analyze

Upload any `.nii.gz` CT volume file. PancrAI will:
1. Preprocess the CT volume (HU windowing + resampling)
2. Run 3D segmentation with 8-flip TTA
3. Display axial / sagittal / coronal views with AI overlay
4. Report tumor volume, RECIST diameter, and anatomical region
5. Generate a downloadable PDF radiology report

### 4. REST API

```python
import requests

with open("ct_scan.nii.gz", "rb") as f:
    response = requests.post(
        "http://localhost:5000/predict",
        files={"file": f}
    )

result = response.json()
print(result["tumor_volume_ml"])
print(result["pancreas_dice"])
print(result["anatomical_region"])
```

---

## 🔬 Model Details

| Component | Specification |
|-----------|---------------|
| **Architecture** | Swin-UNETR (Swin Transformer encoder + UNETR decoder) |
| **Parameters** | 62.2 million |
| **Input** | 3D CT volume — NIfTI (.nii.gz) |
| **Output** | 3-class voxel mask: Background / Pancreas / Tumor |
| **Loss Function** | DiceCE with class weights [0.1, 1.0, 4.0] |
| **Epochs Trained** | 127 (optimal checkpoint: epoch 115) |
| **Augmentations** | 10× MONAI spatial + intensity transforms |
| **TTA Strategy** | 8-flip Test-Time Augmentation |
| **Encoder Pretraining** | Self-Supervised Learning (SSL) on medical images |
| **Training Loss Reduction** | 74% reduction across epochs |

---

## 📊 Dataset — MSD Task07 Pancreas

| Property | Value |
|----------|-------|
| **Source** | Medical Segmentation Decathlon — MICCAI 2018 |
| **Total Annotated Scans** | 281 3D CT volumes |
| **Format** | NIfTI (.nii.gz) |
| **Label Classes** | 0: Background / 1: Pancreas / 2: Tumor |
| **Training Split** | 238 scans (85%) |
| **Validation Split** | 43 scans (15%) |
| **HU Windowing** | −175 to +250 (soft tissue window) |
| **Voxel Resampling** | 1.5 × 1.5 × 2.0 mm isotropic |
| **Class Imbalance** | Tumor: ~0.2% of all voxels |

---

## 🌐 Flask CAD Application

| Feature | Description |
|---------|-------------|
| **Multi-planar Viewer** | Axial, Sagittal, and Coronal slices with AI overlay |
| **Tumor Volume** | Automatic 3D volumetric computation in mL |
| **RECIST Diameter** | Longest axis measurement for clinical staging |
| **Anatomical Region** | Automatic Head / Body / Tail classification |
| **Heatmap Overlay** | Prediction probability maps on CT slices |
| **PDF Report** | Auto-generated structured radiology report |
| **REST API** | POST `/predict` endpoint for integration |

---

## 📈 Training Curves

```
Dice Score Progress (validated every 5 epochs):

Pancreas Dice:  0.32 ──────────────────────────────► 0.732 (Epoch 115)
Tumor Dice:     0.15 ─ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ► 0.422 (Epoch 115)
Mean Dice:      0.23 ──────────────────────────────► 0.577 (Epoch 115)
                                                        ▲
                                             Clinical threshold: 0.5
```

---

## 🔮 Future Scope

| Priority | Feature | Impact |
|----------|---------|--------|
| 🔴 High | **Real-Time GPU Inference** — AWS/GCP deployment | Reduce ~5 min CPU inference to <1 min |
| 🔴 High | **DICOM Integration** — direct hospital PACS/RIS support | Eliminate NIfTI conversion step |
| 🟡 Med | **Multi-Organ Segmentation** — all 13 abdominal organs | Comprehensive abdominal CAD |
| 🟡 Med | **Cancer Staging** — texture features → stage prediction | Automate TNM staging |
| 🟢 Low | **Longitudinal Monitoring** — multi-timepoint comparison | Track treatment response |
| 🟢 Low | **Federated Learning** — multi-hospital privacy training | Better generalization |

---

## ⚠️ Limitations

- **Dataset size:** MSD Task07 has 281 scans — results may not generalize to all scanner types or contrast phases
- **Tumor Dice variability:** Small tumor targets cause validation Dice fluctuation across epochs
- **CPU inference speed:** Without GPU, inference takes ~5–10 minutes per scan
- **NIfTI-only input:** DICOM conversion is required before using with hospital PACS systems

---

## 📚 References

1. A. Hatamizadeh et al. *Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors.* MICCAI BrainLes Workshop, 2022.
2. Y. Tang et al. *Self-Supervised Pre-Training of Swin Transformers for 3D Medical Image Analysis.* IEEE/CVF CVPR, 2022.
3. O. Oktay et al. *Attention U-Net: Learning Where to Look for the Pancreas.* MICCAI, 2018.
4. O. Çiçek et al. *3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation.* MICCAI, 2016.
5. A. Antonelli et al. *The Medical Segmentation Decathlon.* Nature Communications, 2022.
6. M. J. Cardoso et al. *MONAI: An open-source framework for deep learning in healthcare.* arXiv:2211.02701, 2022.
7. Z. Liu et al. *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.* ICCV, 2021.

---

## 👥 Team — Batch A10

| Name | Roll Number | Role |
|------|-------------|------|
| CH. Hari Kumar | 22B91A6141 | Model Training & TTA Pipeline |
| J.D.S Karthikeya | 22B91A6161 | Flask CAD Application |
| Badugu Ajay | 22B91A6118 | Preprocessing & Augmentation |
| B. Hema Sree | 22B91A6134 | Evaluation & Report Generation |

**Project Guide:** CH. Vinod Varma, Assistant Professor, Dept. of CSE, SRKR Engineering College

---

<div align="center">

Made with ❤️ at **SAGI RAMA KRISHNAM RAJU ENGINEERING COLLEGE, Bhimavaram**

Academic Year 2024–25 | B.Tech Final Year Project

</div>
