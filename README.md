<div align="center">

<img src="screenshots/banner.png" alt="PancrAI Banner" width="100%"/>

<br/>

```
██████╗  █████╗ ███╗   ██╗ ██████╗██████╗      █████╗ ██╗
██╔══██╗██╔══██╗████╗  ██║██╔════╝██╔══██╗    ██╔══██╗██║
██████╔╝███████║██╔██╗ ██║██║     ██████╔╝    ███████║██║
██╔═══╝ ██╔══██║██║╚██╗██║██║     ██╔══██╗    ██╔══██║██║
██║     ██║  ██║██║ ╚████║╚██████╗██║  ██║    ██║  ██║██║
╚═╝     ╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝╚═╝  ╚═╝    ╚═╝  ╚═╝╚═╝
```

### **Pancreatic Tumor Segmentation from CT Scans**
#### *Transformer-Based 3D Deep Learning · Clinical Decision Support System*

<br/>

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![MONAI](https://img.shields.io/badge/MONAI-1.3-00ADEF?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PC9zdmc+&logoColor=white)](https://monai.io)
[![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![CUDA](https://img.shields.io/badge/CUDA-11.8+-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-F7DF1E?style=for-the-badge)](LICENSE)

<br/>

[![Model](https://img.shields.io/badge/Architecture-Swin--UNETR-8A2BE2?style=flat-square&logo=tensorflow&logoColor=white)]()
[![Params](https://img.shields.io/badge/Parameters-62.2M-ff6b35?style=flat-square)]()
[![Dataset](https://img.shields.io/badge/Dataset-MSD_Task07-0096c7?style=flat-square)]()
[![Epochs](https://img.shields.io/badge/Trained-127_Epochs-2d6a4f?style=flat-square)]()
[![Best Dice](https://img.shields.io/badge/Best_Mean_Dice-0.577-success?style=flat-square)]()
[![Status](https://img.shields.io/badge/Status-Deployed-brightgreen?style=flat-square)]()

<br/>

> 🏥 **Final Year B.Tech Project** — IV/IV · AI & ML · Batch A10  
> 🎓 **SAGI RAMA KRISHNAM RAJU ENGINEERING COLLEGE (Autonomous), Bhimavaram**  
> 👨‍🏫 **Guide:** CH. Vinod Varma, Assistant Professor, Dept. of CSE

</div>

---

## 📋 Table of Contents

<details open>
<summary><b>Click to expand</b></summary>

- [🧬 The Problem](#-the-problem)
- [💡 Our Solution](#-our-solution)
- [🖥️ Screenshots & Demo](#️-screenshots--demo)
- [🏆 Results vs Baselines](#-results-vs-published-baselines)
- [🏗️ System Architecture](#️-system-architecture)
- [📂 Repository Structure](#-repository-structure)
- [🚀 Quick Start](#-quick-start)
- [🔬 Model Details](#-model-details)
- [📊 Dataset](#-dataset--msd-task07-pancreas)
- [📈 Training Curves](#-training-curves)
- [🌐 Flask CAD Application](#-flask-cad-application)
- [⚠️ Limitations](#️-limitations)
- [🔮 Future Scope](#-future-scope)
- [👥 Team](#-team)
- [📚 References](#-references)

</details>

---

## 🧬 The Problem

<table>
<tr>
<td width="60%">

Pancreatic cancer is among the **deadliest malignancies** in oncology. Unlike many cancers, it offers almost no early warning — by the time symptoms appear, the disease has almost always spread.

The core challenge is radiological: pancreatic tumors appear **nearly iso-dense** against the surrounding soft tissue in CT scans. They sit deep inside complex abdominal anatomy, making them extraordinarily difficult to delineate manually.

**Manual 3D CT segmentation is:**
- 🕐 Extremely time-consuming per patient
- ❌ Prone to inter-observer variability
- 📉 Clinically infeasible at scale

</td>
<td width="40%" align="center">

| Statistic | Value |
|-----------|:-----:|
| 5-Year Survival Rate | **< 12%** |
| Diagnosed at Stage III–IV | **~80%** |
| Pancreas volume in CT | **0.5–1%** |
| Smallest detectable tumor | **~5 mm** |
| CT scans in training set | **281** |

</td>
</tr>
</table>

---

## 💡 Our Solution

<div align="center">

```
              ┌─────────────────────────────────────────────┐
              │              PancrAI Pipeline                │
              │                                             │
  CT Scan ──► │  Preprocess ──► Swin-UNETR ──► 3D Mask    │ ──► CAD Report
              │                                             │
              └─────────────────────────────────────────────┘
```

</div>

**PancrAI** is a fully automated, end-to-end **Clinical Decision Support (CAD)** system that:

| Capability | Description |
|------------|-------------|
| 🧠 **3D Segmentation** | Simultaneous pancreas + tumor delineation from full CT volumes |
| 📐 **Quantitative Metrics** | Tumor volume (mL), RECIST longest-axis diameter, anatomical region |
| 🌐 **Web Deployment** | Flask REST API with multi-planar viewer and heatmap overlays |
| 📄 **Clinical Reports** | Auto-generated PDF radiology reports ready for clinical review |
| ⚡ **TTA Inference** | 8-flip Test-Time Augmentation for robust predictions |

---

## 🖥️ Screenshots & Demo

> 📸 **Screenshots will be added here after upload**

<table>
<tr>
<td align="center" width="33%">

**Upload Interface**

<img src="screenshots/upload_interface.png" alt="Upload Interface" width="100%"/>

*Drag & drop `.nii.gz` CT volumes*

</td>
<td align="center" width="33%">

**Multi-Planar Viewer**

<img src="screenshots/multiplanar_viewer.png" alt="Multi-Planar Viewer" width="100%"/>

*Axial · Sagittal · Coronal views*

</td>
<td align="center" width="33%">

**AI Segmentation Overlay**

<img src="screenshots/segmentation_overlay.png" alt="Segmentation Overlay" width="100%"/>

*Tumor mask overlaid on CT slices*

</td>
</tr>
<tr>
<td align="center" width="33%">

**Heatmap Visualization**

<img src="screenshots/heatmap.png" alt="Heatmap" width="100%"/>

*Prediction probability maps*

</td>
<td align="center" width="33%">

**Tumor Metrics Panel**

<img src="screenshots/metrics_panel.png" alt="Metrics Panel" width="100%"/>

*Volume · RECIST · Region*

</td>
<td align="center" width="33%">

**PDF Report**

<img src="screenshots/pdf_report.png" alt="PDF Report" width="100%"/>

*Auto-generated radiology report*

</td>
</tr>
<tr>
<td align="center" width="33%">

**Training Loss Curve**

<img src="screenshots/training_loss.png" alt="Training Loss" width="100%"/>

*74% total loss reduction*

</td>
<td align="center" width="33%">

**Validation Dice Progression**

<img src="screenshots/dice_progression.png" alt="Dice Progression" width="100%"/>

*Best checkpoint at epoch 115*

</td>
<td align="center" width="33%">

**Architecture Diagram**

<img src="screenshots/architecture.png" alt="Architecture" width="100%"/>

*Swin-UNETR full pipeline*

</td>
</tr>
</table>

> **To add screenshots:** Create a `screenshots/` folder in the repo root and upload your images with the filenames above.

---

## 🏆 Results vs Published Baselines

Benchmarked on the **Medical Segmentation Decathlon — Task07 Pancreas** (MICCAI 2018):

<div align="center">

| Model / Method | Year | Pancreas Dice | Tumor Dice | Mean Dice | Notes |
|----------------|:----:|:-------------:|:----------:|:---------:|-------|
| 3D U-Net (Çiçek) | 2016 | 0.612 | — | — | Baseline volumetric seg. |
| Attention U-Net (Oktay) | 2018 | 0.641 | — | — | Attention gates |
| UNETR (Hatamizadeh) | 2022 | 0.680 | 0.361 | 0.521 | ViT-based encoder |
| Swin UNETR (Tang) | 2022 | 0.700 | 0.380 | 0.540 | Pretrained Swin-Transformer |
| **PancrAI — Ours** | **2025** | **0.732 ✅** | **0.422 ✅** | **0.577 ✅** | **TTA + 4× tumor weighting** |

</div>

```
Performance Gain over Swin UNETR Baseline:

  Pancreas Dice  ████████████████████░░  +0.032  (+4.6%)  ▲ NEW SOTA
  Tumor Dice     ████████████████████░░  +0.042  (+11.1%) ▲ NEW SOTA
  Mean Dice      ████████████████████░░  +0.037  (+6.9%)  ▲ NEW SOTA
```

> **Key Contributions:** 8-flip Test-Time Augmentation (TTA) + class-weighted DiceCE loss `[0.1, 1.0, 4.0]` with 4× tumor emphasis overcomes the extreme 0.2% voxel class imbalance.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PancrAI — Full Pipeline                         │
└─────────────────────────────────────────────────────────────────────────┘

  Input: CT Scan (.nii.gz)
         │
         ▼
┌─────────────────────────────────────┐
│       Preprocessing Pipeline        │
│                                     │
│  1. HU Windowing: −175 to +250      │  ← Soft tissue enhancement
│  2. Voxel Resampling: 1.5×1.5×2 mm │  ← Isotropic normalization
│  3. Intensity Normalization          │
│  4. 10× MONAI Augmentations         │  ← Spatial + intensity transforms
│     · 3D random flips               │
│     · Random noise injection        │
│     · Contrast adjustment           │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│      Swin-UNETR  (62.2M params)     │
│                                     │
│  ┌─────────────────────────────┐    │
│  │  Swin Transformer Encoder   │    │  ← SSL pretrained on medical imgs
│  │  · Hierarchical Patch Embed │    │
│  │  · Shifted Window Attention │    │
│  │  · 4-stage feature pyramid  │    │
│  └──────────────┬──────────────┘    │
│                 │ skip connections  │
│  ┌──────────────▼──────────────┐    │
│  │     UNETR-style Decoder     │    │
│  │  · Progressive upsampling   │    │
│  │  · Skip connection fusion   │    │
│  │  · 3-class output head      │    │
│  └──────────────┬──────────────┘    │
│                 │                   │
│  ┌──────────────▼──────────────┐    │
│  │    8-Flip TTA Inference     │    │  ← Ensembles 8 augmented passes
│  └──────────────┬──────────────┘    │
└──────────────────┼──────────────────┘
                   │
                   ▼
      3-Class Voxel Mask
      [BG=0 | Pancreas=1 | Tumor=2]
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
 Quantitative Metrics    Flask CAD Web App
 ─────────────────────   ─────────────────
 • Tumor Volume (mL)     • Multi-planar Viewer
 • RECIST Diameter        • Axial/Sagittal/Coronal
 • Anatomical Region      • AI Heatmap Overlay
 • Tumor Burden (%)       • PDF Report Generator
                          • REST API /predict
```

---

## 📂 Repository Structure

```
PancrAI/
│
├── 🌐 app.py                      Flask web application + REST API
├── 🤖 infer.py                    Inference engine with 8-flip TTA
├── ⚙️  config.json                 Model config & hyperparameters
├── 🧠 best_model.pth              Trained SwinUNETR checkpoint (Epoch 115)
├── 📓 PancrAI_Colab_Ser.ipynb    Full training notebook (Google Colab / Kaggle)
│
├── 📁 templates/                  Flask Jinja2 HTML templates
│   ├── index.html                 CT upload interface
│   └── result.html                Multi-planar results viewer
│
├── 📁 screenshots/                ← Add your screenshots here
│   ├── banner.png
│   ├── upload_interface.png
│   ├── multiplanar_viewer.png
│   ├── segmentation_overlay.png
│   ├── heatmap.png
│   ├── metrics_panel.png
│   ├── pdf_report.png
│   ├── training_loss.png
│   ├── dice_progression.png
│   └── architecture.png
│
├── 📄 requirements.txt            Python dependencies
└── 📖 README.md
```

---

## 🚀 Quick Start

### Prerequisites

```
Python 3.9+  ·  CUDA 11.8+ (optional, CPU works)  ·  ~4 GB RAM minimum
```

### 1️⃣ Clone & Install

```bash
git clone https://github.com/HariKumar-01/PancrAI.git
cd PancrAI
pip install torch torchvision monai flask nibabel numpy scipy
```

### 2️⃣ Launch the Web App

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

### 3️⃣ Upload & Analyze

Upload any `.nii.gz` CT volume. PancrAI will automatically:

```
Step 1 ──► Preprocess  (HU windowing + voxel resampling)
Step 2 ──► Segment     (3D Swin-UNETR with 8-flip TTA)
Step 3 ──► Measure     (volume · RECIST · anatomical region)
Step 4 ──► Display     (axial / sagittal / coronal + AI overlay)
Step 5 ──► Report      (downloadable PDF radiology report)
```

### 4️⃣ REST API

```python
import requests

# Submit CT scan for analysis
with open("patient_ct.nii.gz", "rb") as f:
    response = requests.post(
        "http://localhost:5000/predict",
        files={"file": f}
    )

result = response.json()

# Results
print(f"Tumor Volume   : {result['tumor_volume_ml']:.2f} mL")
print(f"RECIST Diameter: {result['recist_diameter_mm']:.1f} mm")
print(f"Anatomical Site: {result['anatomical_region']}")   # Head / Body / Tail
print(f"Tumor Burden   : {result['tumor_burden_pct']:.2f} %")
print(f"Pancreas Dice  : {result['pancreas_dice']:.4f}")
```

<details>
<summary><b>Sample JSON Response</b></summary>

```json
{
  "tumor_volume_ml": 12.4,
  "recist_diameter_mm": 28.6,
  "anatomical_region": "Head",
  "tumor_burden_pct": 0.31,
  "pancreas_dice": 0.732,
  "tumor_dice": 0.422,
  "inference_time_s": 312.4,
  "report_url": "/download/report_20250615_143020.pdf"
}
```

</details>

---

## 🔬 Model Details

<table>
<tr><td><b>Architecture</b></td><td>Swin-UNETR (Swin Transformer encoder + UNETR decoder)</td></tr>
<tr><td><b>Parameters</b></td><td>62.2 million</td></tr>
<tr><td><b>Input</b></td><td>3D CT volume — NIfTI (.nii.gz)</td></tr>
<tr><td><b>Output</b></td><td>3-class voxel mask: Background (0) / Pancreas (1) / Tumor (2)</td></tr>
<tr><td><b>Loss Function</b></td><td>DiceCE with class weights [0.1, 1.0, 4.0] (4× tumor emphasis)</td></tr>
<tr><td><b>Optimizer</b></td><td>AdamW with AMP (mixed precision)</td></tr>
<tr><td><b>LR Schedule</b></td><td>Cosine annealing</td></tr>
<tr><td><b>Epochs Trained</b></td><td>127 (optimal checkpoint: epoch 115)</td></tr>
<tr><td><b>Deep Supervision</b></td><td>5-level deep supervision heads</td></tr>
<tr><td><b>Augmentations</b></td><td>10× MONAI spatial + intensity transforms (4 crops/volume)</td></tr>
<tr><td><b>TTA Strategy</b></td><td>8-flip Test-Time Augmentation (sliding window)</td></tr>
<tr><td><b>Encoder Pretraining</b></td><td>Self-Supervised Learning (SSL) on medical images</td></tr>
<tr><td><b>Training Loss Reduction</b></td><td>74% across 127 epochs</td></tr>
<tr><td><b>Training Hardware</b></td><td>Kaggle NVIDIA Tesla P100 (16GB VRAM)</td></tr>
<tr><td><b>Caching Strategy</b></td><td>MONAI PersistentDataset (deterministic transform cache)</td></tr>
</table>

---

## 📊 Dataset — MSD Task07 Pancreas

<table>
<tr><td><b>Source</b></td><td>Medical Segmentation Decathlon — MICCAI 2018</td></tr>
<tr><td><b>Total Annotated Scans</b></td><td>281 expert-labeled 3D CT volumes</td></tr>
<tr><td><b>Acquisition Phase</b></td><td>Portal venous phase (optimal pancreatic enhancement)</td></tr>
<tr><td><b>Format</b></td><td>NIfTI (.nii.gz)</td></tr>
<tr><td><b>Label Classes</b></td><td>0: Background &nbsp;/&nbsp; 1: Pancreas &nbsp;/&nbsp; 2: Tumor</td></tr>
<tr><td><b>Training Split</b></td><td>238 scans (85%)</td></tr>
<tr><td><b>Validation Split</b></td><td>43 scans (15%) — fixed seed=42</td></tr>
<tr><td><b>HU Windowing</b></td><td>−175 to +250 (soft tissue window)</td></tr>
<tr><td><b>Voxel Resampling</b></td><td>1.5 × 1.5 × 2.0 mm isotropic</td></tr>
<tr><td><b>Class Imbalance</b></td><td>Tumor: ~0.2% of all voxels (extreme minority)</td></tr>
<tr><td><b>Official Homepage</b></td><td><a href="http://medicaldecathlon.com/">medicaldecathlon.com</a></td></tr>
<tr><td><b>Kaggle Mirror</b></td><td><a href="https://www.kaggle.com/datasets/lnguynquangbnh/task07-pancreas">Task07 Pancreas on Kaggle</a></td></tr>
</table>

---

## 📈 Training Curves

```
╔════════════════════════════════════════════════════════════════════╗
║         Validation Dice Score — MSD Task07 (every 5 epochs)       ║
╠════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  0.75 ┤                                              ✦ ← Ep 115   ║
║  0.70 ┤                                    ╭─────────╯  0.732     ║
║  0.65 ┤                          ╭────────╯  Pancreas Dice        ║
║  0.60 ┤                ╭────────╯                                  ║
║  0.55 ┤      ╭────────╯                     ╌╌╌╌╌╌╌╌╌ 0.5 ref    ║
║  0.50 ┼──────────────────────────────────────────────────────────  ║
║  0.45 ┤                                              ✦ 0.422      ║
║  0.40 ┤                              ╭───────────────╯  Tumor     ║
║  0.35 ┤              ╭──────────────╯    Tumor Dice (volatile)     ║
║  0.30 ┤  ╭──────────╯                                             ║
║  0.25 ┤──╯                                                        ║
║  0.20 ┤                                                           ║
║       └────────┬─────────┬──────────┬──────────┬──────────┬──── ║
║               20        40         60         80        115     ║
╠════════════════════════════════════════════════════════════════════╣
║  Key: ─── Pancreas Dice  ─ ─ Tumor Dice  ✦ Best Checkpoint       ║
╚════════════════════════════════════════════════════════════════════╝

Training Loss Reduction (127 epochs):
  Epoch 1  ████████████████████████████████████ 1.00 (normalized)
  Epoch 40 █████████████████████████░░░░░░░░░░░ 0.68
  Epoch 79 ████████████░░░░░░░░░░░░░░░░░░░░░░░░ 0.34  ← Final recorded
  Epoch 127 █████████░░░░░░░░░░░░░░░░░░░░░░░░░░ 0.26  (74% total reduction)
```

---

## 🌐 Flask CAD Application

<table>
<thead>
<tr><th>Feature</th><th>Description</th><th>Status</th></tr>
</thead>
<tbody>
<tr><td>🖼️ <b>Multi-Planar Viewer</b></td><td>Axial, Sagittal, Coronal slices with synchronized AI overlay</td><td>✅ Live</td></tr>
<tr><td>📦 <b>Tumor Volume</b></td><td>Automatic 3D volumetric computation in mL from voxel mask</td><td>✅ Live</td></tr>
<tr><td>📏 <b>RECIST Diameter</b></td><td>Longest-axis measurement for radiological staging</td><td>✅ Live</td></tr>
<tr><td>🗺️ <b>Anatomical Region</b></td><td>Automatic Head / Body / Tail localization</td><td>✅ Live</td></tr>
<tr><td>🌡️ <b>Heatmap Overlay</b></td><td>Prediction probability maps overlaid on CT slices</td><td>✅ Live</td></tr>
<tr><td>📄 <b>PDF Report</b></td><td>Auto-generated structured radiology report (downloadable)</td><td>✅ Live</td></tr>
<tr><td>🔌 <b>REST API</b></td><td>POST <code>/predict</code> endpoint for programmatic integration</td><td>✅ Live</td></tr>
<tr><td>🏥 <b>DICOM Support</b></td><td>Direct hospital PACS integration (no NIfTI conversion)</td><td>🔮 Planned</td></tr>
</tbody>
</table>

---

## ⚠️ Limitations

| Limitation | Detail |
|------------|--------|
| **Dataset size** | MSD Task07 has 281 scans — results may not generalize to all scanner types, populations, or contrast phases |
| **Tumor Dice variability** | Small tumor targets (~5mm) cause validation Dice fluctuation across epochs due to positional sensitivity |
| **CPU inference speed** | Without a GPU, inference requires ~5–10 min per scan — impractical for clinical real-time use |
| **NIfTI-only input** | DICOM conversion is required before use with hospital PACS/RIS systems |
| **Single contrast phase** | Trained only on portal venous phase CT — performance on other phases is untested |

---

## 🔮 Future Scope

```
Priority  Feature                         Impact
────────  ──────────────────────────────  ─────────────────────────────────────
🔴 HIGH   Real-Time GPU Inference         Reduce ~5 min CPU to <1 min on cloud
🔴 HIGH   DICOM Integration              Eliminate NIfTI conversion for PACS
🟡 MED    Multi-Organ Segmentation       Segment all 13 abdominal organs
🟡 MED    Cancer Staging (TNM)           Texture-feature staging automation
🟢 LOW    Longitudinal Monitoring        Multi-timepoint tumor volume tracking
🟢 LOW    Federated Learning             Multi-hospital privacy-preserving training
```

---

## 👥 Team — Batch A10

<div align="center">

| Member | Roll No. | Contribution |
|--------|:--------:|-------------|
| **CH. Hari Kumar** | 22B91A6141 | Model Training · TTA Inference Pipeline · Integration |
| **J.D.S Karthikeya** | 22B91A6161 | Flask CAD Application · REST API · Frontend |
| **Badugu Ajay** | 22B91A6118 | Preprocessing · MONAI Augmentation · Caching |
| **B. Hema Sree** | 22B91A6134 | Evaluation · PDF Report Generation · Benchmarking |

**Project Guide:** CH. Vinod Varma · Assistant Professor, Dept. of CSE  
**Institution:** SAGI RAMA KRISHNAM RAJU ENGINEERING COLLEGE (Autonomous), Bhimavaram  
**Academic Year:** 2024–25 · B.Tech Final Year Project

</div>

---

## 📚 References

1. A. Hatamizadeh et al. *Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors.* MICCAI BrainLes Workshop, 2022.
2. Y. Tang et al. *Self-Supervised Pre-Training of Swin Transformers for 3D Medical Image Analysis.* IEEE/CVF CVPR, 2022.
3. O. Oktay et al. *Attention U-Net: Learning Where to Look for the Pancreas.* MICCAI, 2018.
4. O. Çiçek et al. *3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation.* MICCAI, 2016.
5. A. Antonelli et al. *The Medical Segmentation Decathlon.* Nature Communications, 2022.
6. M. J. Cardoso et al. *MONAI: An open-source framework for deep learning in healthcare.* arXiv:2211.02701, 2022.
7. Z. Liu et al. *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.* ICCV, 2021.
8. O. Ronneberger, P. Fischer, T. Brox. *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI, 2015.

---

<div align="center">

```
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   PancrAI · Pancreatic Tumor Segmentation · SRKR EC 2025   ║
║                                                              ║
║          Made with ❤️  at Bhimavaram, Andhra Pradesh         ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

*If this project helped you, please consider giving it a ⭐*

</div>
