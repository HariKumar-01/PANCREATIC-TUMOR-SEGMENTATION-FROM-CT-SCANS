<div align="center">

# 🔬 PancrAI
### Pancreatic Tumor Segmentation from CT Scans
#### *Transformer-Based 3D Deep Learning · Clinical Decision Support*

<br/>

[![Python](https://img.shields.io/badge/Python-3.10-0096C7?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-0096C7?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![MONAI](https://img.shields.io/badge/MONAI-1.3-00B4D8?style=for-the-badge&logoColor=white)](https://monai.io)
[![Flask](https://img.shields.io/badge/Flask-3.0-023E8A?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-48CAE4?style=for-the-badge)](LICENSE)

[![Model](https://img.shields.io/badge/Swin--UNETR-62.2M_params-00B4D8?style=flat-square)]()
[![Dataset](https://img.shields.io/badge/MSD_Task07-281_CT_scans-0096C7?style=flat-square)]()
[![Best Dice](https://img.shields.io/badge/Mean_Dice-0.577_%E2%86%91_SOTA-2DC653?style=flat-square)]()
[![Epochs](https://img.shields.io/badge/Trained-127_epochs-48CAE4?style=flat-square)]()

<br/>

> 🎓 **Final Year B.Tech Project** · IV/IV · AI & ML · Batch A10  
> 🏫 **SAGI RAMA KRISHNAM RAJU ENGINEERING COLLEGE (Autonomous), Bhimavaram**  
> 👨‍🏫 **Guide:** CH. Vinod Varma · Assistant Professor, Dept. of CSE

</div>

---

## 🧬 The Problem

Pancreatic cancer has a **< 12% five-year survival rate** — largely because ~80% of cases are diagnosed only at Stage III or IV, when curative surgery is no longer possible. The root cause: tumors appear nearly **iso-dense** against surrounding abdominal tissue in CT scans, making manual 3D segmentation slow, error-prone, and clinically infeasible at scale.

---

## 💡 Our Solution — PancrAI

PancrAI is a fully automated end-to-end **Clinical Decision Support (CAD)** system that:

- 🧠 Segments the pancreas and tumor simultaneously in full 3D CT volumes using **Swin-UNETR**
- 📐 Computes **tumor volume** (mL), **RECIST diameter**, and **anatomical region** (Head / Body / Tail)
- 🌐 Deploys a **Flask web app** with multi-planar viewer (axial · sagittal · coronal) and heatmap overlays
- 📄 Auto-generates **PDF radiology reports** ready for clinical review

---

## 🏆 Results vs Published Baselines

Benchmarked on **MSD Task07 Pancreas** (MICCAI 2018 · 281 annotated 3D CT scans):

| Model | Year | Pancreas Dice | Tumor Dice | Mean Dice |
|-------|:----:|:-------------:|:----------:|:---------:|
| 3D U-Net (Çiçek) | 2016 | 0.612 | — | — |
| Attention U-Net (Oktay) | 2018 | 0.641 | — | — |
| UNETR (Hatamizadeh) | 2022 | 0.680 | 0.361 | 0.521 |
| Swin UNETR (Tang) | 2022 | 0.700 | 0.380 | 0.540 |
| **PancrAI — Ours** | **2025** | **0.732 ✅** | **0.422 ✅** | **0.577 ✅** |

> **+4.6% Pancreas Dice · +11.1% Tumor Dice** over the Swin-UNETR baseline, achieved through **8-flip TTA** and **4× tumor class weighting** in DiceCE loss.

### Validation Results at Best Checkpoint (Epoch 115)

![Best Results](screenshots/best_results.jpg)

### Comparison with Literature

![Comparison Table](screenshots/comparison_table.jpg)

---

## 📊 Training Results

### Loss Curve (74% total reduction over 127 epochs)

![Training Loss](screenshots/training_loss.jpg)

### Validation Dice Score Progression

![Dice Progression](screenshots/dice_progression.jpg)

---

## 🌐 Web Application

### Upload Interface

![Web Upload Interface](screenshots/web_upload.jpg)

### Multi-Planar CAD Viewer (Axial · Sagittal · Coronal)

![CAD Viewer](screenshots/cad_viewer.jpg)

### Auto-Generated AI Radiology Report

![PDF Report](screenshots/pdf_report.jpg)

### CT Segmentation — Visual Sanity Check

![CT Segmentation](screenshots/ct_segmentation.jpg)

> **Green** = Pancreas · **Red** = Tumor · Segmentation mask aligned spatially with CT anatomy at z=38

---

## 🏗️ System Architecture

```
  Input: CT Scan (.nii.gz)
         │
         ▼
┌─────────────────────────────────────┐
│        Preprocessing Pipeline       │
│  • HU Windowing: −175 to +250 HU   │
│  • Voxel Resampling: 1.5×1.5×2 mm  │
│  • 10× MONAI Augmentations          │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│    Swin-UNETR  (62.2M parameters)   │
│  • SSL Pretrained Swin Encoder      │
│  • Hierarchical Shifted Window Attn │
│  • UNETR-style Skip Connections     │
│  • 5-Level Deep Supervision         │
│  • 8-Flip Test-Time Augmentation    │
└──────────────────┬──────────────────┘
                   │
        3-Class Voxel Mask
     [BG=0 | Pancreas=1 | Tumor=2]
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
 Quantitative Metrics    Flask CAD App
 • Volume (mL)           • Multi-Planar Viewer
 • RECIST Diameter        • Heatmap Overlay
 • Anatomical Region      • PDF Report
 • Tumor Burden %         • REST API /predict
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
| **Optimizer** | AdamW with AMP (mixed precision) |
| **LR Schedule** | Cosine annealing |
| **Epochs Trained** | 127 (best checkpoint: epoch 115) |
| **Deep Supervision** | 5-level deep supervision heads |
| **Augmentations** | 10× MONAI spatial + intensity transforms |
| **TTA Strategy** | 8-flip Test-Time Augmentation |
| **Encoder Pretraining** | Self-Supervised Learning (SSL) on medical images |
| **Training Hardware** | Kaggle NVIDIA Tesla P100 (16 GB VRAM) |

---

## 📂 Repository Structure

```
PancrAI/
├── app.py                    Flask web application + REST API
├── infer.py                  Inference engine with 8-flip TTA
├── config.json               Model config & hyperparameters
├── best_model.pth            Trained SwinUNETR checkpoint (epoch 115)
├── PancrAI_Colab_Ser.ipynb   Full training notebook (Google Colab)
├── templates/
│   ├── index.html            CT upload interface
│   └── result.html           Multi-planar results viewer
├── screenshots/              PPT slide images used in README
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/HariKumar-01/PancrAI.git
cd PancrAI
pip install torch torchvision monai flask nibabel numpy scipy
```

### 2. Run the Web App

```bash
python app.py
# Open http://localhost:5000
```

### 3. REST API

```python
import requests

with open("ct_scan.nii.gz", "rb") as f:
    response = requests.post("http://localhost:5000/predict", files={"file": f})

result = response.json()
print(result["tumor_volume_ml"])       # e.g. 7.3
print(result["recist_diameter_mm"])    # e.g. 34.0
print(result["anatomical_region"])     # Head / Body / Tail
```

---

## ⚠️ Limitations

| Limitation | Detail |
|------------|--------|
| Dataset size | 281 scans — may not generalize to all scanner types or contrast phases |
| Tumor Dice variability | Small targets (~5 mm) cause validation fluctuation across epochs |
| CPU inference | ~5–10 min per scan without GPU |
| NIfTI-only input | DICOM conversion required for hospital PACS integration |

---

## 🔮 Future Scope

| Priority | Feature |
|:--------:|---------|
| 🔴 High | Real-Time GPU Inference (AWS/GCP — <1 min) |
| 🔴 High | Full DICOM Integration for hospital PACS/RIS |
| 🟡 Med | Multi-Organ Segmentation (all 13 abdominal organs) |
| 🟡 Med | Cancer Staging via texture feature extraction |
| 🟢 Low | Longitudinal Tumor Monitoring across timepoints |
| 🟢 Low | Federated Learning for multi-hospital privacy training |

---

## 📚 References

1. A. Hatamizadeh et al. *Swin UNETR: Swin Transformers for Semantic Segmentation.* MICCAI BrainLes, 2022.
2. Y. Tang et al. *Self-Supervised Pre-Training of Swin Transformers for 3D Medical Image Analysis.* CVPR, 2022.
3. O. Oktay et al. *Attention U-Net: Learning Where to Look for the Pancreas.* MICCAI, 2018.
4. O. Çiçek et al. *3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation.* MICCAI, 2016.
5. A. Antonelli et al. *The Medical Segmentation Decathlon.* Nature Communications, 2022.
6. M. J. Cardoso et al. *MONAI: An open-source framework for deep learning in healthcare.* arXiv:2211.02701, 2022.
7. Z. Liu et al. *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.* ICCV, 2021.
8. O. Ronneberger et al. *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI, 2015.

---

## 👥 Team — Batch A10

| Name | Roll No. | Role |
|------|:--------:|------|
| CH. Hari Kumar | 22B91A6141 | Model Training · TTA Pipeline |
| J.D.S Karthikeya | 22B91A6161 | Flask CAD Application |
| Badugu Ajay | 22B91A6118 | Preprocessing · Augmentation |
| B. Hema Sree | 22B91A6134 | Evaluation · Report Generation |

**Guide:** CH. Vinod Varma · Assistant Professor, Dept. of CSE · SRKR Engineering College

---

<div align="center">

Made with ❤️ at **SRKR Engineering College, Bhimavaram** · Academic Year 2024–25

*If this project helped you, consider giving it a ⭐*

</div>
