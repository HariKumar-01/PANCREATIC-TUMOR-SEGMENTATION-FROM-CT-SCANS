# PancrAI — Pancreatic Tumor Segmentation from CT Scans

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red?style=for-the-badge&logo=pytorch)
![MONAI](https://img.shields.io/badge/MONAI-1.3-green?style=for-the-badge)
![Flask](https://img.shields.io/badge/Flask-3.0-black?style=for-the-badge&logo=flask)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**Final Year B.Tech Project — Department of CSE (AI & ML)**
*SAGI RAMA KRISHNAM RAJU ENGINEERING COLLEGE (Autonomous), Bhimavaram*

---

## Project Overview

**PancrAI** is an AI-powered Clinical Decision Support (CAD) system that performs automated 3D pancreatic tumor segmentation from CT scans using the **Swin-UNETR** transformer architecture.

Pancreatic cancer carries a devastating 5-year survival rate of **less than 12%**, largely due to late-stage detection. PancrAI addresses this by:

- Automatically detecting and segmenting tumors in 3D CT volumes
- Computing tumor volume (mL), RECIST diameter, and anatomical location (Head/Body/Tail)
- Providing a Flask web interface with multi-planar CT viewer (axial, sagittal, coronal)
- Generating automated PDF radiology reports

---

## Key Results

| Metric | PancrAI (Ours) | Swin UNETR Baseline | UNETR |
|--------|:--------------:|:-------------------:|:-----:|
| **Pancreas Dice** | **0.732** | 0.700 | 0.680 |
| **Tumor Dice** | **0.422** | 0.380 | 0.361 |
| **Mean Dice** | **0.577** | 0.540 | 0.521 |
| Best Epoch | 115 / 127 | — | — |

PancrAI surpasses all published baselines on the **MSD Task07 Pancreas** benchmark dataset.

---

## System Architecture

```
CT Scan (NIfTI) --> Preprocessing --> Swin-UNETR --> 3D Segmentation Mask
                         |                                    |
               HU Windowing (-175,+250)          Pancreas + Tumor Labels
               Voxel Resampling 1.5x1.5x2mm               |
               10x MONAI Augmentations         +-----------+----------+
                                               |                      |
                                        Volume (mL)           Flask CAD App
                                        RECIST Axis           Multi-planar Viewer
                                        Tumor Region          PDF Report
```

---

## Repository Structure

```
PancrAI/
├── app.py                  # Flask web application (REST API + UI)
├── infer.py                # Inference pipeline with 8-flip TTA
├── config.json             # Model and pipeline configuration
├── best_model.pth          # Trained SwinUNETR weights (62.2M params)
├── PancrAI_Colab_Ser.ipynb # Google Colab training notebook
├── templates/              # Flask HTML templates
│   ├── index.html
│   └── result.html
└── README.md
```

---

## Quick Start

**1. Clone & Install**
```bash
git clone https://github.com/YOUR_USERNAME/PancrAI.git
cd PancrAI
pip install -r requirements.txt
```

**2. Run Flask App**
```bash
python app.py
# Open http://localhost:5000
```

**3. Upload a CT Scan**

Upload any `.nii.gz` CT volume via the web interface. PancrAI will preprocess it, run 3D segmentation with 8-flip TTA, display axial/sagittal/coronal views with AI overlay, and generate a downloadable PDF report.

---

## Model Details

| Component | Detail |
|-----------|--------|
| Architecture | Swin-UNETR (Swin Transformer + UNETR decoder) |
| Parameters | 62.2 million |
| Input | 3D CT volume (NIfTI .nii.gz) |
| Output | 3-class voxel mask (Background / Pancreas / Tumor) |
| Loss Function | Class-weighted DiceCE [0.1, 1.0, 4.0] |
| Epochs Trained | 127 (best checkpoint: epoch 115) |
| Augmentations | 10x MONAI spatial + intensity transforms |
| TTA | 8-flip Test-Time Augmentation |
| Pretrained Encoder | SSL pretrained Swin-Transformer |

---

## Dataset — MSD Task07 Pancreas

| Property | Value |
|----------|-------|
| Source | Medical Segmentation Decathlon — MICCAI 2018 |
| Total Scans | 281 annotated 3D CT volumes |
| Format | NIfTI (.nii.gz) |
| Classes | Background / Pancreas / Tumor |
| Train Split | 238 scans (85%) |
| Validation | 43 scans (15%) |
| HU Windowing | -175 to +250 (soft tissue) |
| Voxel Resampling | 1.5 x 1.5 x 2.0 mm |

---

## Flask CAD Application Features

- Multi-planar CT Viewer: Axial, Sagittal, Coronal views with AI overlay
- Tumor Volume: Automatic volumetric computation in mL
- RECIST Diameter: Longest axis measurement for staging
- Anatomical Region: Head / Body / Tail classification
- Heatmap Overlay: Probability maps on CT slices
- PDF Report: Auto-generated radiology report download
- REST API: POST /predict endpoint for programmatic access

---

## Future Scope

| Feature | Description |
|---------|-------------|
| Multi-Organ Segmentation | Extend to all 13 abdominal organs |
| Cancer Staging | Predict stage from tumor texture features |
| Real-Time GPU Inference | AWS/GCP deployment under 1 min latency |
| DICOM Integration | Direct hospital PACS/RIS support |
| Longitudinal Monitoring | Multi-timepoint tumor tracking |
| Federated Learning | Privacy-preserving multi-hospital training |

---

## Team — Batch A10

| Name | Roll Number |
|------|-------------|
| CH. Hari Kumar | 22B91A6141 |
| J.D.S Karthikeya | 22B91A6161 |
| Badugu Ajay | 22B91A6118 |
| B. Hema Sree | 22B91A6134 |

**Project Guide:** CH. Vinod Varma, Assistant Professor, Dept. of CSE

---

## References

1. Hatamizadeh et al. *Swin UNETR: Swin Transformers for Semantic Segmentation.* MICCAI BrainLes, 2022.
2. Tang et al. *Self-Supervised Pre-Training of Swin Transformers for 3D Medical Image Analysis.* CVPR, 2022.
3. Oktay et al. *Attention U-Net: Learning Where to Look for the Pancreas.* MICCAI, 2018.
4. Antonelli et al. *The Medical Segmentation Decathlon.* Nature Communications, 2022.
5. Cardoso et al. *MONAI: An open-source framework for deep learning in healthcare.* arXiv:2211.02701, 2022.

---

Made with love at **SRKR Engineering College, Bhimavaram** — 2024-25
