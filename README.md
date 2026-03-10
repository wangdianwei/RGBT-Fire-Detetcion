# Fire-aware Cross-modal Alignment Framework for UAV RGB–Thermal Fire and Smoke Detection


🔥 **FireRGBT-YOLO**

An RGB–thermal detection framework for **UAV-based wildfire monitoring**.  
This repository contains the official implementation of our method for **joint fire and smoke detection using RGB–thermal imagery**.

---

# 📖 Abstract

Detecting early-stage fire and smoke from UAV platforms is challenging due to **cross-modal misalignment, varying modality reliability, and weak thermal signatures of small fire targets**.

We propose a unified RGB–thermal detection framework that integrates:

- **Fire-aware Cross-modal Deformable Alignment (FCDAM)**  
  selectively aligns infrared features to RGB features only in fire-relevant regions.

- **Modality-Aware Mixture-of-Experts (MA-MoE)**  
  dynamically balances RGB and thermal information under different environmental conditions.

- **Fire Feature Extractor (FFE)**  
  enhances weak thermal fire responses to improve early-stage fire detection.

---

# ✨ Highlights

🔥 **Selective Cross-modal Alignment**  
Fire-aware deformable alignment reduces spatial mismatch between RGB and thermal modalities.

⚖️ **Adaptive Multimodal Fusion**  
The MA-MoE module dynamically adjusts modality importance depending on scene conditions.

🚁 **UAV-oriented Detection Framework**  
Designed for UAV wildfire monitoring where small targets and environmental variations are common.

---

# 📊 Performance

Results on the **RGBT-3M UAV dataset**

| Method | Input | mAP50 | mAP50–95 |
|------|------|------|------|
YOLOv11 | RGB | 94.0 | 62.6 |
YOLOv11 | IR | 92.2 | 61.8 |
YOLOv11 MidFusion | RGB+IR | 95.8 | 67.0 |
**Ours** | RGB+IR | **96.4** | **71.5** |

Our method improves detection accuracy while maintaining competitive inference efficiency.

---

# ⚙️ Environment

Tested environment:

- Python 3.10
- PyTorch 2.3.0
- CUDA 12.1
- Ultralytics YOLO framework

# 📂 Dataset

Experiments are conducted on the **RGBT-3M UAV dataset**, which provides synchronized RGB–thermal image pairs for wildfire monitoring.

The dataset can be downloaded from the official website:

https://complex.ustc.edu.cn/sjwwataset/list.htm

In this work, we focus on two detection categories:

- fire  
- smoke  

The original **person** category in the dataset is removed during preprocessing to focus on wildfire detection.

Dataset split used in our experiments:

- Training set: 7854 image pairs  
- Validation set: 3366 image pairs  

Please follow the dataset license and cite the original dataset paper if you use it in your research.
