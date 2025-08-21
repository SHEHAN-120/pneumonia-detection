# 🩺 Pneumonia Detection from Chest X-rays

## 📌 Project Overview
This project builds an end-to-end deep learning pipeline to automatically detect **pneumonia from chest X-ray images** using **PyTorch Lightning** and **ResNet18**. The dataset comes from the [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge).

The model not only classifies X-rays as **normal** or **pneumonia**, but also provides **explainability** via **Class Activation Maps (CAMs)** to highlight the regions of the lung that contributed most to the prediction.

---

## 🔹 Pipeline

### **1. Data Preprocessing**
- Load chest X-rays from **DICOM format**.
- Resize images to **224×224** (ResNet input size).
- Normalize pixel intensities (`mean=0.49, std=0.248`).
- Save processed images as `.npy` files for efficient loading.
- Split dataset into **train (24k)** and **validation (rest)**.

### **2. Model Training**
- Base model: **ResNet18 (pretrained on ImageNet)**.
- Modified:
  - First convolution: accept **1-channel (grayscale)** input.
  - Final FC layer: **1 output neuron** (binary classification).
- Loss: **BCEWithLogitsLoss** with class imbalance handling (`pos_weight`).
- Optimizer: **Adam (lr=1e-4)**.
- Augmentations: rotations, translations, random crops.
- Framework: **PyTorch Lightning** for training loops, checkpointing, logging.

### **3. Evaluation**
- Metrics: **Accuracy, Precision, Recall, Confusion Matrix**.
- Adjusted threshold to **0.25** → increased recall (critical for medical diagnosis).
- Achieved Better Result.

### **4. Interpretability (CAMs)**
- Implemented **Class Activation Maps** to show which lung regions influenced predictions.
- Heatmaps overlaid on X-rays to provide **trust and transparency** for clinicians.

### **5. Deployment Considerations** (In Future)
- Wrap model in **FastAPI** or **Flask** for hospital integration.
- Scale with GPU inference and quantization.
- Ensure compliance with **HIPAA/GDPR** (data anonymization, encryption).

---

---

## 🖼 Example Visualization

Heatmap showing pneumonia region:

![WhatsApp Image 2025-08-21 at 21 54 23](https://github.com/user-attachments/assets/5efea6d0-9900-41ed-b72b-4a9111cdd545)



---

## ⚡ Key Learnings
- **Recall > Accuracy** in medical AI .
- **Transfer learning** improves performance and reduces training time.
- **Explainability** (CAMs) is essential for clinical trust.
- **PyTorch Lightning** makes training clean and production-ready.

---

## ⚠️ Challenges Faced
- **Large dataset size** – RSNA dataset was heavy to download and required preprocessing.
- **Long training time** – Even with GPU, training was computationally intensive.
- **Class imbalance** – More normal cases than pneumonia, needed weighting.
- **DICOM handling** – Conversion to NumPy for faster data loading.
- **Explainability** – Implementing CAMs for better model trust.

---

## 🧠 Skills Gained
- Advanced **PyTorch & PyTorch Lightning** training pipelines.
- **Medical image preprocessing** (DICOM, normalization, augmentations).
- **Model interpretability** techniques like CAMs.
- **Evaluation metrics** in healthcare context (precision vs recall).
- **End-to-end ML workflow**: preprocessing → training → evaluation → deployment.

---

## 🚀 Getting Started

### Requirements
`requirements.txt`
```
pip==21.2.2
celluloid==0.2.0
dicom2nifti==2.3.0
jupyter==1.0.0
imgaug==0.4.0
tensorboard==2.7.0
torchio==0.18.57
torchmetrics==0.5.1
tqdm==4.62.3
pytorch-lightning==1.4.9
opencv-python==4.5.3.56
numpy==1.21.2
nibabel==3.2.1
matplotlib==3.4.3
pandas==1.1.5
pydicom==2.2.2
scikit-learn==1.0.1
```




---

## 👨‍⚕️ Author
**SHEHAN** – Deep Learning Enthusiast | Portfolio Project

---

## 📜 License
This project is for **educational purposes only**.
