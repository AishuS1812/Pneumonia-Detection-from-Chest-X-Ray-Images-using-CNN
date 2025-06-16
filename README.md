# 🫁 Pneumonia Detection from Chest X-ray Images

## 📌 Project Overview
This project builds a deep learning model using Convolutional Neural Networks (CNN) to detect pneumonia from chest X-ray images. The end-to-end pipeline includes data preprocessing, model training, evaluation, and deployment via a web app. It can be applied in telemedicine platforms or hospital triage systems, especially in resource-limited settings.

## 🧠 Objective
To develop a **binary image classification** model that classifies chest X-ray images into:
- **NORMAL** (healthy lungs)
- **PNEUMONIA** (infected lungs)

## 📂 Dataset
- **Source**: [Kaggle – Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Format**: JPEG images organized in subfolders by class and data split

### 📊 Data Distribution
| Split      | NORMAL | PNEUMONIA | Total |
|------------|--------|-----------|-------|
| Train      | 1,082  | 3,110     | 4,192 |
| Validation | 267    | 773       | 1,040 |
| Test       | 234    | 390       | 624   |

---

## 🧪 Model Development

### 🔄 Preprocessing
- Resize images to 224x224 pixels
- Normalize pixel values to range [0, 1]
- Data augmentation (training set only):
  - Horizontal flip
  - Random rotation
  - Random zoom

### 🧱 CNN Architecture
- **Model**: Transfer Learning with **MobileNetV2**
- **Layers**:
  - Pretrained base (frozen)
  - GlobalAveragePooling2D
  - Dropout (rate: 0.3)
  - Dense layer with 2 units (softmax)

### ⚙️ Training Setup
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy
- **Epochs**: 10 (with EarlyStopping)
- **Batch Size**: 32

---

## 📊 Evaluation Metrics
| Class     | Precision | Recall | F1-score | Support |
|-----------|-----------|--------|----------|---------|
| NORMAL    | 0.96      | 0.73   | 0.83     | 234     |
| PNEUMONIA | 0.86      | 0.98   | 0.91     | 390     |

- **Overall Accuracy**: 88%
- Also included:
  - Confusion Matrix
  - Accuracy/Loss curves
  - Visual sample predictions

---

## 💾 Deployment

### 🌐 Web App (Streamlit)
A lightweight and interactive web interface built using **Streamlit**.

#### 🖥 Features
- Upload `.jpg`, `.jpeg`, or `.png` chest X-ray images
- Get real-time prediction and confidence score
- Image preview and feedback

### 🔧 How to Run the App
Install dependencies:
```bash
pip install streamlit tensorflow pillow
