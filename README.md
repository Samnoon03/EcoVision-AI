# ♻️ EcoVision AI – Waste Classification

EcoVision AI is a Deep Learning based image classification project that
automatically identifies different types of waste from images.

The project uses a **MobileNetV2 Transfer Learning model** trained on the
TrashNet dataset and provides an interactive **Streamlit web application**
for real-time waste classification.

---

## 🚀 Project Overview

Proper waste classification is an important step toward efficient recycling
and sustainable waste management.

EcoVision AI uses Computer Vision and Deep Learning to classify waste images
into six different categories:

- 📦 Cardboard
- 🪟 Glass
- ⚙️ Metal
- 📄 Paper
- 🧴 Plastic
- 🗑️ Trash

---

## 🧠 Model

The project experimented with three different CNN approaches:

| Model | Test Accuracy |
|---|---:|
| CNN Model 1 | 74.67% |
| Improved CNN Model 2 | 70.24% |
| **MobileNetV2 Transfer Learning** | **78.59%** |

The **MobileNetV2 model** achieved the highest test accuracy and was selected
as the final model.

### Final Model Performance

- **Test Accuracy:** 78.59%
- **Macro F1-Score:** 76.89%
- **Weighted F1-Score:** 78.52%

---

## 📊 Dataset

The project uses the **TrashNet** image dataset.

### Dataset Statistics

- **Total Images:** 2,527
- **Number of Classes:** 6
- **Image Resolution:** 512 × 384
- **Color Mode:** RGB

### Class Distribution

| Class | Images |
|---|---:|
| Cardboard | 403 |
| Glass | 501 |
| Metal | 410 |
| Paper | 594 |
| Plastic | 482 |
| Trash | 137 |

The dataset was inspected for:

- Image dimensions
- Color modes
- Corrupted images
- Duplicate images
- Class distribution
- Dataset balance

No corrupted images were found.

---

## 🔍 Data Preprocessing

The images were processed before training using the following pipeline:

```text
Input Image
     ↓
RGB Conversion
     ↓
Resize to 128 × 128
     ↓
Pixel Normalization
     ↓
Data Augmentation
     ↓
CNN / MobileNetV2
