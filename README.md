# Image and Video Inspection using Deep Learning

This project implements an image and video inspection system using a pretrained **ResNet-50** model.  
It performs real-time classification on video frames and displays predictions with confidence scores.

The system leverages pretrained ImageNet weights and can be extended with a company-specific trained dataset to improve domain-specific detection accuracy.

---

## 🔍 Features

- 📸 Image classification using pretrained ResNet-50
- 🎥 Video frame classification (real-time)
- ⚡ Efficient processing by sampling frames (every 15th frame)
- 🧠 Uses ImageNet pretrained weights
- 🖥️ Simple GUI file picker using Tkinter
- 📊 Displays prediction labels with confidence scores

---

## 🧠 Model Details

- Architecture: **ResNet-50**
- Framework: PyTorch
- Pretrained on: ImageNet dataset
- Output: Top predicted class with confidence score

---

## ⚙️ Tech Stack

- Python
- PyTorch
- Torchvision
- OpenCV
- NumPy
- PIL (Python Imaging Library)
- Tkinter (for file selection)

---

## 📁 How It Works

1. Load pretrained ResNet-50 model
2. Apply image transformations (resize, crop, normalize)
3. Extract frames from video using OpenCV
4. Classify every 15th frame
5. Display predictions on video in real-time

---

## ▶️ Usage

### 1. Install dependencies
```bash
pip install torch torchvision opencv-python pillow numpy

python your_script_name.py
