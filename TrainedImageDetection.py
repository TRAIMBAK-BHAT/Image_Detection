import torch
import torchvision.transforms as transforms
from torchvision import models, datasets
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import os

# -------------------------------
# Path to your dataset root
# -------------------------------
#DATASET_PATH = "yl_images/train"   # change if needed
DATASET_PATH = r"C:\Users\Admin\Documents\object identifier\yl_images\train"
# Load class names automatically
# -------------------------------
dataset = datasets.ImageFolder(DATASET_PATH)
labels = dataset.classes
num_classes = len(labels)

print("Detected Classes:", labels)

# -------------------------------
# Load Modified ResNet Model
# -------------------------------
model = models.resnet50(pretrained=False)

# Replace final layer
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# Load trained weights
model.load_state_dict(torch.load("machinery_model.pth", map_location=torch.device('cpu')))
model.eval()

# -------------------------------
# Image Transform (must match training)
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------------
# Image Classification Function
# -------------------------------
def classify_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)

    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    top_prob, top_class = torch.max(probabilities, dim=0)

    print("\nPrediction:")
    print(f"Class: {labels[top_class.item()]}")
    print(f"Confidence: {top_prob.item() * 100:.2f}%")

    image.show()

# -------------------------------
# File Picker GUI
# -------------------------------
root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename(
    title="Select an Image",
    filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
)

if file_path:
    classify_image(file_path)
else:
    print("No file selected.")