import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import os

# -------------------------------
# Load ResNet model (pretrained)
# -------------------------------
model = models.resnet50(pretrained=True)
model.eval()

# -------------------------------
# Load ImageNet labels (local file)
# -------------------------------
LABELS_FILE = "imagenet_classes.txt"

if not os.path.exists(LABELS_FILE):
    print("Downloading ImageNet labels...")
    import requests
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    r = requests.get(url)
    with open(LABELS_FILE, "w") as f:
        f.write(r.text)

with open(LABELS_FILE) as f:
    labels = [line.strip() for line in f.readlines()]

# -------------------------------
# Image Transform
# -------------------------------
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
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
    top_prob, top_class = torch.topk(probabilities, 3)

    print("\nTop Predictions:")
    for i in range(3):
        print(f"{labels[top_class[i]]}: {top_prob[i].item()*100:.2f}%")

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