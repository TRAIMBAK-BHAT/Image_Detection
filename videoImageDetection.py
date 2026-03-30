import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import os
import cv2
import numpy as np

# -------------------------------
# Load ResNet model (pretrained)
# -------------------------------
model = models.resnet50(pretrained=True)
model.eval()

# -------------------------------
# Load ImageNet labels
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
# Frame Classification Function
# -------------------------------
def classify_frame(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)

    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    top_prob, top_class = torch.topk(probabilities, 1)

    label = labels[top_class.item()]
    confidence = top_prob.item() * 100

    return label, confidence

# -------------------------------
# Video Classification Function
# -------------------------------
def classify_video(video_path):
    cap = cv2.VideoCapture(video_path)

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Classify every 15th frame (reduce computation)
        if frame_count % 15 == 0:
            label, confidence = classify_frame(frame)
            text = f"{label} ({confidence:.2f}%)"
        frame_count += 1

        # Overlay prediction text
        cv2.putText(frame, text,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)

        cv2.imshow("Video Classification", frame)

        # Press q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# -------------------------------
# File Picker GUI
# -------------------------------
root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename(
    title="Select a Video",
    filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
)

if file_path:
    classify_video(file_path)
else:
    print("No file selected.")