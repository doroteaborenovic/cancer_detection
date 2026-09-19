import os
import glob
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg') # Režim za servere bez ekrana
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# 1. HARDVER I PUTANJE
DATASET_PATH = os.path.expanduser("~/data/multi_cancer")
IMG_SIZE = (128, 128)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. DATASET I MAPIRANJE KLASA
valid_extensions = {".jpg", ".jpeg", ".png"}
filepaths, labels_raw = [], []

for root, _, files in os.walk(DATASET_PATH):
    folder_name = os.path.basename(root)
    for file in files:
        if os.path.splitext(file)[1].lower() in valid_extensions:
            filepaths.append(os.path.join(root, file))
            labels_raw.append(folder_name)

class_names = sorted(list(set(labels_raw)))
class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
numeric_labels = [class_to_idx[name] for name in labels_raw]

# Strogo isti test split kao na treningu
_, test_paths, _, test_labels = train_test_split(
    filepaths, numeric_labels, test_size=0.20, random_state=42, stratify=numeric_labels
)

# 3. DEFINICIJA MODELA
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.25):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_rate)
        )
    def forward(self, x):
        return self.block(x)

class ScratchTumorCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32, dropout_rate=0.2),
            ConvBlock(32, 64, dropout_rate=0.25),
            ConvBlock(64, 128, dropout_rate=0.3),
            ConvBlock(128, 256, dropout_rate=0.35)
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

model = ScratchTumorCNN(num_classes=len(class_names)).to(device)
model.load_state_dict(torch.load("best_scratch_cancer_model.pth", map_location=device))
model.eval()
print("--> Model uspešno učitan!")

# 4. GRAD-CAM ENGINE
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Postavljanje "kuka" (hooks) na poslednji konvolucioni sloj
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()

        score = output[0, target_class]
        score.backward()

        # Global Average Pooling nad gradijentima
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        cam = torch.clamp(cam, min=0) # ReLU
        
        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (IMG_SIZE[0], IMG_SIZE[1]))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, target_class

# Ciljni sloj je drugi Conv2d u poslednjem bloku
target_layer = model.features[3].block[3]
grad_cam = GradCAM(model, target_layer)

transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 5. GENERISANJE GRAD-CAM GALERIJE ZA 6 RAZLIČITIH TUMORA
print("--> Pravim Grad-CAM analitičku galeriju...")
sample_indices = np.linspace(0, len(test_paths)-1, 6, dtype=int)
fig, axes = plt.subplots(6, 3, figsize=(12, 20))
plt.subplots_adjust(hspace=0.4)

for i, idx in enumerate(sample_indices):
    img_path = test_paths[idx]
    true_label = class_names[test_labels[idx]]

    raw_img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
    img_tensor = transform(raw_img).unsqueeze(0).to(device)

    cam, pred_idx = grad_cam.generate(img_tensor)
    pred_label = class_names[pred_idx]

    # Kreiranje Heatmap-a
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Preklapanje (Overlay) originalne slike i toplotne mape
    raw_np = np.array(raw_img)
    overlay = np.uint8(0.6 * raw_np + 0.4 * heatmap)

    # 1. Kolona: Original
    axes[i, 0].imshow(raw_img)
    axes[i, 0].set_title(f"Stvarna klasa: {true_label}", fontsize=10, fontweight='bold')
    axes[i, 0].axis('off')

    # 2. Kolona: Grad-CAM Heatmap
    axes[i, 1].imshow(heatmap)
    axes[i, 1].set_title("AI Heatmap pažnje", fontsize=10)
    axes[i, 1].axis('off')

    # 3. Kolona: Preklopljeno
    axes[i, 2].imshow(overlay)
    status = "Tačno" if true_label == pred_label else "Netačno"
    color = "green" if true_label == pred_label else "red"
    axes[i, 2].set_title(f"Predikcija: {pred_label} ({status})", fontsize=10, color=color, fontweight='bold')
    axes[i, 2].axis('off')

plt.savefig("gradcam_gallery.png", dpi=300, bbox_inches='tight')
print(" Sačuvana slika: gradcam_gallery.png")

# 6. GENERISANJE CONFUSION MATRIX (26x26)
print("--> Računam Matricu Konfuzije za test skup...")
y_true, y_pred = [], []
with torch.no_grad():
    for p, l in zip(test_paths[:5000], test_labels[:5000]): # Na poduzorku od 5000 slika radi brzine
        img = Image.open(p).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)
        out = model(tensor)
        y_pred.append(torch.argmax(out, dim=1).item())
        y_true.append(l)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(18, 15))
sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix (26 Klasa Tumora)", fontsize=16, fontweight='bold')
plt.xlabel("Predviđena klasa", fontsize=12)
plt.ylabel("Stvarna klasa", fontsize=12)
plt.xticks(rotation=90)
plt.yticks(rotation=0)

plt.savefig("confusion_matrix_26classes.png", dpi=300, bbox_inches='tight')
print(" Sačuvana slika: confusion_matrix_26classes.png")
print("--> Gotovo! Obe slike su spremne za GitHub i rad.")
