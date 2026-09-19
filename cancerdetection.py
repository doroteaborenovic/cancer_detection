import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ==============================================================================
# 1. PARAMETRI I HARDVER
# ==============================================================================
DATASET_PATH = os.path.expanduser("~/data/multi_cancer")
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

START_EPOCH = 30
ADDITIONAL_EPOCHS = 70
TOTAL_EPOCHS = START_EPOCH + ADDITIONAL_EPOCHS  # Ukupno 100 epoha

FINE_TUNE_LR = 0.0002  # Manji Learning Rate za fino učenje

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n==========================================")
print(f" [HARDVER] Uređaj: {device}")
if torch.cuda.is_available():
    print(f" [HARDVER] GPU: {torch.cuda.get_device_name(0)}")
print(f"==========================================\n")

# ==============================================================================
# 2. UČITAVANJE PODATAKA (IDENTIČAN SPLIT)
# ==============================================================================
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

# Identičan seed (random_state=42) garantuje da je Test skup 100% isti!
train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
    filepaths, numeric_labels, test_size=0.20, random_state=42, stratify=numeric_labels
)
train_paths, val_paths, train_labels, val_labels = train_test_split(
    train_val_paths, train_val_labels, test_size=0.15, random_state=42, stratify=train_val_labels
)

print(f"Podaci uspešno raspoređeni: {len(train_paths)} Train | {len(val_paths)} Val | {len(test_paths)} Test")

# ==============================================================================
# 3. AUGMENTACIJA I DATA LOADERS
# ==============================================================================
train_transforms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

eval_transforms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

class CustomCancerDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

train_loader = DataLoader(CustomCancerDataset(train_paths, train_labels, train_transforms), batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(CustomCancerDataset(val_paths, val_labels, eval_transforms), batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(CustomCancerDataset(test_paths, test_labels, eval_transforms), batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# ==============================================================================
# 4. ARHITEKTURA MODELA & UČITAVANJE PRETHODNIH TEŽINA
# ==============================================================================
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

# Učitavanje modela iz prvih 30 epoha
checkpoint_file = "best_scratch_cancer_model.pth"
if os.path.exists(checkpoint_file):
    print(f"--> [CHECKPOINT] Učitavam prethodno naučene težine iz: {checkpoint_file}")
    model.load_state_dict(torch.load(checkpoint_file))
    print("--> [CHECKPOINT] Težine uspešno učitane! Nastavljamo trening do 100 epoha.\n")
else:
    print("--> [UPOZORENJE] Checkpoint nije pronađen, krećem od nule.\n")

# ==============================================================================
# 5. NASTAVAK TRENIRANJA (EPOHE 31 DO 100)
# ==============================================================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=FINE_TUNE_LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

best_val_acc = 0.9596  # Postavljamo prethodni rekord kao donji prag

print("================== NASTAVAK TRENINGA (31 - 100) ==================")
for epoch in range(ADDITIONAL_EPOCHS):
    current_epoch = START_EPOCH + epoch + 1
    model.train()
    running_loss, running_corrects = 0.0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data).item()

    train_loss = running_loss / len(train_paths)
    train_acc = running_corrects / len(train_paths)

    # Validacija
    model.eval()
    val_loss, val_corrects = 0.0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            val_corrects += torch.sum(preds == labels.data).item()

    val_loss /= len(val_paths)
    val_acc = val_corrects / len(val_paths)
    scheduler.step(val_loss)

    print(f"Epoha [{current_epoch:03d}/{TOTAL_EPOCHS}] | "
          f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc*100:.2f}% | "
          f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc*100:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_scratch_cancer_model.pth")
        print(f"  --> Oboren rekord! Sačuvan novi model (Val Acc: {best_val_acc*100:.2f}%)")

# ==============================================================================
# 6. FINALNO TESTIRANJE NA TEST SKUPU (100 EPOHA)
# ==============================================================================
print("\n================== KONAČNA EVALUACIJA (100 EPOHA) ==================")
model.load_state_dict(torch.load("best_scratch_cancer_model.pth"))
model.eval()

y_test_true, y_test_pred = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        y_test_true.extend(labels.numpy())
        y_test_pred.extend(preds.cpu().numpy())

print("\n--- DETALJAN REZULTAT PO KLASAMA (Classification Report) ---")
print(classification_report(y_test_true, y_test_pred, target_names=class_names, digits=4))