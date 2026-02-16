import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import timm
from tqdm import tqdm
from sklearn.metrics import accuracy_score
BASE_DIR = r"C:\Users\Dell\Desktop\final\face"
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train")
TEST_IMG_DIR  = os.path.join(BASE_DIR, "test")

TRAIN_CSV = os.path.join(BASE_DIR, "train_labels.csv")
TEST_CSV  = os.path.join(BASE_DIR, "test_labels.csv")

IMG_SIZE = 224
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BEST_MODEL_PATH = os.path.join(BASE_DIR, "best_face_vit.pt")
def generate_csv(img_dir, csv_path):
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Image directory not found: {img_dir}")

    rows = []
    for class_name in sorted(os.listdir(img_dir)):
        class_folder = os.path.join(img_dir, class_name)
        if os.path.isdir(class_folder):
            for img_file in os.listdir(class_folder):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    rows.append({"image": img_file, "label": int(class_name)})
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"CSV saved at {csv_path}, total {len(df)} images.")
if not os.path.exists(TRAIN_CSV):
    generate_csv(TRAIN_IMG_DIR, TRAIN_CSV)
if not os.path.exists(TEST_CSV):
    generate_csv(TEST_IMG_DIR, TEST_CSV)
class FaceDataset(Dataset):
    def __init__(self, csv_file, img_root, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_root = img_root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["image"]
        label = int(row["label"])
        class_folder = str(label)
        img_path = os.path.join(self.img_root, class_folder, img_name)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Missing file: {img_path}")

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label - 1  
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])
train_dataset = FaceDataset(TRAIN_CSV, TRAIN_IMG_DIR, transform)
test_dataset  = FaceDataset(TEST_CSV, TEST_IMG_DIR, transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
model = timm.create_model("vit_base_patch16_224", pretrained=True)
model.head = nn.Linear(model.head.in_features, 7)  # 7 classes
model = model.to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
best_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_preds, train_labels = [], []
    train_loss = 0.0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch} Training"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * imgs.size(0)
        train_preds.extend(logits.argmax(1).cpu().numpy())
        train_labels.extend(labels.cpu().numpy())

    train_acc = accuracy_score(train_labels, train_preds)
    train_loss /= len(train_dataset)
    print(f"Epoch {epoch} Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    model.eval()
    test_preds, test_labels = [], []
    test_loss = 0.0
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc=f"Epoch {epoch} Testing"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = model(imgs)
            loss = criterion(logits, labels)
            test_loss += loss.item() * imgs.size(0)

            test_preds.extend(logits.argmax(1).cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    test_acc = accuracy_score(test_labels, test_preds)
    test_loss /= len(test_dataset)
    print(f"Epoch {epoch} Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"Best model saved at epoch {epoch} with accuracy {best_acc:.4f}\n")
