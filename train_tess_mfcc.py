import os
import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import joblib
DATASET_PATH = r"C:\Users\Dell\Desktop\final\TESS"
SAMPLE_RATE = 22050
MFCC_FEATURES = 40
BATCH_SIZE = 32
EPOCHS = 4
LR = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
audio_paths, labels = [], []

for root, _, files in os.walk(DATASET_PATH):
    for f in files:
        if f.endswith(".wav"):
            audio_paths.append(os.path.join(root, f))
            labels.append(os.path.basename(root))

le = LabelEncoder()
labels = le.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(
    audio_paths, labels, test_size=0.2, stratify=labels, random_state=42
)
def extract_mfcc(path):
    y, sr = librosa.load(path, sr=SAMPLE_RATE)
    y = librosa.effects.preemphasis(y)
    y = y + 0.005 * np.random.randn(len(y))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=MFCC_FEATURES)
    return np.mean(mfcc.T, axis=0)

X_train_feat = np.array([extract_mfcc(p) for p in tqdm(X_train)])
X_test_feat = np.array([extract_mfcc(p) for p in tqdm(X_test)])

scaler = StandardScaler()
X_train_feat = scaler.fit_transform(X_train_feat)
X_test_feat = scaler.transform(X_test_feat)
class VoiceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

train_loader = DataLoader(VoiceDataset(X_train_feat, y_train), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(VoiceDataset(X_test_feat, y_test), batch_size=BATCH_SIZE)
class MFCC_MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        x = self.relu(self.fc2(x))
        return self.fc3(x)

model = MFCC_MLP(MFCC_FEATURES, len(le.classes_)).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
for e in range(EPOCHS):
    model.train()
    preds, trues = [], []
    for X, y in tqdm(train_loader):
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        preds.extend(out.argmax(1).cpu().numpy())
        trues.extend(y.cpu().numpy())

    print(f"Epoch {e+1}/{EPOCHS} Accuracy: {accuracy_score(trues, preds):.4f}")
model.eval()
preds, trues = [], []
with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        out = model(X)
        preds.extend(out.argmax(1).cpu().numpy())
        trues.extend(y.cpu().numpy())

print("\nFINAL TEST ACCURACY:", accuracy_score(trues, preds))
print(classification_report(trues, preds, target_names=le.classes_))
os.makedirs("TESS", exist_ok=True)
torch.save(model.state_dict(), "TESS/tess_mfcc_model.pt")
joblib.dump(scaler, "TESS/mfcc_scaler.save")
joblib.dump(le.classes_, "TESS/voice_labels.pkl")  

print("Voice model + assets saved successfully")