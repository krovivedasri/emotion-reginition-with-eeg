import os
import numpy as np
import torch
import torch.nn as nn
import timm
import cv2
import sounddevice as sd
import librosa
from PIL import Image
import pyttsx3
CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "face_model_path": "face/best_face_vit.pt",
    "audio_model_path": "TESS/tess_mfcc_model.pt",
    "eeg_model_path": "eeg/eeg_mlp_model.pt",
    "eeg_live_path": "eeg/live_eeg.npy",
    "audio_seconds": 3,
    "audio_sample_rate": 16000,
    "face_img_size": 224
}
EMOTIONS = ["Surprise","Fear","Disgust","Happy","Sad","Angry","Neutral"]
EEG_EMO = ["Positive","Neutral","Negative"]
class EEG_MLP(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.fc1 = nn.Linear(inp,128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,out)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.3)
    def forward(self,x):
        x = self.drop(self.relu(self.fc1(x)))
        x = self.drop(self.relu(self.fc2(x)))
        return self.fc3(x)
class MFCC_MLP(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.fc1 = nn.Linear(inp,128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,out)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.3)
    def forward(self,x):
        x = self.drop(self.relu(self.fc1(x)))
        x = self.drop(self.relu(self.fc2(x)))
        return self.fc3(x)
def load_face_model(path, device):
    state = torch.load(path, map_location=device)
    tmp = timm.create_model("vit_base_patch16_224", pretrained=False)
    model = timm.create_model("vit_base_patch16_224", pretrained=False)
    model.head = nn.Linear(tmp.head.in_features, state["head.weight"].shape[0])
    model.load_state_dict(state)
    model.to(device).eval()
    return model
def load_audio_model(path, device):
    state = torch.load(path, map_location=device)
    inp = state["fc1.weight"].shape[1]
    out = state["fc3.weight"].shape[0]
    model = MFCC_MLP(inp,out)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, inp
def load_eeg_model(path, device):
    state = torch.load(path, map_location=device)
    inp = state["fc1.weight"].shape[1]
    out = state["fc3.weight"].shape[0]
    model = EEG_MLP(inp,out)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, inp
def detect_face(frame, size):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cv2.CascadeClassifier(
        cv2.data.haarcascades+"haarcascade_frontalface_default.xml"
    ).detectMultiScale(gray,1.3,5)
    if len(faces)==0: return None
    x,y,w,h = max(faces, key=lambda f:f[2]*f[3])
    face = frame[y:y+h, x:x+w]
    img = Image.fromarray(cv2.cvtColor(face,cv2.COLOR_BGR2RGB)).resize((size,size))
    arr = (np.array(img)/255.0 - 0.5)/0.5
    arr = np.transpose(arr,(2,0,1))
    return torch.tensor(arr,dtype=torch.float32).unsqueeze(0)
def run_face(model, frame, device):
    x = detect_face(frame, CONFIG["face_img_size"])
    if x is None: return None
    with torch.no_grad():
        out = model(x.to(device))
        return torch.softmax(out,1).cpu().numpy()[0]
def record_audio(sec, sr):
    print("\n[Audio] Speak now...")
    audio = sd.rec(int(sec*sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    return audio.squeeze()
def run_audio(model, inp, device):
    audio = record_audio(CONFIG["audio_seconds"], CONFIG["audio_sample_rate"])
    mfcc = librosa.feature.mfcc(y=audio, sr=CONFIG["audio_sample_rate"], n_mfcc=inp)
    mfcc = np.mean(mfcc.T, axis=0)
    x = torch.tensor(mfcc,dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        return torch.softmax(out,1).cpu().numpy()[0]
def run_eeg(model, inp, device):
    if not os.path.exists(CONFIG["eeg_live_path"]): return None
    vec = np.load(CONFIG["eeg_live_path"]).reshape(-1)
    if len(vec)!=inp: return None
    x = torch.tensor(vec,dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        return torch.softmax(out,1).cpu().numpy()[0]
AUDIO_MAP = {
    0:"Angry",1:"Angry",
    2:"Disgust",3:"Disgust",
    4:"Fear",5:"Fear",
    6:"Happy",7:"Happy",
    8:"Neutral",9:"Neutral",
    10:"Pleasant",11:"Pleasant",
    12:"Sad",13:"Sad"
}
def speak(emotion):
    engine = pyttsx3.init()
    msg = f"You seem {emotion}. Stay positive and strong!"
    print("\n[Message]:", msg)
    engine.say(msg)
    engine.runAndWait()
def main():
    print("Using device:", CONFIG["device"])
    print("\nLoading models...")
    face_model = load_face_model(CONFIG["face_model_path"], CONFIG["device"])
    audio_model, a_in = load_audio_model(CONFIG["audio_model_path"], CONFIG["device"])
    eeg_model, e_in = load_eeg_model(CONFIG["eeg_model_path"], CONFIG["device"])
    print("Models loaded.\n")
    cap = cv2.VideoCapture(0)
    print("[Face] Press SPACE to capture, ESC to exit.")
    while True:
        ret, frame = cap.read()
        if not ret: continue
        cv2.imshow("Webcam", frame)
        key = cv2.waitKey(1)
        if key == 27: return
        if key == 32:
            frame_cap = frame.copy()
            break
    cap.release()
    cv2.destroyAllWindows()

    # -------- FACE --------
    f = run_face(face_model, frame_cap, CONFIG["device"])
    face_em = EMOTIONS[np.argmax(f)] if f is not None else "None"
    print("[FACE EMOTION] →", face_em)

    # -------- AUDIO --------
    a = run_audio(audio_model, a_in, CONFIG["device"])
    audio_em = AUDIO_MAP[int(np.argmax(a))]
    print("[AUDIO EMOTION] →", audio_em)

    # -------- EEG --------
    e = run_eeg(eeg_model, e_in, CONFIG["device"])
    eeg_em = EEG_EMO[np.argmax(e)] if e is not None else "None"
    print("[EEG EMOTION] →", eeg_em)

    # -------- FINAL (PURE LOGIC) --------
    if face_em == audio_em:
        final_emotion = face_em
    else:
        final_emotion = face_em   # face priority (PURE, NO FIX)

    print("\nFINAL EMOTION:", final_emotion)
    speak(final_emotion)

if __name__ == "__main__":
    main()
