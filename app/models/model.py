import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mediapipe import solutions
import cv2
from tqdm import tqdm

# Paths and Configurations
DATASET_JSON = "/home/sathvik/Documents/Sign-Language-translator-API/app/dataset/WLASL_v0.3.json"  # Path to the dataset JSON file
VIDEOS_DIR = "videos/"            # Directory containing the videos
SAVE_MODEL_PATH = "best_model.pth"  # Path to save the best model
NUM_CLASSES = 2000                # Number of classes in WLASL
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Dataset Metadata
def load_dataset(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Ensure compatibility with both dictionary and list structures
    if isinstance(data, dict) and "root" in data:
        data = data["root"]  # Access "root" if the JSON is structured as a dictionary

    dataset = []
    for entry in data:  # Now `data` is a list of entries
        gloss = entry.get("gloss")
        for instance in entry.get("instances", []):
            dataset.append({
                "gloss": gloss,
                "video_id": instance["video_id"],
                "split": instance["split"],
                "url": instance["url"]
            })
    return dataset


# Extract Keypoints from Video
def extract_keypoints(video_path):
    mp_hands = solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
    keypoints = []

    # Path for saving keypoints
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    keypoints_path = os.path.join(VIDEOS_DIR, f"{video_id}.npy")

    # Create directory if it doesn't exist
    os.makedirs(VIDEOS_DIR, exist_ok=True)

    # If keypoints already exist, load them
    if os.path.exists(keypoints_path):
        return np.load(keypoints_path)

    # Process the video and extract keypoints
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        frame_keypoints = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                frame_keypoints.append([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

        keypoints.append(frame_keypoints)
    cap.release()

    # Save the keypoints for future use
    np.save(keypoints_path, keypoints)
    return np.array(keypoints)



# Dataset Class
class WLASLDataset(Dataset):
    def __init__(self, data, split):
        self.data = [item for item in data if item["split"] == split and self.has_valid_keypoints(item)]
        self.gloss_to_idx = {gloss: idx for idx, gloss in enumerate(sorted(set([item["gloss"] for item in data])))}
        self.idx_to_gloss = {idx: gloss for gloss, idx in self.gloss_to_idx.items()}

    def has_valid_keypoints(self, item):
        video_path = os.path.join(VIDEOS_DIR, f"{item['video_id']}.mp4")
        keypoints = extract_keypoints(video_path)
        return keypoints.size > 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        video_path = os.path.join(VIDEOS_DIR, f"{item['video_id']}.mp4")
        keypoints = extract_keypoints(video_path)
        label = self.gloss_to_idx[item["gloss"]]
        return torch.tensor(keypoints.mean(axis=0).flatten(), dtype=torch.float32), torch.tensor(label, dtype=torch.long)



# Model Definition
class SimpleModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

# Training Function
def train_model(model, train_loader, val_loader, criterion, optimizer):
    best_accuracy = 0
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        num_batches = len(train_loader)
        print(f"Epoch {epoch + 1}/{EPOCHS}")

        # Training Loop with Progress Bar
        for batch_idx, (keypoints, labels) in enumerate(tqdm(train_loader, desc="Training")):
            keypoints, labels = keypoints.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(keypoints)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (batch_idx + 1) % 10 == 0:  # Print status every 10 batches
                print(
                    f"  Batch {batch_idx + 1}/{num_batches}, Loss: {loss.item():.4f}"
                )

        avg_loss = running_loss / num_batches
        print(f"  Training Loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for keypoints, labels in tqdm(val_loader, desc="Validation"):
                keypoints, labels = keypoints.to(DEVICE), labels.to(DEVICE)
                outputs = model(keypoints)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        print(f"  Validation Accuracy: {accuracy:.4f}")

        # Save the best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), SAVE_MODEL_PATH)
            print("  Best model saved.")


# Main Function
if __name__ == "__main__":
    dataset = load_dataset(DATASET_JSON)

    # Split the data
    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
    train_dataset = WLASLDataset(train_data, split="train")
    val_dataset = WLASLDataset(val_data, split="val")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model and Training
    input_dim = 21 * 3  # 21 keypoints, each with x, y, z
    model = SimpleModel(input_dim=input_dim, num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_model(model, train_loader, val_loader, criterion, optimizer)

# Integrating with TranslationModel
"""
class TranslationModel:
    def __init__(self, model_path):
        self.model = SimpleModel(input_dim=21 * 3, num_classes=2000).to(DEVICE)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def predict(self, keypoints):
        input_tensor = torch.tensor(keypoints, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = self.model(input_tensor)
            return output.argmax().item()
"""
