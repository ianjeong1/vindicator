import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from transformers import ViTModel, ViTFeatureExtractor
from PIL import Image
from tqdm import tqdm

# --- Dataset ---
class ViTDataset(Dataset):
    def __init__(self, root_dir, feature_extractor):
        self.data = ImageFolder(root=root_dir)
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        return {k: v.squeeze(0) for k, v in inputs.items()}, label

# --- Model ---
class ViTFakeDetector(nn.Module):
    def __init__(self, vit_model, hidden_size=768, num_classes=2):
        super().__init__()
        self.vit = vit_model
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, pixel_values):
        with torch.no_grad():  # Freeze ViT
            outputs = self.vit(pixel_values=pixel_values)
            cls_token = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        return self.classifier(cls_token)

# --- Training ---
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
    vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
    model = ViTFakeDetector(vit_model).to(device)

    dataset = ViTDataset("dataset", feature_extractor)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=1e-4)

    model.train()
    for epoch in range(5):
        total_loss = 0
        correct = 0
        total = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            inputs, labels = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs["pixel_values"])
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = correct / total
        print(f"Epoch {epoch+1} - Loss: {total_loss:.4f} - Accuracy: {acc:.4f}")

if __name__ == "__main__":
    train()