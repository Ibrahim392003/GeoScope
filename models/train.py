import os
import argparse
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from architectures import get_model

class EuroSATDataset(Dataset):
    """Custom Dataset wrapper to read split definitions from Phase 2 CSV."""
    def __init__(self, csv_file, split='train', transform=None):
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['filepath'] if 'filepath' in row else row['path']
        if not os.path.exists(img_path):
            img_path = os.path.join('data/eurosat/2750', img_path)
            image = Image.open(img_path).convert('RGB')
            label = int(row['label'])

        if self.transform:
            image = self.transform(image)

        return image, label

def train_model(model_name, batch_size=32, epochs=5, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"-> Using device: {device}")

    # Transforms (Resizing to 224x224 and ImageNet Normalization)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load Datasets via Phase 2 splits
    train_dataset = EuroSATDataset('data/splits.csv', split='train', transform=train_transform)
    val_dataset = EuroSATDataset('data/splits.csv', split='val', transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Initialize model, loss, and optimizer
    model = get_model(model_name, num_classes=10, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    os.makedirs('models/checkpoints', exist_ok=True)
    best_val_acc = 0.0

    print(f"-> Starting training for {model_name} over {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation Loop
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # Save checkpoint if validation accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = f"models/checkpoints/{model_name}_best.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"   [+] Saved new best model checkpoint to {ckpt_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EuroScope models")
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'efficientnet', 'vit'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    train_model(args.model, args.batch_size, args.epochs, args.lr)