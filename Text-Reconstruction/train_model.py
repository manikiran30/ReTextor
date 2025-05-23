import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
from PIL import Image

class HandwritingCNN(nn.Module):
    def __init__(self, num_classes):
        super(HandwritingCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)

    def forward(self, x):
        x = self.pool(torch.relu(self.batch_norm1(self.conv1(x))))
        x = self.pool(torch.relu(self.batch_norm2(self.conv2(x))))
        x = self.pool(torch.relu(self.batch_norm3(self.conv3(x))))
        x = x.view(-1, 128 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class HandwritingDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        self.char_to_idx = {}
        self.idx_to_char = {}
        
        # Create character mapping
        chars = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for idx, char in enumerate(chars):
            self.char_to_idx[char] = idx
            self.idx_to_char[idx] = char
            
        # Load all images and labels
        for char in os.listdir(data_dir):
            char_dir = os.path.join(data_dir, char)
            if os.path.isdir(char_dir):
                for img_file in os.listdir(char_dir):
                    if img_file.endswith('.png') or img_file.endswith('.jpg'):
                        img_path = os.path.join(char_dir, img_file)
                        self.samples.append((img_path, self.char_to_idx[char]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], '
                      f'Loss: {running_loss/100:.4f}, '
                      f'Accuracy: {100 * correct/total:.2f}%')
                running_loss = 0.0
                correct = 0
                total = 0

def create_character_dataset():
    """
    Create a directory structure for storing character samples:
    data/
        a/
        b/
        c/
        ...
        0/
        1/
        ...
    """
    base_dir = "training_data"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        
    chars = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for char in chars:
        char_dir = os.path.join(base_dir, char)
        if not os.path.exists(char_dir):
            os.makedirs(char_dir)
    
    print(f"Created dataset directory structure in {base_dir}")
    print("Please add character images to their respective folders.")
    print("Each image should contain a single character.")
    return base_dir

def main():
    # Create dataset structure
    data_dir = create_character_dataset()
    
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Create dataset and dataloader
    dataset = HandwritingDataset(data_dir, transform=transform)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HandwritingCNN(num_classes=62).to(device)  # 62 classes (10 digits + 26*2 letters)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    train_model(model, train_loader, criterion, optimizer, device)
    
    # Save the model
    torch.save(model.state_dict(), 'handwriting_model.pth')
    print("Model saved successfully!")

if __name__ == "__main__":
    main()
