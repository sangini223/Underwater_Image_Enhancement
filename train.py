import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
from PIL import Image
from model import PhysicalNN  # Import your updated model

# Custom Dataset Class for EUVP Dataset
class EUVP_Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_pairs = []

        # Load image pairs
        trainA_dir = os.path.join(root_dir, 'underwater_dark', 'trainA')
        trainB_dir = os.path.join(root_dir, 'underwater_dark', 'trainB')

        for img_name in os.listdir(trainA_dir):
            if img_name.endswith('.jpg') or img_name.endswith('.png'):
                self.image_pairs.append((os.path.join(trainA_dir, img_name), os.path.join(trainB_dir, img_name)))

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        imgA_path, imgB_path = self.image_pairs[idx]
        imgA = Image.open(imgA_path).convert("RGB")
        imgB = Image.open(imgB_path).convert("RGB")

        if self.transform:
            imgA = self.transform(imgA)
            imgB = self.transform(imgB)

        return imgA, imgB

# Transformations: Normalize input to [-1, 1] for Tanh compatibility
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])

# Initialize dataset and dataloader
dataset = EUVP_Dataset(root_dir='C:/Users/sangi/Desktop/Underwater_image_enhancement/EUVP', transform=transform)

# Limit dataset size for faster training
limited_dataset = Subset(dataset, range(min(100, len(dataset))))
dataloader = DataLoader(limited_dataset, batch_size=1, shuffle=True)

# Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PhysicalNN().to(device)
criterion = nn.L1Loss()  # L1 loss tends to give sharper images
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
num_epochs = 5  # You can increase this as needed
print_interval = 10  # Print updates every 10 batches

for epoch in range(num_epochs):
    print(f'Starting Epoch {epoch + 1}/{num_epochs}')
    for i, (imgA, imgB) in enumerate(dataloader):
        imgA, imgB = imgA.to(device), imgB.to(device)

        optimizer.zero_grad()
        outputs = model(imgA)
        loss = criterion(outputs, imgB)
        loss.backward()
        optimizer.step()

        if (i + 1) % print_interval == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}')

# Save the trained model
torch.save(model.state_dict(), 'model.pth')
print("Model saved as model.pth")
