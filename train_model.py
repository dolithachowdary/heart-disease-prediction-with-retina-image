import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

# === Config ===
data_dir = "dataset"
model_save_path = "heart_disease_model.pth"
img_size = 224
batch_size = 16
num_epochs = 10
learning_rate = 1e-4

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Transformations ===
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# === Dataset & Dataloader ===
train_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
class_names = train_dataset.classes
print(f"Classes: {class_names}")  # ['heart_disease', 'no_heart_disease']

# === Model ===
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 2)  # Binary classification
model = model.to(device)

# === Loss & Optimizer ===
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# === Training Loop ===
loss_list = []
acc_list = []

for epoch in range(num_epochs):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    loss_list.append(avg_loss)
    acc_list.append(accuracy)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

# === Save Model ===
torch.save(model.state_dict(), model_save_path)
print(f"✅ Model saved to: {model_save_path}")

# === Plot Loss & Accuracy ===
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(loss_list, label="Loss")
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(acc_list, label="Accuracy", color="green")
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.grid(True)

plt.tight_layout()
plt.savefig("training_plot.png")
plt.show()
