import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import EMNIST
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim import AdamW

# 1. Charger le jeu de données EMNIST
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
device = torch.device("cpu") # marche pas en cuda car doit h 24 transferer fichier 
criterion = nn.CrossEntropyLoss().to(device)  

emnist_train = EMNIST(root='path_to_emnist_data', split='byclass', train=True, download=True, transform=transform)
emnist_test = EMNIST(root='path_to_emnist_data', split='byclass', train=False, download=True, transform=transform)



batch_size = 32
train_loader = DataLoader(emnist_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(emnist_test, batch_size=batch_size)

# 2. Définir le modèle
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 62)  # 62 classes dans EMNIST (byclass)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# 3. Initialiser le modèle, la fonction de perte (loss) et l'optimiseur
model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9) # je le trouve plus stable que adam

# 4. Entraîner le modèle
n_epochs = 6

for epoch in range(n_epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)  # Déplacez les données d'entrée et les étiquettes sur le GPU
        optimizer.zero_grad()
        output = model(data)  
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 200 == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}")

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy}%")

model_path = "mon_modele.pth"  # Spécifiez le chemin où vous souhaitez enregistrer votre modèle
torch.save(model.state_dict(), model_path)


# 86.35437531700524%