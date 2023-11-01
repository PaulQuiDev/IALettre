import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 62)  # 61 classes dans EMNIST (byclass)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)



# 1. Charger le modèle à partir du fichier "mon_modele.pth"
model_path = "mon_modele.pth"
device = torch.device("cpu")  # Vous pouvez changer "cpu" en "cuda" si vous avez un GPU compatible
model = Net().to(device)  # Assurez-vous de définir la classe Net similaire à celle de l'entraînement
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 2. Charger et prétraiter l'image d'entrée
def load_and_preprocess_image(image_path):
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                    transforms.Resize((28, 28)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    image = Image.open(image_path).convert("L")  # Ouvre l'image en niveau de gris
    image = transform(image).to(device)
    return image

image_path = "imageA.png"
image = load_and_preprocess_image(image_path)

# 3. Effectuer la prédiction
with torch.no_grad():
    output = model(image.unsqueeze(0))  # Utilisez unsqueeze pour ajouter une dimension de lot (batch dimension)
    output = abs(output)
    #output[0][26] = int(output[0][26]/2)
    predicted_class = output.argmax().item()

LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
          'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
          'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']
# 4. Afficher le résultat
print(f"Le caractère prédit est : {predicted_class}\n{output}\n {LABELS[int(predicted_class)]}")
