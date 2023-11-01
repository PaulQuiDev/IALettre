import matplotlib.pyplot as plt
from torchvision.datasets import EMNIST
import torchvision.transforms as transforms

# Chemin vers les données EMNIST
path_to_emnist_data = 'path_to_emnist_data'

# Transformation pour afficher les images
transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

# Charger le jeu de données EMNIST
emnist_train = EMNIST(root=path_to_emnist_data, split='byclass', train=True, download=True, transform=transform)

# Afficher une image de chaque classe
class_labels = emnist_train.classes
fig, axs = plt.subplots(1, len(class_labels), figsize=(20, 5))

for i, label in enumerate(class_labels):
    img, _ = emnist_train[i]
    axs[i].imshow(img.squeeze(0), cmap='gray')
    axs[i].set_title(label)

plt.show()
