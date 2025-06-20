import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    random_idx = torch.randint(len(training_data), size=(1,))
    label, img = training_data[random_idx]
    plt.subplots(rows, cols, i)
    plt.title(labels_map[label])
    plt.imshow(img)

plt.show()

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)


train_features, train_labels = next(iter(train_dataloader))
# plt.imshow(train_features[])

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])


ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transform,
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

