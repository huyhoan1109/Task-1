from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms.transforms import ToTensor

batch_size = 32
train_data = datasets.KMNIST(
    root = 'KMIST',
    train = True,
    download = True,
    transform = ToTensor()
)
test_data = datasets.KMNIST(
    root = 'KMIST',
    train = False,
    transform = ToTensor()
)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)