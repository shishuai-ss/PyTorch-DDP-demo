import torchvision
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader


def MNIST(data_dir: str, batch_size: int, is_training: bool = True) -> DataLoader:
    ds = datasets.MNIST(data_dir, transform=transforms.ToTensor(), train=is_training)
    return DataLoader(ds, batch_size, shuffle=is_training)
