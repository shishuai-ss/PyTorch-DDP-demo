import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch import distributed as dist

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def MNIST(data_dir: str, batch_size: int, is_training: bool = True) -> DataLoader:
    ds = datasets.MNIST(data_dir, transform=transforms.ToTensor(), train=is_training, download=True)
    return DataLoader(ds, batch_size, shuffle=is_training)


def CIFAR10(data_dir: str, batch_size: int, is_training: bool = True) -> DataLoader:
    tran_list = []
    if is_training:
        tran_list.extend([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])
    else:
        tran_list.extend([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])
    trans = transforms.Compose(tran_list)
    dataset = datasets.CIFAR10(data_dir, is_training, trans, download=True)
    if is_training:
        sampler = DistributedSampler(dataset, shuffle=True)
    else:
        world_size = dist.get_world_size()
        current_rank = dist.get_rank()
        indices = torch.arange(current_rank, len(dataset), world_size)
        sampler = SubsetRandomSampler(indices)
    return DataLoader(dataset, batch_size, num_workers=2, sampler=sampler)


if __name__ == '__main__':
    CIFAR10("../data", 32, True)
