from datetime import datetime
from argparse import ArgumentParser, Namespace
import torchvision
import torch
from torchvision import models
from torchvision import datasets
from torch import nn
import torch.distributed as dist
from tqdm import tqdm
from model import ConvNet
from data import MNIST


def parse_args() -> Namespace:
    parser = ArgumentParser("训练网络的参数")
    parser.add_argument("--gpuid", type=int, default=0, help="哟啊使用那一块gpu")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--data-dir", type=str, default="../data")
    args = parser.parse_args()
    return args


def train(net, args, train_loader, device, criterion, optimizer) -> torch.Tensor:
    net.train()
    start = datetime.now()
    train_ds_len = len(train_loader)
    for i in range(args.epochs):
        loss_epoch = torch.tensor(0, dtype=torch.float)
        bar = tqdm(total=train_ds_len)
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            predictions = net(images)
            loss = criterion(predictions, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch += loss.cpu()
            bar.update()
        bar.close()
        print(f"epoch: {i}, loss: {loss_epoch.item() / train_ds_len}")
    end = datetime.now()
    print(f"训练消耗时间: {start - end}")


def test(net, val_loader, device, criterion):
    net.eval()
    loss_epoch = torch.tensor(0, dtype=torch.float)
    val_ds_len = len(val_loader)
    num_right_predictions = 0
    num_total_images = 0
    bar = tqdm(total=val_ds_len)
    for images, labels in val_loader:
        with torch.no_grad():
            images = images.to(device)
            labels = labels.to(device)
            predictions = net(images)
        loss = criterion(predictions, labels)
        predictions = torch.argmax(predictions, dim=1)
        predictions = torch.squeeze(predictions)
        num_right_predictions += (predictions == labels).sum().cpu().item()
        num_total_images += images.shape[0]

        loss_epoch += loss.cpu()
        bar.update()
    bar.close()
    print(
        f"val loss: {loss_epoch.item() / val_ds_len}, val accuracy: {(num_right_predictions / num_total_images) * 100:.2f}%")


def main(args) -> None:
    model = ConvNet(10)
    device = torch.device("cuda", args.gpuid)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)

    train_loader = MNIST(args.data_dir, args.batch_size, True)
    val_loader = MNIST(args.data_dir, args.batch_size, False)
    train(model, args, train_loader, device, criterion, optimizer)
    test(model, val_loader, device, criterion)


if __name__ == '__main__':
    args = parse_args()
    main(args)
