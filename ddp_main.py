from argparse import ArgumentParser, Namespace
import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
from model import ResNet34
from data import CIFAR10
from torch.cuda.amp import GradScaler, autocast
from torch import optim
from torch.utils.data import DataLoader
from pathlib import Path
import logging
from timm.utils import AverageMeter, accuracy
import os
from utils import reduce_tensor


def parse_args() -> Namespace:
    parser = ArgumentParser("训练网络的参数")
    # parser.add_argument("--gpuid", type=int, default=0, help="使用那一块gpu")
    parser.add_argument("--batch-size", type=int, default=120)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--data-dir", type=str, default="../data")
    # local_rank：这里指的是当前进程在当前机器中的序号，注意和在全部进程中序号的区别。在ENV模式中，
    # 这个参数是必须的，由启动脚本自动划分，不需要手动指定。要善用local_rank来分配GPU_ID。
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--use_mix_precision", action="store_true")
    parser.add_argument("--val-epoch", type=int, default=3)
    parser.add_argument("--save-dir", type=str, default="./weights")
    args = parser.parse_args()
    if args.local_rank == -1:
        args.local_rank = int(os.environ["LOCAL_RANK"])
    return args


def train(model: nn.Module,
          train_loader: DataLoader,
          val_loader: DataLoader,
          optimizer: optim.Optimizer,
          lr_scheduler: torch.optim.lr_scheduler.MultiStepLR,
          criterion: nn.Module,
          device: torch.device,
          scaler: GradScaler,
          args: Namespace
          ):
    val_losses = []
    for epoch in range(args.epochs):
        train_loader.sampler.set_epoch(epoch)
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler, args)
        if dist.get_rank() == 0:
            logging.info(f"rank 0 epoch: {epoch + 1} train loss: {train_loss}")
        if epoch % args.val_epoch == 0:
            val_loss, val_accuracy = val_epoch(model, val_loader, criterion, device)
            if dist.get_rank() == 0:
                logging.info(f"epoch: {epoch + 1} val loss: {val_loss}, val accuracy: {val_accuracy: .2f}")
            if dist.get_rank() == 0:
                if len(val_losses) == 0:
                    checkpoint = Path(args.save_dir) / f"resnet34-{epoch + 1}-{val_loss}.pth"
                    torch.save(model.module.state_dict(), checkpoint)
                elif min(val_losses) > val_loss:
                    checkpoint = Path(args.save_dir) / f"resnet34-{epoch + 1}-{val_loss}.pth"
                    torch.save(model.module.state_dict(), checkpoint)
                val_losses.append(val_loss)
        if dist.get_rank() == 0:
            checkpoint = Path(args.save_dir) / f"latest.pth"
            torch.save(model.module.state_dict(), checkpoint)
        lr_scheduler.step()


def test(model: nn.Module,
         val_loader: DataLoader,
         criterion: nn.Module,
         device: torch.device):
    val_loss, val_accuracy = val_epoch(model, val_loader, criterion, device)
    logging.info(f"test loss: {val_loss}, test accuracy: {val_accuracy * 100: .2f}")


# TODO 研究一下Scaler 是否在测试阶段也使用
@torch.no_grad()
def val_epoch(model: nn.Module,
              val_loader: DataLoader,
              criterion: nn.Module,
              device: torch.device,
              ):
    epoch_loss = AverageMeter()
    epoch_top1 = AverageMeter()
    model.eval()
    loss, acc_top1 = None, None
    if dist.get_rank() == 0:
        val_loader = tqdm(desc=f"val", iterable=val_loader)
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        predictions = model(images)
        loss = criterion(predictions, labels)
        acc_top1 = accuracy(predictions, labels)[0]
        loss = reduce_tensor(loss)
        acc_top1 = reduce_tensor(acc_top1)
        epoch_loss.update(loss.cpu().item(), labels.shape[0])
        epoch_top1.update(acc_top1.cpu().item())
    logging.info(f"{dist.get_rank()} loss: {loss.cpu().item()}, acc: {acc_top1.cpu().item()}")
    return epoch_loss.avg, epoch_top1.avg


def train_epoch(model: nn.Module,
                train_loader: DataLoader,
                optimizer: optim.Optimizer,
                criterion: nn.Module,
                device: torch.device,
                scaler: GradScaler,
                args: Namespace,
                ):
    model.train()
    if dist.get_rank() == 0:
        train_loader = tqdm(desc=f"train", iterable=train_loader)
    train_loss_epoch = torch.tensor(0.)
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        with autocast(enabled=args.use_mix_precision):
            predictions = model(images)
            loss = criterion(predictions, labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss_epoch += loss.cpu()
        # bar.update()
    return train_loss_epoch.cpu().item() / len(train_loader)


def main(args):
    if not Path(args.save_dir).exists():
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    dist.init_process_group(backend="nccl", init_method="env://")
    dist.barrier()
    device = torch.device("cuda", args.local_rank)
    model = ResNet34(10)
    model.to(device)
    model = DistributedDataParallel(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), 1e-1)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 60, 80], gamma=0.1)
    train_loader = CIFAR10(args.data_dir, args.batch_size, True)
    val_loader = CIFAR10(args.data_dir, args.batch_size, False)
    scaler = GradScaler(enabled=args.use_mix_precision)
    logging.info(f"train sampler: {train_loader.sampler.__class__.__name__}")
    logging.info(f"val sampler: {val_loader.sampler.__class__.__name__}")
    train(model, train_loader, val_loader, optimizer, lr_scheduler, criterion, device, scaler, args)
    test(model, val_loader, criterion, device)


if __name__ == '__main__':
    args = parse_args()
    main(args)
