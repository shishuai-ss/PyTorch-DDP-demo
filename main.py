from datetime import datetime
from argparse import ArgumentParser, Namespace
import torchvision
import torch
from torchvision import models
from torchvision import datasets
from torch import nn
import torch.distributed as dist
from tqdm import tqdm


