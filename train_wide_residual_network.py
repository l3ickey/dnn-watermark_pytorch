import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchinfo import summary
from tqdm import tqdm

from models.wide_residual_network import WideResNet


def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    global loss
    progress_bar = tqdm(train_loader, leave=False)
    for batch_idx, (data, target) in enumerate(progress_bar):
        progress_bar.set_description(f"Epoch: {epoch}")
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    tqdm.write(f"Epoch: {epoch}, Train Loss: {loss.item():.4f}", end=", ")


def test(model, device, test_loader, criterion, start_time):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    tqdm.write(
        f"Test Loss: {test_loss:.4f}, "
        f"Test Accuracy: {correct}/{len(test_loader.dataset)} ({(100. * correct / len(test_loader.dataset)):.0f}%), "
        f"Elapsed Time: {int(time.time() - start_time)} sec")


def main():
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10", type=str, help="学習データセット")
    parser.add_argument("--history", default="result/train_history.h5", type=str, help="loss, accの保存ファイルパス")
    parser.add_argument("--batch_size", default=64, type=int, help="学習バッチサイズ")
    parser.add_argument("--epochs", default=200, type=int, help="学習エポック数")
    parser.add_argument("--scale", default=0.01, type=float, help="電子透かしの正則化重み")
    parser.add_argument("--embed_dim", default=256, type=int, help="電子透かしの強度")
    parser.add_argument("--N", default=1, type=int, help="wide residual networkの深さ")
    parser.add_argument("--k", default=4, type=int, help="wide residual networkの幅")
    parser.add_argument("--target_blk_id", default=1, type=int, choices=[1, 2, 3], help="電子透かしを埋め込むブロックのID")
    parser.add_argument("--wmark_wtype", default="random", type=str, choices=["direct", "diff", "random"],
                        help="電子透かしの種類")
    parser.add_argument("--base_modelw_fname", default="", type=str, help="事前学習重みのファイルパス")
    args = parser.parse_args()

    # device setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_kwargs = {}
    if device == "cuda":
        device_kwargs.update({'num_workers': os.cpu_count(), 'pin_memory': True})

    # data augment
    torch.manual_seed(0)
    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    transform = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(5 / 32, 5 / 32)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # load dataset
    if args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root='/root/dataset', train=True, download=False, transform=transform)
        test_dataset = datasets.CIFAR10(root='/root/dataset', train=False, download=False, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **device_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **device_kwargs)

    # network settings
    model = WideResNet().to(device)
    summary(model)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    train_criterion = nn.CrossEntropyLoss()
    test_criterion = nn.CrossEntropyLoss(reduction='sum')

    # train
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train(model, device, train_loader, optimizer, train_criterion, epoch)
        test(model, device, test_loader, test_criterion, start_time)
        scheduler.step()


if __name__ == '__main__':
    main()
