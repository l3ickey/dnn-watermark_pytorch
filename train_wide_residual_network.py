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
from watermark.watermark_regularizers import CNNWatermarkRegularizer


def train(model, device, train_loader, optimizer, criterion, regularizer, epoch):
    global train_loss, regularizer_loss
    model.train()
    progress_bar = tqdm(train_loader, leave=False)
    for batch_idx, (data, target) in enumerate(progress_bar):
        progress_bar.set_description(f"Epoch: {epoch}")
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        optimizer.zero_grad()
        output = model(data)
        train_loss = criterion(output, target)
        regularizer_loss = regularizer()
        loss = train_loss + regularizer_loss
        loss.backward()
        optimizer.step()
    tqdm.write(f"Epoch: {epoch}, Train loss: {train_loss.item():.4f}, "
               f"Watermark regularizer loss: {regularizer_loss.item():.4f}", end=", ")


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
        f"Test loss: {test_loss:.4f}, "
        f"Test accuracy: {correct}/{len(test_loader.dataset)} ({(100. * correct / len(test_loader.dataset)):.0f}%), "
        f"Elapsed time: {int(time.time() - start_time)} sec")
    return test_loss


class SaveBestModel:
    def __init__(self):
        self.best_loss = float('inf')

    def __call__(self, epoch, model, optimizer, criterion, current_loss):
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': criterion},
                       'outputs/wide_residual_network/best_model.pth')


def main():
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10", type=str, help="dataset")
    parser.add_argument("--history", default="outputs/wide_residual_network/train_history.h5", type=str,
                        help="history file path")
    parser.add_argument("--batch_size", default=64, type=int, help="batch size")
    parser.add_argument("--epochs", default=200, type=int, help="learning epochs")
    parser.add_argument("--scale", default=0.01, type=float, help="lambda of regulaiization loss")
    parser.add_argument("--embed_dim", default=256, type=int, help="number of dimensions of the embedding vector")
    parser.add_argument("--N", default=1, type=int, help="depth of wide residual network")
    parser.add_argument("--k", default=4, type=int, help="width of wide residual network")
    parser.add_argument("--target_blk_id", default=1, type=int, choices=[0, 1, 2, 3],
                        help="If 0, without embedding a watermark")
    parser.add_argument("--wmark_wtype", default="random", type=str, choices=["direct", "diff", "random"],
                        help="watarmark type")
    parser.add_argument("--base_modelw_fname", default="", type=str, help="pre-trained model file path (.pth)")
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
    model = WideResNet(n=args.N, widen_factor=args.k).to(device)
    if args.base_modelw_fname:  # load pre-trained model
        model.load_state_dict(torch.load(args.base_modelw_fname)['model_state_dict'])
    summary(model)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    train_criterion = nn.CrossEntropyLoss(reduction='mean')
    test_criterion = nn.CrossEntropyLoss(reduction='sum')
    saver = SaveBestModel()

    # watermark settings
    if args.target_blk_id == 0:
        embed_weight = None
    elif args.target_blk_id == 1:
        embed_weight = model.block1.layer[0].conv2.weight  # (N,C,H,W)
    elif args.target_blk_id == 2:
        embed_weight = model.block2.layer[0].conv2.weight  # (N,C,H,W)
    elif args.target_blk_id == 3:
        embed_weight = model.block3.layer[0].conv2.weight  # (N,C,H,W)
    else:
        raise ValueError(f"Unsupported target block id: {args.target_blk_id}")
    wmark_regularizer = CNNWatermarkRegularizer(device, args.scale, args.embed_dim, args.wmark_wtype, embed_weight)
    print(f"Watermark matrix:\n{wmark_regularizer.get_matrix()}")

    # train
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train(model, device, train_loader, optimizer, train_criterion, wmark_regularizer, epoch)
        test_loss = test(model, device, test_loader, test_criterion, start_time)
        saver(epoch, model, optimizer, train_criterion, test_loss)
        scheduler.step()


if __name__ == '__main__':
    main()
