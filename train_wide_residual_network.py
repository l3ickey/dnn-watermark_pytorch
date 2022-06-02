import argparse
import os

from torch import manual_seed, device, cuda
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, MultiStepLR
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchinfo import summary

from models.wide_residual_network import WideResNet

if __name__ == '__main__':
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10", type=str, help="学習データセット")
    parser.add_argument("--history", default="result/train_history.h5", type=str, help="loss, accの保存ファイルパス")
    parser.add_argument("--batch_size", default=64, type=int, help="学習バッチサイズ")
    parser.add_argument("--epoch", default=200, type=int, help="学習エポック数")
    parser.add_argument("--scale", default=0.01, type=float, help="電子透かしの正則化重み")
    parser.add_argument("--embed_dim", default=256, type=int, help="電子透かしの強度")
    parser.add_argument("--N", default=1, type=int, help="wide residual networkの深さ")
    parser.add_argument("--k", default=4, type=int, help="wide residual networkの幅")
    parser.add_argument("--target_blk_id", default=1, type=int, choices=[1, 2, 3], help="電子透かしを埋め込むブロックのID")
    parser.add_argument("--wmark_wtype", default="random", type=str, choices=["direct", "diff", "random"],
                        help="電子透かしの種類")
    parser.add_argument("--base_modelw_fname", default="", type=str, help="事前学習重みのファイルパス")
    args = parser.parse_args()

    # data augment
    manual_seed(0)
    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    transforms = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(5 / 32, 5 / 32)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # load dataset
    if args.dataset == 'cifar10':
        trainset = datasets.CIFAR10(root='/root/dataset', train=True, download=False, transforms=transforms)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shffle=True, num_workers=os.cpu_count(),
                             pin_memory=True)

    # device setting
    device = device("cuda:0" if cuda.is_available() else "cpu")

    # network settings
    model = WideResNet().to(device)
    summary(model)

    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True)
    scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    criterion = CrossEntropyLoss()

    # train
