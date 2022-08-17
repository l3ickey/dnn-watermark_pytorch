import argparse
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.wide_residual_network import WideResNet


def main():
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dirname", type=str,
                        help="this directory must have trained model file (.pth) and watermark matrix file (.npy)")
    parser.add_argument("--embed_dim", default=256, type=int, help="number of dimensions of the embedding vector")
    parser.add_argument("--N", default=1, type=int, help="depth of wide residual network")
    parser.add_argument("--k", default=4, type=int, help="width of wide residual network")
    parser.add_argument("--target_blk_id", default=1, type=int, choices=[0, 1, 2, 3],
                        help="If 0, without embedding a watermark")
    parser.add_argument("--wmark_wtype", default="random", type=str, choices=["direct", "diff", "random"],
                        help="watarmark type")
    args = parser.parse_args()

    # load model
    model = WideResNet(n=args.N, widen_factor=args.k)
    model.load_state_dict(torch.load(f"{args.output_dirname}best_model.pth")['model_state_dict'])

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

    # embedded weight to embedded watermark (before thresholding)
    w = torch.mean(embed_weight, dim=0)  # (N, C, H, W) to (C, H, W)
    w = torch.reshape(w, (1, -1))  # (C, H, W) to (1, C x H x W)
    X = torch.from_numpy(np.load(f"{args.output_dirname}watermark_matrix.npy"))
    embedded_watermark = torch.reshape(torch.sigmoid(torch.mm(w, X)), (-1,))

    # draw histogram
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(embedded_watermark.detach().numpy(), bins=40, alpha=0.5, range=(0, 1), label="Embedded (random)")
    ax.set_xlim(0, 1)
    ax.set_ylabel('Frequency')
    ax.legend(loc='upper left')
    plt.savefig(f"{args.output_dirname}embedded_watermark.png")


if __name__ == '__main__':
    main()
