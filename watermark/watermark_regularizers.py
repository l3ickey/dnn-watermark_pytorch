import math

import torch
import torch.nn.functional as F


def random_index_generator(row, col):
    indices = torch.randint(0, row, (col,))
    for idx in indices:
        yield idx


class CNNWatermarkRegularizer:
    def __init__(self, device, wmark_lambda, embed_dim, wmark_type, embed_weigth):
        self.device = device
        self.wmark_lambda = wmark_lambda
        self.b = torch.ones(1, embed_dim).to(self.device)
        self.embed_weight = embed_weigth  # None or convolution weights

        # make matrix
        if self.embed_weight is not None:
            embed_weight_shape = embed_weigth.shape  # (N, C, H, W)
            X_rows = math.prod(embed_weight_shape[1:])  # prod(C, H, W)
            X_cols = embed_dim

            if wmark_type == "direct":
                self.X = torch.zeros(X_rows, X_cols).to(self.device)
                rand_idx = random_index_generator(X_rows, X_cols)
                for col in range(X_cols):
                    self.X[next(rand_idx)][col] = 1.
            elif wmark_type == "diff":
                self.X = torch.zeros(X_rows, X_cols).to(self.device)
                rand_idx = random_index_generator(X_rows, X_cols * 2)
                for col in range(X_cols):
                    self.X[next(rand_idx)][col] = 1.
                    self.X[next(rand_idx)][col] = -1.
            elif wmark_type == "random":
                self.X = torch.randn(X_rows, X_cols).to(self.device)
            else:
                raise ValueError(f"Unsupported watermark type: {wmark_type}")

    def get_matrix(self):
        return self.X

    def __call__(self):
        if self.embed_weight is not None:
            w = torch.mean(self.embed_weight, dim=0).to(self.device)  # (N, C, H, W) to (C, H, W)
            w = torch.reshape(w, (1, -1))  # (C, H, W) to (1, C x H x W)
            regularization_loss = F.binary_cross_entropy(torch.sigmoid(torch.mm(w, self.X)), self.b, reduction='sum')
            return self.wmark_lambda * regularization_loss
        else:
            return torch.tensor(0.)  # no regularization
