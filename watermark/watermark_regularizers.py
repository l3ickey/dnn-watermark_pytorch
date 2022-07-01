import math

import torch


def random_index_generator(row, col):
    indices = torch.randint(0, row, (col,))
    for idx in indices:
        yield idx


class CNNWatermarkRegularizer():
    """
    __init__で重みと電子透かしの種類を受け取り，電子透かしを作成．（どのレイヤーの重みを使うかはクラス外で定義する）
    __call__でlossを計算して返す．
    """

    def __init__(self, wmark_lambda, embed_dim, wmark_type, embed_weigth):
        self.wmark_lambda = wmark_lambda
        self.b = torch.ones(1, embed_dim)
        self.embed_weight = embed_weigth

        # make matrix
        embed_weight_shape = embed_weigth.shape  # (N, C, H, W)
        X_rows = math.prod(embed_weight_shape[1:])  # prod(C, H, W)
        X_cols = embed_dim

        if wmark_type == "direct":
            self.X = torch.zeros(X_rows, X_cols)
            rand_idx = random_index_generator(X_rows, X_cols)
            for col in range(X_cols):
                self.X[next(rand_idx)][col] = 1.
        elif wmark_type == "diff":
            self.X = torch.zeros(X_rows, X_cols)
            rand_idx = random_index_generator(X_rows, X_cols * 2)
            for col in range(X_cols):
                self.X[next(rand_idx)][col] = 1.
                self.X[next(rand_idx)][col] = -1.
        elif wmark_type == "random":
            self.X = torch.randn(X_rows, X_cols)
        else:
            raise ValueError(f"Unsupported watermark type: {wmark_type}")
        print(self.X)
        print(self.X.shape)

    def __call__(self):
        regularized_loss = self.wmark_lambda * K.sum(
            K.binary_crossentropy(K.sigmoid(K.dot(y, z)), K.cast_to_floatx(self.b)))
        return regularized_loss
