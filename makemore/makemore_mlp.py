"""
A separate file recreating the models in the videos using more PyTorch abstractions.
"""

import torch
import torch.nn as nn
from torch.random import manual_seed
import random

random.seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
g = torch.Generator(device=device).manual_seed(1234)

words = open("makemore/names.txt", "r").read().splitlines()
chars = sorted(list(set("".join(words))))

stoi = {s: i for i, s in enumerate(chars, start=1)}
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}

vocab_size = len(itos)
block_size = 3

max_steps = 200_000
batch_size = 32


class MakemoreMLP(nn.Module):
    def __init__(self, n_emb: int):
        super().__init__()
        manual_seed(1234)
        self.emb = nn.Embedding(vocab_size, n_emb)
        self.layers = nn.Sequential(
            nn.Linear(block_size * n_emb, 200, bias=True),
            nn.BatchNorm1d(200, momentum=0.001),
            nn.Tanh(),
            nn.Linear(200, vocab_size, bias=True),
        )

    def forward(self, x):
        emb = self.emb(x)
        emb = emb.view(emb.shape[0], -1)
        out = self.layers(emb)
        return out


def build_dataset(words: list[str]) -> tuple[torch.tensor, torch.tensor]:
    X, y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + ".":
            ix = stoi[ch]
            X.append(context)
            y.append(ix)
            context = context[1:] + [ix]
    X, y = torch.tensor(X, device=device), torch.tensor(y, device=device)
    print(X.shape, y.shape)

    return X, y


# Recreating PyTorch classes for learning purposes. Not used in the training loop currently.
class Linear:
    def __init__(self, features_in: int, features_out: int, bias: bool = True):
        # Initialize weights with linear Kaiming init
        self.weight = (
            torch.randn((features_in, features_out), generator=g) / features_in**0.5
        )
        self.bias = torch.zeros(features_out) if bias else None

    def __call__(self, x: torch.tensor):
        # y = xA.T + b
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([self.bias] if self.bias else [])


class BatchNorm1d:
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        self.eps = eps  # Epsilon added to variance to prevent divide-by-zero issues
        self.momentum = momentum
        self.training = True

        # Gain and bias params trained through backprop
        self.gamma = torch.ones((1, num_features))
        self.beta = torch.zeros((1, num_features))

        # Buffers for calculating batch norm stats with moving averages
        self.bn_mean_running = torch.zeros((1, num_features))
        self.bn_var_running = torch.ones((1, num_features))

    def __call__(self, x: torch.Tensor):
        if self.training:
            batch_mean = x.mean(0, keepdim=True)
            batch_var = x.std(0, keepdim=True)
        else:
            batch_mean = self.bn_mean_running
            batch_var = self.bn_var_running

        out = (
            self.gamma * (x - batch_mean) / torch.sqrt(batch_var + self.eps) + self.beta
        )  # Batch normalization operation using the params defined earlier

        if self.training:
            with torch.no_grad():
                self.bn_mean_running = (
                    1 - self.momentum
                ) * self.bn_mean_running + self.momentum * batch_mean
                self.bn_var_running = (
                    1 - self.momentum
                ) * self.bn_var_running + self.momentum * batch_var

        return out

    def parameters(self):
        return [self.gamma, self.beta]


if __name__ == "__main__":
    random.seed(42)
    random.shuffle(words)
    n1 = int(0.8 * len(words))
    n2 = int(0.9 * len(words))

    X_train, y_train = build_dataset(words[:n1])  # 80%
    X_val, y_val = build_dataset(words[n1:n2])  # 10%
    X_test, y_test = build_dataset(words[n2:])  # 10%

    model = MakemoreMLP(n_emb=10)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(params=model.parameters(), lr=0.1)

    lossi = []
    for i in range(max_steps):
        ix = torch.randint(
            0, X_train.shape[0], (batch_size,), generator=g, device=device
        )
        Xb, yb = X_train[ix], y_train[ix]
        model.train()
        optim.zero_grad(set_to_none=True)
        out = model(Xb)
        loss = criterion(out, yb)
        loss.backward()
        optim.step()

        if i % 10_000 == 0:
            print(f"{i:7d}/{max_steps:7d}: {loss.item():.4f}")
        if i >= 100_000:
            for params in optim.param_groups:
                params["lr"] = 0.001
        lossi.append(loss.log10().item())
