import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensors import channels
import numpy as np


class Permute(nn.Module):
    def __init__(self, order):
        super(Permute, self).__init__()
        self.order = order

    def forward(self, x):
        return x.permute(*self.order)


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


def masked_softmax(A, mask, dim=1):
    A_max = torch.max(A[mask == 1])
    A_exp = torch.exp(A - A_max)
    A_exp = A_exp * mask
    A_softmax = A_exp / torch.sum(A_exp, dim=dim, keepdim=True)
    return A_softmax


class Agent(nn.Module):
    def __init__(self, filters=1):
        super(Agent, self).__init__()

        self.m = nn.Sequential(
            torch.nn.Conv2d(channels, filters, 3),
            Permute((0, 2, 3, 1)),
            nn.Linear(filters, 1),  # apply dense layer across each pixel independently
            Flatten(),
        )

    def forward(self, x, valid):
        return masked_softmax(self.m(x), valid)

    def step(self, x, valid, a, r, optimizer):
        optimizer.zero_grad()
        pi = self(x, valid)
        assert not torch.any(pi.gather(1, a.view(-1, 1)) == 0)
        loss = -r @ torch.log(pi.gather(1, a.view(-1, 1)))
        loss.backward()
        optimizer.step()

    def act(self, x, valid):
        """
        Returns an index in [0, n*n)
        Can be transformed with x, y = (idx//n, idx % n)
        """
        pi = self(x, valid)
        return torch.multinomial(pi, 1).detach().numpy()[0, 0]
