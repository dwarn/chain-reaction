import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensors import channels


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

class Agent(nn.Module):
    def __init__(self, filters=1):
        super(Agent, self).__init__()

        self.m = nn.Sequential(
            torch.nn.Conv2d(channels, filters, 3),
            Permute((0, 2, 3, 1)),
            nn.Linear(filters, 1),
        )

    def forward(self, x):
        return F.log_softmax(self.m(x), dim=1)

    def step(self, x, a, r, optimizer):
        optimizer.zero_grad()
        logpi = self(x)
        loss = r @ logpi[range(a.size(0)), a]
        loss.backward()
        optimizer.step()

    def act(self, x):
        """
        Returns an index in [0, n*n)
        Can be transformed with x, y = (idx//n, idx % n)
        """
        pi = F.softmax(self.m(x), dim=1)
        return random.choice(len(pi[0]), p=pi[0])


