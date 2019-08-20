import numpy as np
import torch
from typing import List
from game import N, State, Action

channels = 7

def to_tensors(
        state_list: List[State],
        action_list: List[Action],
):
    """
    Converts lists of states and actions to 3 tensors
    """
    batch = torch.zeros([len(state_list), channels, N + 2, N + 2], dtype=torch.float)

    # Board Edge
    batch[:, 0, 0, :] = 1
    batch[:, 0, N + 1, :] = 1
    batch[:, 0, :, 0] = 1
    batch[:, 0, :, N + 1] = 1


    for i, s in enumerate(state_list):
        # These represent the "current player"

        board = torch.from_numpy(s.board)
        batch[i, 1, 1:N + 1, 1:N + 1] = board == 1 * s.player
        batch[i, 2, 1:N + 1, 1:N + 1] = board == 2 * s.player
        batch[i, 3, 1:N + 1, 1:N + 1] = board == 3 * s.player

        # These represent the "other player"
        batch[i, 4, 1:N + 1, 1:N + 1] = board == -1 * s.player
        batch[i, 5, 1:N + 1, 1:N + 1] = board == -2 * s.player
        batch[i, 6, 1:N + 1, 1:N + 1] = board == -3 * s.player

    actions = [a.y + N*a.x for a in action_list]
    actions = torch.tensor(actions, dtype=torch.int64)

    # did "current player" win
    winners = torch.tensor([s.player*s.winner for s in state_list]).float()

    return batch, actions, winners

