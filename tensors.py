import numpy as np
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
    batch = np.zeros([len(state_list), channels, N + 2, N + 2])

    # Board Edge
    batch[:, 0, 0, :] = 1
    batch[:, 0, N + 1, :] = 1
    batch[:, 0, :, 0] = 1
    batch[:, 0, :, N + 1] = 1


    for i, s in enumerate(state_list):
        # These represent the "current player"
        batch[i, 1, 1:N + 1, 1:N + 1] = s.board == 1 * s.player
        batch[i, 2, 1:N + 1, 1:N + 1] = s.board == 2 * s.player
        batch[i, 3, 1:N + 1, 1:N + 1] = s.board == 3 * s.player

        # These represent the "other player"
        batch[i, 4, 1:N + 1, 1:N + 1] = s.board == -1 * s.player
        batch[i, 5, 1:N + 1, 1:N + 1] = s.board == -2 * s.player
        batch[i, 6, 1:N + 1, 1:N + 1] = s.board == -3 * s.player

    actions = [N * a.y + a.x for a in action_list]
    actions = np.array(actions, dtype=int)

    # did "current player" win
    winners = np.array([s.player*s.winner for s in state_list], dtype=float)

    return batch, actions, winners

