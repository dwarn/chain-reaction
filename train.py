# %%
from game import State, random_action, Action, N
from agent import Agent
from tensors import state_tensor, action_tensor
import numpy as np
import torch.optim as optim


def self_play(agent):
    s = State()
    states = []
    actions = []
    while s.winner == 0:
        batch, valid, _ = state_tensor([s])
        idx = agent.act(batch, valid)
        a = Action(idx // N, idx % N)
        states.append(s.copy())
        actions.append(a)

        s.make_move(a)

    for i in range(len(states)):
        states[i].winner = s.winner

    return states, actions


def duel(*players):
    s = State()
    player = 0
    while s.winner == 0:
        batch, valid, _ = state_tensor([s])
        while True:
            agent = players[player]
            idx = agent(batch, valid)
            a = Action(idx // N, idx % N)
            if s.legal_move(a):
                break

        s.make_move(a)
        player = (player + 1) % len(players)

    return s.winner


def random_agent(*_):
    return np.random.randint(0, N * N)


def main():
    episodes = 100000
    eval_games = 10

    batch_size = 32

    state_buffer = []
    action_buffer = []

    agent = Agent()
    optimizer = optim.SGD(agent.parameters(), lr=0.005, momentum=0.0)
    tot_wins = 0
    tot_games = 0
    scores = []

    for i in range(episodes):
        s = State()
        states, actions = self_play(agent)
        n_to_store = 5 + i // 200

        # Only store the last 5 moves
        state_buffer += states[-n_to_store:]
        action_buffer += actions[-n_to_store:]

        if len(state_buffer) < batch_size:
            continue

        # get batch tensor
        states, valids, rewards = state_tensor(
            state_buffer,
        )
        actions = action_tensor(
            action_buffer,
        )

        agent.step(states, valids, actions, rewards, optimizer)

        state_buffer = []
        action_buffer = []

        if i % 1 == 0:
            wins = 0
            for _ in range(0, eval_games, 2):
                winner = duel(agent.act, random_agent)
                if winner == 1:
                    wins += 1
                winner = duel(random_agent, agent.act)
                if winner == -1:
                    wins += 1

            tot_wins += wins
            tot_games += eval_games
            scores.append(wins / eval_games)
            res = {'episode': i, 'score': wins / eval_games, 'n_games': eval_games,
                   'std': 1 / np.sqrt(eval_games), 'total_score': tot_wins / tot_games,
                   'trailing_20_avg': np.mean(scores[-20:]),
                   'n_to_store': n_to_store}
            print(res)


main()
