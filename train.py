from game import State, random_action
from agent import Agent
from tensors import to_tensors
import numpy as np
import torch.optim as optim

def simulate(self):
    states = []
    actions = []
    while self.winner == 0:
        while True:
            a = random_action()
            if self.legal_move(a):
                break
        states.append(self.copy())
        actions.append(a)

        self.make_move(a)

    for i in range(len(states)):
        states[i].winner = winner

    return states, actions

episodes = 10
n_to_store = 5
batch_size = 32

state_buffer = []
action_buffer = []

agent = Agent()
optimizer = optim.SGD(agent.parameters(), lr=0.01, momentum=0.5)
for _ in range(episodes):
    s = State()
    states, actions, winner = s.simulate()

    # Only store the last 5 moves
    state_buffer += states[-n_to_store:]
    action_buffer += actions[-n_to_store:]

    if len(state_buffer) < batch_size:
        continue

    # sample historical data
    idx = np.random.choice(len(state_buffer), batch_size, replace=False)

    # get batch tensor
    states, actions, rewards = to_tensors(
        [state_buffer[i] for i in idx],
        [action_buffer[i] for i in idx],
    )

    agent.step(states, actions, rewards)



