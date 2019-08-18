from game import Action, State, random_action, n
from math import sqrt, log
import random

def argmax(ls):
    ls = list(ls)
    (y,x) = ls[0]
    for (b, a) in ls[1:]:
        if b > y:
            (y,x) = (b,a)
    return x

class Node:
    def __init__(self, state):
        self.lost = state.winner != 0
        if not self.lost:
            self.wins = 0
            self.n = 0
            self.children = []
            self.to_try = []
            for i in range(n): 
                for j in range(n):
                    a = Action(i,j)
                    if state.legal_move(a):
                        self.to_try.append(a)
            random.shuffle(self.to_try)

    def ucb(self, parent_n):
        if self.lost:
            return 0
        return self.wins / self.n + sqrt( 2 * log(parent_n) / self.n )

    def rollout(self, state):
        cur_player = state.player
        while state.winner == 0:
            while True:
                a = random_action()
                if state.legal_move(a):
                    break
            state.make_move(a)
        return cur_player == state.winner

    def go(self, state):
        if self.lost:
            return 0 
        if self.n == 0:
            result = 1 - self.rollout(state)
        else:
            if self.to_try:
                action = self.to_try.pop()
                state.make_move(action)
                child = Node(state)
                self.children.append( (action, child) )
            else:
                (child, action) = argmax( (v.ucb(self.n), (v, a))  for (a,v) in self.children )
                state.make_move(action)
            result = 1 - child.go(state)
        self.wins += result
        self.n += 1
        return result

iterations = 500

def get_move(state):
    root = Node(state)
    for t in range(iterations):
        root.go(state.copy())
    return argmax( (iterations*2 if v.lost else v.wins, a) for (a,v) in root.children )

def simulate(state):
    while state.winner == 0:
        state.show_board()
        a = get_move(state)
        state.make_move(a)
    print("And the winner is:", state.winner)
