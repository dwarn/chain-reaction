import numpy as np
import random
import time

n = 7

diffs = [(1,0), (-1,0), (0,1), (0,-1)]

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def visa(x):
    if x==0:
        return " "
    if x < 0:
        return bcolors.OKGREEN + str(-x) + bcolors.ENDC
    if x > 0:
        return bcolors.OKBLUE + str(x) + bcolors.ENDC

class Action:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def valid_action(self):
        return 0 <= min(self.x,self.y) and max(self.x,self.y) < n

    def neighbours(self):
        return list(filter(lambda b: b.valid_action(), (Action(self.x + dx, self.y + dy) for (dx, dy) in diffs)))

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

def random_action():
    return Action( random.choice(range(n)), random.choice(range(n)) )

class State:
    def __init__(self, other = None):
        if other:
            self.board = other.board.copy()
            self.player = other.player
            self.cells = other.cells.copy()
            self.winner = other.winner
        else:
            self.board = np.zeros( (n,n), dtype=int) 
            self.player = 1
            self.cells = { 
                      1: 0, 
                     -1: 0
                    }
            self.winner = 0

    def copy(self):
        return State(self)

    def legal_move(self, a):
        return a.valid_action() and self.player * self.board[a.x][a.y] >= 0
    
    def make_move(self, a, change_player = True):
        if self.player * self.board[a.x][a.y] < 0:
            self.board[a.x][a.y] *= -1
            self.cells[ self.player] += 1
            self.cells[-self.player] -= 1
        elif self.board[a.x][a.y] == 0:
            self.cells[self.player] += 1
        if self.cells[-self.player] == 0 and self.cells[self.player] >= 2:
            self.winner = self.player 
            return 
        self.board[a.x][a.y] += self.player
        nbrs = a.neighbours()
        if self.player * self.board[a.x][a.y] >= len(nbrs):
            self.board[a.x][a.y] -= self.player * len(nbrs)
            if self.board[a.x][a.y] == 0:
                self.cells[self.player] -= 1
            for b in nbrs:
                self.make_move(b, False)
        if change_player:
            self.player = -self.player

    def show_board(self):
        print(self.player)
        print( '+' + "".join('-' for i in range(n)) + '+' )
        for r in range(n):
            print("|" + "".join( visa(self.board[r][c]) for c in range(n) ) + "|")
        print( '+' + "".join('-' for i in range(n)) + '+' )

    def simulate(self):
        while self.winner == 0:
            self.show_board()
            time.sleep(1)
            while True:
                a = random_action()
                if self.legal_move(a):
                    break
            self.make_move(a)
        print(self.winner)

if __name__ == '__main__':
    state = State()
    state.simulate()
