
from game.grid import Grid
from game.pieces import sample
import copy

class Env:
    def __init__(self):
        self.reset()

    def reset(self):
        self.grid = Grid()
        self.pieces = sample()
        self.score = 0
        self.done = False

    def step(self, action):
        i,x,y = action
        shape = self.pieces[i]
        if not self.grid.can_place(shape,x,y):
            self.done = True
            return
        self.grid.place(shape,x,y)
        cleared = self.grid.clear()
        self.score += len(shape) + cleared*10
        self.pieces.pop(i)
        if not self.pieces:
            self.pieces = sample()
