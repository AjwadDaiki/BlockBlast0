
import numpy as np

class Grid:
    def __init__(self, size=8):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)

    def can_place(self, shape, x, y):
        for dx, dy in shape:
            xx, yy = x+dx, y+dy
            if xx<0 or yy<0 or xx>=self.size or yy>=self.size:
                return False
            if self.board[yy, xx] == 1:
                return False
        return True

    def place(self, shape, x, y):
        for dx, dy in shape:
            self.board[y+dy, x+dx] = 1

    def clear(self):
        rows = [i for i in range(self.size) if self.board[i].all()]
        cols = [i for i in range(self.size) if self.board[:,i].all()]
        for r in rows: self.board[r,:] = 0
        for c in cols: self.board[:,c] = 0
        return len(rows)+len(cols)
