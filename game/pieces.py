
import random
PIECES = [
    [(0,0)],
    [(0,0),(1,0)],
    [(0,0),(0,1)],
    [(0,0),(1,0),(2,0)],
    [(0,0),(0,1),(0,2)],
    [(0,0),(1,0),(0,1),(1,1)],
]
def sample(n=3):
    return random.sample(PIECES, n)
