
import random
from typing import List, Tuple
from dataclasses import dataclass


PIECES_DATA = {

    '1': [(0, 0)],


    '2H': [(0, 0), (1, 0)],
    '2V': [(0, 0), (0, 1)],


    '3H': [(0, 0), (1, 0), (2, 0)],
    '3V': [(0, 0), (0, 1), (0, 2)],
    'L3_1': [(0, 0), (0, 1), (1, 1)],
    'L3_2': [(0, 0), (1, 0), (0, 1)],
    'L3_3': [(0, 0), (1, 0), (1, 1)],
    'L3_4': [(1, 0), (0, 1), (1, 1)],


    '4H': [(0, 0), (1, 0), (2, 0), (3, 0)],
    '4V': [(0, 0), (0, 1), (0, 2), (0, 3)],


    'SQ': [(0, 0), (1, 0), (0, 1), (1, 1)],


    'T_N': [(0, 0), (1, 0), (2, 0), (1, 1)],
    'T_E': [(1, 0), (0, 1), (1, 1), (1, 2)],
    'T_S': [(1, 0), (0, 1), (1, 1), (2, 1)],
    'T_W': [(0, 0), (0, 1), (1, 1), (0, 2)],


    'L_1': [(0, 0), (0, 1), (0, 2), (1, 2)],
    'L_2': [(0, 0), (1, 0), (2, 0), (0, 1)],
    'L_3': [(1, 0), (1, 1), (1, 2), (0, 2)],
    'L_4': [(2, 0), (0, 1), (1, 1), (2, 1)],
    'L_5': [(0, 0), (0, 1), (0, 2), (1, 0)],
    'L_6': [(0, 0), (1, 0), (2, 0), (2, 1)],
    'L_7': [(0, 0), (1, 0), (1, 1), (1, 2)],
    'L_8': [(0, 0), (0, 1), (1, 1), (2, 1)],


    'S4': [(1, 0), (2, 0), (0, 1), (1, 1)],
    'Z4': [(0, 0), (1, 0), (1, 1), (2, 1)],


    '5H': [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)],
    '5V': [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],


    'SQ3': [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2)],
}


COLORS = ['yellow', 'purple', 'green', 'orange', 'blue', 'red', 'cyan', 'pink']


COLOR_RGB = {
    'yellow': (255, 200, 50),
    'purple': (170, 120, 230),
    'green': (100, 220, 100),
    'orange': (255, 140, 50),
    'blue': (80, 160, 230),
    'red': (230, 80, 80),
    'cyan': (100, 220, 220),
    'pink': (230, 120, 170),
}

@dataclass
class Piece:
    id: str
    shape: List[Tuple[int, int]]
    color: str

    def get_bounds(self) -> Tuple[int, int, int, int]:
        xs = [x for x, y in self.shape]
        ys = [y for x, y in self.shape]
        return min(xs), max(xs), min(ys), max(ys)

    def get_size(self) -> Tuple[int, int]:
        min_x, max_x, min_y, max_y = self.get_bounds()
        return max_x - min_x + 1, max_y - min_y + 1

    def num_cells(self) -> int:
        return len(self.shape)



PIECE_IDS = list(PIECES_DATA.keys())


def get_piece_by_id(piece_id: str, color: str = None) -> Piece:
    if piece_id not in PIECES_DATA:
        raise ValueError(f"Unknown piece ID: {piece_id}")
    if color is None:
        color = random.choice(COLORS)
    return Piece(id=piece_id, shape=PIECES_DATA[piece_id], color=color)


def sample_pieces(n: int = 3) -> List[Piece]:
    piece_ids = random.choices(PIECE_IDS, k=n)
    return [get_piece_by_id(pid, random.choice(COLORS)) for pid in piece_ids]
