"""
Block Blast - 27 Pieces Definition
Extracted from block_blast_complete.py
"""

import random
from typing import List, Tuple, Dict
from dataclasses import dataclass

# 27 pièces exactes du jeu Block Blast
PIECES_DATA = {
    # 1 bloc
    '1': [(0, 0)],

    # 2 blocs
    '2H': [(0, 0), (1, 0)],
    '2V': [(0, 0), (0, 1)],

    # 3 blocs
    '3H': [(0, 0), (1, 0), (2, 0)],
    '3V': [(0, 0), (0, 1), (0, 2)],
    'L3_1': [(0, 0), (0, 1), (1, 1)],
    'L3_2': [(0, 0), (1, 0), (0, 1)],
    'L3_3': [(0, 0), (1, 0), (1, 1)],
    'L3_4': [(1, 0), (0, 1), (1, 1)],

    # 4 blocs - lignes
    '4H': [(0, 0), (1, 0), (2, 0), (3, 0)],
    '4V': [(0, 0), (0, 1), (0, 2), (0, 3)],

    # 4 blocs - carré
    'SQ': [(0, 0), (1, 0), (0, 1), (1, 1)],

    # 4 blocs - T
    'T_N': [(0, 0), (1, 0), (2, 0), (1, 1)],
    'T_E': [(1, 0), (0, 1), (1, 1), (1, 2)],
    'T_S': [(1, 0), (0, 1), (1, 1), (2, 1)],
    'T_W': [(0, 0), (0, 1), (1, 1), (0, 2)],

    # 4 blocs - L
    'L_1': [(0, 0), (0, 1), (0, 2), (1, 2)],
    'L_2': [(0, 0), (1, 0), (2, 0), (0, 1)],
    'L_3': [(1, 0), (1, 1), (1, 2), (0, 2)],
    'L_4': [(2, 0), (0, 1), (1, 1), (2, 1)],
    'L_5': [(0, 0), (0, 1), (0, 2), (1, 0)],
    'L_6': [(0, 0), (1, 0), (2, 0), (2, 1)],
    'L_7': [(0, 0), (1, 0), (1, 1), (1, 2)],
    'L_8': [(0, 0), (0, 1), (1, 1), (2, 1)],

    # 4 blocs - S/Z
    'S4': [(1, 0), (2, 0), (0, 1), (1, 1)],
    'Z4': [(0, 0), (1, 0), (1, 1), (2, 1)],

    # 5 blocs
    '5H': [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)],
    '5V': [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],

    # 3x3 carré
    'SQ3': [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2)],
}

# Couleurs disponibles
COLORS = ['yellow', 'purple', 'green', 'orange', 'blue', 'red', 'cyan', 'pink']

# Couleurs RGB pour le renderer
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
        """Retourne (min_x, max_x, min_y, max_y)"""
        xs = [x for x, y in self.shape]
        ys = [y for x, y in self.shape]
        return min(xs), max(xs), min(ys), max(ys)

    def get_size(self) -> Tuple[int, int]:
        """Retourne (width, height)"""
        min_x, max_x, min_y, max_y = self.get_bounds()
        return max_x - min_x + 1, max_y - min_y + 1

    def num_cells(self) -> int:
        return len(self.shape)


# Liste ordonnée des IDs de pièces
PIECE_IDS = list(PIECES_DATA.keys())

# Dictionnaire pré-construit des pièces (sans couleur, pour référence)
PIECES: Dict[str, List[Tuple[int, int]]] = PIECES_DATA


def get_piece_by_id(piece_id: str, color: str = None) -> Piece:
    """Retourne une pièce par son ID avec une couleur optionnelle"""
    if piece_id not in PIECES_DATA:
        raise ValueError(f"Unknown piece ID: {piece_id}")
    if color is None:
        color = random.choice(COLORS)
    return Piece(id=piece_id, shape=PIECES_DATA[piece_id], color=color)


def get_all_piece_ids() -> List[str]:
    """Retourne tous les IDs de pièces disponibles"""
    return PIECE_IDS.copy()


def sample_pieces(n: int = 3) -> List[Piece]:
    """Échantillonne n pièces aléatoires avec couleurs aléatoires"""
    piece_ids = random.choices(PIECE_IDS, k=n)
    return [get_piece_by_id(pid, random.choice(COLORS)) for pid in piece_ids]


def get_piece_index(piece_id: str) -> int:
    """Retourne l'index numérique d'une pièce (0-29)"""
    return PIECE_IDS.index(piece_id)


def get_piece_id_from_index(index: int) -> str:
    """Retourne l'ID de pièce depuis son index"""
    return PIECE_IDS[index]
