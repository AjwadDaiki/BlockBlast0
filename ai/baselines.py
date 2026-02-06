"""
Baseline Agents for Block Blast
Random, ValidRandom, and Heuristic agents
"""

import numpy as np
from typing import List, Dict, Optional
import random


class RandomAgent:
    """Completely random agent - picks any action (may be invalid)"""

    def __init__(self):
        self.name = "Random"

    def select_action(self, env) -> int:
        """Select random action from full action space"""
        return random.randint(0, env.action_dim - 1)

    def reset(self):
        pass


class ValidRandomAgent:
    """Random agent that only picks valid actions"""

    def __init__(self):
        self.name = "ValidRandom"

    def select_action(self, env) -> int:
        """Select random valid action"""
        mask = env.get_valid_action_mask()
        valid_actions = np.where(mask)[0]

        if len(valid_actions) == 0:
            return 0  # Game over anyway

        return np.random.choice(valid_actions)

    def reset(self):
        pass


class HeuristicAgent:
    """
    Heuristic agent that uses simple rules:
    1. Prefer actions that clear lines
    2. Prefer actions that don't create holes
    3. Prefer placing near existing blocks
    4. Prefer using larger pieces first
    """

    def __init__(self,
                 clear_weight: float = 100.0,
                 hole_penalty: float = 10.0,
                 edge_penalty: float = 2.0,
                 cluster_bonus: float = 1.0):
        self.name = "Heuristic"
        self.clear_weight = clear_weight
        self.hole_penalty = hole_penalty
        self.edge_penalty = edge_penalty
        self.cluster_bonus = cluster_bonus

    def select_action(self, env) -> int:
        """Select best action according to heuristics"""
        valid_actions = env.get_all_valid_actions_with_info()

        if not valid_actions:
            return 0

        best_action = None
        best_score = float('-inf')

        for action_info in valid_actions:
            score = self._evaluate_action(env, action_info)
            if score > best_score:
                best_score = score
                best_action = action_info['action_id']

        return best_action if best_action is not None else valid_actions[0]['action_id']

    def _evaluate_action(self, env, action_info: Dict) -> float:
        """Evaluate an action using heuristics"""
        score = 0.0

        # Big bonus for clears
        k_clears = action_info.get('k_clears', 0)
        score += k_clears * self.clear_weight

        # Bonus for score
        score += action_info.get('score', 0)

        # Simulate to get resulting grid state
        piece_idx = action_info['piece_idx']
        x, y = action_info['x'], action_info['y']
        piece = env.pieces[piece_idx]

        if piece is None:
            return score

        # Create simulated grid
        temp_grid = env.grid.copy()
        for dx, dy in piece.shape:
            temp_grid[y + dy, x + dx] = 1

        # Clear lines for accurate hole counting
        cleared_rows = action_info.get('cleared_rows', [])
        cleared_cols = action_info.get('cleared_cols', [])
        for r in cleared_rows:
            temp_grid[r, :] = 0
        for c in cleared_cols:
            temp_grid[:, c] = 0

        # Penalty for holes (empty cells below filled cells)
        holes = self._count_holes(temp_grid)
        score -= holes * self.hole_penalty

        # Penalty for edge placements (harder to clear)
        if x == 0 or x + piece.get_size()[0] >= 8:
            score -= self.edge_penalty
        if y == 0 or y + piece.get_size()[1] >= 8:
            score -= self.edge_penalty

        # Bonus for clustering (placing near other blocks)
        neighbors = self._count_neighbors(env.grid, x, y, piece.shape)
        score += neighbors * self.cluster_bonus

        # Small bonus for using bigger pieces (clear them early)
        score += len(piece.shape) * 0.5

        return score

    def _count_holes(self, grid: np.ndarray) -> int:
        """Count holes (empty cells with filled cells above/left)"""
        holes = 0
        for y in range(1, 8):
            for x in range(8):
                if grid[y, x] == 0:
                    # Check if there's a block above
                    if grid[y - 1, x] == 1:
                        holes += 1
        return holes

    def _count_neighbors(self, grid: np.ndarray, x: int, y: int, shape: List) -> int:
        """Count adjacent filled cells"""
        neighbors = 0
        for dx, dy in shape:
            px, py = x + dx, y + dy
            # Check all 4 directions
            for nx, ny in [(px - 1, py), (px + 1, py), (px, py - 1), (px, py + 1)]:
                if 0 <= nx < 8 and 0 <= ny < 8:
                    if grid[ny, nx] == 1:
                        neighbors += 1
        return neighbors

    def reset(self):
        pass


class GreedyAgent:
    """Agent that always picks the action with maximum immediate score"""

    def __init__(self):
        self.name = "Greedy"

    def select_action(self, env) -> int:
        """Select action with highest immediate score"""
        valid_actions = env.get_all_valid_actions_with_info()

        if not valid_actions:
            return 0

        # Sort by score descending
        valid_actions.sort(key=lambda a: a.get('score', 0), reverse=True)
        return valid_actions[0]['action_id']

    def reset(self):
        pass


class MaxClearAgent:
    """Agent that always picks the action with maximum clears"""

    def __init__(self):
        self.name = "MaxClear"

    def select_action(self, env) -> int:
        """Select action with most clears"""
        valid_actions = env.get_all_valid_actions_with_info()

        if not valid_actions:
            return 0

        # Sort by k_clears descending, then by score
        valid_actions.sort(key=lambda a: (a.get('k_clears', 0), a.get('score', 0)), reverse=True)
        return valid_actions[0]['action_id']

    def reset(self):
        pass


def get_agent(name: str):
    """Get agent by name"""
    agents = {
        'random': RandomAgent,
        'validrandom': ValidRandomAgent,
        'heuristic': HeuristicAgent,
        'greedy': GreedyAgent,
        'maxclear': MaxClearAgent,
    }
    name_lower = name.lower().replace('_', '').replace('-', '')
    if name_lower not in agents:
        raise ValueError(f"Unknown agent: {name}. Available: {list(agents.keys())}")
    return agents[name_lower]()
