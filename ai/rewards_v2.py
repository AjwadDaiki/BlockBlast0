"""
Improved Reward Shaping V2 for Block Blast RL
More aggressive bonuses for clears and survival
"""

from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass
class RewardV2Config:
    """Configuration for improved reward shaping"""
    # Base rewards - HIGHER SCALE
    score_scale: float = 0.05         # Lower scale to avoid huge values

    # Clear bonuses - MUCH HIGHER
    clear_bonus: float = 10.0         # Was 5.0
    multi_clear_multiplier: float = 3.0  # Exponential bonus for multi-clears

    # Combo bonuses - HIGHER
    combo_bonus: float = 5.0          # Was 3.0
    combo_multiplier: float = 1.5     # Exponential for long combos

    # Penalties
    game_over_penalty: float = 20.0   # Was 10.0

    # Survival bonuses
    survival_bonus: float = 0.1       # Small bonus for each step survived
    valid_moves_bonus: float = 0.05   # Bonus per valid move available

    # Strategic bonuses
    space_cleared_bonus: float = 0.1  # Bonus for each cell cleared
    center_bonus: float = 0.02        # Small bonus for playing near center


def compute_reward_v2(info: Dict,
                      config: RewardV2Config = None,
                      prev_info: Dict = None) -> float:
    """
    Compute improved reward from step info.

    Key improvements:
    - Exponential bonuses for multi-clears
    - Higher clear bonuses
    - Survival bonuses
    - Strategic position bonuses
    """
    if config is None:
        config = RewardV2Config()

    reward = 0.0

    # 1. Base score reward (normalized)
    score_delta = info.get('score_delta', 0)
    reward += score_delta * config.score_scale

    # 2. Clear bonuses - EXPONENTIAL for multiple clears
    k_clears = info.get('k_clears', 0)
    if k_clears > 0:
        # Base clear bonus
        clear_reward = config.clear_bonus * k_clears

        # Exponential multiplier for multi-clears (2+ lines)
        if k_clears >= 2:
            clear_reward *= (config.multi_clear_multiplier ** (k_clears - 1))

        reward += clear_reward

    # 3. Combo bonus - EXPONENTIAL for long combos
    combo = info.get('combo_streak', 0)
    if combo > 1:
        combo_reward = config.combo_bonus * (config.combo_multiplier ** (combo - 1))
        reward += combo_reward

    # 4. Survival bonus (encourages not dying)
    reward += config.survival_bonus

    # 5. Valid moves bonus (encourages keeping options open)
    num_valid = info.get('valid_mask_summary', {}).get('num_valid', 0)
    if num_valid > 10:
        reward += config.valid_moves_bonus * min(num_valid / 50, 1.0)
    elif num_valid < 5:
        # Penalty for dangerous situations
        reward -= (5 - num_valid) * 0.5

    # 6. Space cleared bonus
    cleared_rows = len(info.get('cleared_rows', []))
    cleared_cols = len(info.get('cleared_cols', []))
    cells_cleared = cleared_rows * 8 + cleared_cols * 8 - cleared_rows * cleared_cols  # Avoid double counting
    reward += cells_cleared * config.space_cleared_bonus

    # 7. Center placement bonus (if action info available)
    action = info.get('action', {})
    if action and action.get('x') is not None:
        x, y = action.get('x', 0), action.get('y', 0)
        # Distance from center (3.5, 3.5)
        center_dist = abs(x - 3.5) + abs(y - 3.5)
        if center_dist < 3:
            reward += config.center_bonus * (3 - center_dist)

    return reward


def compute_terminal_reward_v2(score: int, steps: int, config: RewardV2Config = None) -> float:
    """Compute reward for terminal state (game over)"""
    if config is None:
        config = RewardV2Config()

    # Base penalty
    penalty = -config.game_over_penalty

    # Reduce penalty if survived long
    survival_factor = min(steps / 100, 1.0)  # Max reduction at 100 steps
    penalty *= (1 - survival_factor * 0.5)  # Up to 50% reduction

    return penalty


def get_reward_summary(reward: float) -> str:
    """Get human-readable reward description"""
    if reward > 50:
        return "AMAZING!"
    elif reward > 20:
        return "Great!"
    elif reward > 10:
        return "Good"
    elif reward > 0:
        return "OK"
    elif reward > -5:
        return "Meh"
    else:
        return "Bad"
