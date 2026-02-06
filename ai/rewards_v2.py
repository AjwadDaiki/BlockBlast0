from dataclasses import dataclass
from typing import Dict


@dataclass
class RewardV2Config:
    score_scale: float = 0.05
    clear_bonus: float = 10.0
    multi_clear_multiplier: float = 3.0
    combo_bonus: float = 5.0
    combo_multiplier: float = 1.5
    game_over_penalty: float = 20.0
    survival_bonus: float = 0.1
    valid_moves_bonus: float = 0.05
    space_cleared_bonus: float = 0.1
    center_bonus: float = 0.02


def compute_reward_v2(info: Dict,
                      config: RewardV2Config = None,
                      prev_info: Dict = None) -> float:
    if config is None:
        config = RewardV2Config()

    reward = 0.0

    score_delta = info.get('score_delta', 0)
    reward += score_delta * config.score_scale

    k_clears = info.get('k_clears', 0)
    if k_clears > 0:
        clear_reward = config.clear_bonus * k_clears

        if k_clears >= 2:
            clear_reward *= (config.multi_clear_multiplier ** (k_clears - 1))

        reward += clear_reward

    combo = info.get('combo_streak', 0)
    if combo > 1:
        combo_reward = config.combo_bonus * (config.combo_multiplier ** (combo - 1))
        reward += combo_reward

    reward += config.survival_bonus

    num_valid = info.get('valid_mask_summary', {}).get('num_valid', 0)
    if num_valid > 10:
        reward += config.valid_moves_bonus * min(num_valid / 50, 1.0)
    elif num_valid < 5:
        reward -= (5 - num_valid) * 0.5

    cleared_rows = len(info.get('cleared_rows', []))
    cleared_cols = len(info.get('cleared_cols', []))
    cells_cleared = cleared_rows * 8 + cleared_cols * 8 - cleared_rows * cleared_cols
    reward += cells_cleared * config.space_cleared_bonus

    action = info.get('action', {})
    if action and action.get('x') is not None:
        x, y = action.get('x', 0), action.get('y', 0)
        center_dist = abs(x - 3.5) + abs(y - 3.5)
        if center_dist < 3:
            reward += config.center_bonus * (3 - center_dist)

    return reward


def compute_terminal_reward_v2(score: int, steps: int, config: RewardV2Config = None) -> float:
    if config is None:
        config = RewardV2Config()

    penalty = -config.game_over_penalty

    survival_factor = min(steps / 100, 1.0)
    penalty *= (1 - survival_factor * 0.5)

    return penalty


def get_reward_summary(reward: float) -> str:
    if reward > 50:
        return "very_high"
    elif reward > 20:
        return "high"
    elif reward > 10:
        return "medium"
    elif reward > 0:
        return "low"
    elif reward > -5:
        return "very_low"
    else:
        return "negative"
