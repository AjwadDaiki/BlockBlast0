"""
Reward Shaping for Block Blast RL
Configurable reward functions
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class RewardConfig:
    """Configuration for reward shaping"""
    # Base rewards
    score_scale: float = 0.1          # Scale factor for score delta
    clear_bonus: float = 5.0          # Bonus per line/column cleared
    multi_clear_bonus: float = 10.0   # Extra bonus for clearing 2+ lines
    combo_bonus: float = 3.0          # Bonus per combo streak level

    # Penalties
    game_over_penalty: float = 10.0   # Penalty for game over
    invalid_action_penalty: float = 10.0  # Penalty for invalid action

    # Shaping bonuses
    hole_penalty: float = 0.5         # Penalty per hole created
    fill_bonus: float = 0.01          # Small bonus for filling cells
    valid_moves_bonus: float = 0.1    # Bonus for maintaining valid moves


def compute_reward(info: Dict, config: RewardConfig = None, prev_info: Dict = None) -> float:
    """
    Compute shaped reward from step info.

    Args:
        info: Current step info dict
        config: Reward configuration
        prev_info: Previous step info (for delta calculations)

    Returns:
        Shaped reward value
    """
    if config is None:
        config = RewardConfig()

    reward = 0.0

    # Score delta reward
    score_delta = info.get('score_delta', 0)
    reward += score_delta * config.score_scale

    # Clear bonuses
    k_clears = info.get('k_clears', 0)
    if k_clears > 0:
        reward += k_clears * config.clear_bonus

        # Multi-clear bonus
        if k_clears >= 2:
            reward += (k_clears - 1) * config.multi_clear_bonus

    # Combo bonus
    combo = info.get('combo_streak', 0)
    if combo > 1:
        reward += combo * config.combo_bonus

    # Penalty for low valid moves (risky situation)
    num_valid = info.get('valid_mask_summary', {}).get('num_valid', 0)
    if num_valid < 5:
        reward -= (5 - num_valid) * 0.5

    # Small bonus for maintaining options
    if num_valid > 20:
        reward += config.valid_moves_bonus

    return reward


def compute_terminal_reward(won: bool, score: int, config: RewardConfig = None) -> float:
    """Compute reward for terminal state"""
    if config is None:
        config = RewardConfig()

    if won:
        return score * config.score_scale
    else:
        return -config.game_over_penalty


class RewardTracker:
    """Track reward statistics during training"""

    def __init__(self):
        self.episode_rewards = []
        self.current_episode = 0.0

    def add(self, reward: float):
        self.current_episode += reward

    def end_episode(self) -> float:
        total = self.current_episode
        self.episode_rewards.append(total)
        self.current_episode = 0.0
        return total

    def get_mean(self, last_n: int = 100) -> float:
        if not self.episode_rewards:
            return 0.0
        recent = self.episode_rewards[-last_n:]
        return sum(recent) / len(recent)

    def get_stats(self) -> Dict:
        if not self.episode_rewards:
            return {"mean": 0, "max": 0, "min": 0, "count": 0}

        return {
            "mean": sum(self.episode_rewards) / len(self.episode_rewards),
            "max": max(self.episode_rewards),
            "min": min(self.episode_rewards),
            "count": len(self.episode_rewards),
            "recent_mean": self.get_mean(100)
        }
