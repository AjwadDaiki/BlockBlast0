"""
Experience Replay Buffer for DQN
"""

import numpy as np
import random
from collections import deque
from typing import Tuple, List, NamedTuple
import torch


class Transition(NamedTuple):
    """Single transition in replay buffer"""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    valid_mask: np.ndarray


class Batch(NamedTuple):
    """Batch of transitions for training"""
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor
    valid_masks: torch.Tensor


class ReplayBuffer:
    """Experience replay buffer for DQN training"""

    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self,
             state: np.ndarray,
             action: int,
             reward: float,
             next_state: np.ndarray,
             done: bool,
             valid_mask: np.ndarray):
        """Add transition to buffer"""
        self.buffer.append(Transition(
            state=state.copy(),
            action=action,
            reward=reward,
            next_state=next_state.copy(),
            done=done,
            valid_mask=valid_mask.copy()
        ))

    def sample(self, batch_size: int, device: str = 'cpu') -> Batch:
        """Sample random batch of transitions"""
        transitions = random.sample(self.buffer, min(batch_size, len(self.buffer)))

        states = np.array([t.state for t in transitions])
        actions = np.array([t.action for t in transitions])
        rewards = np.array([t.reward for t in transitions])
        next_states = np.array([t.next_state for t in transitions])
        dones = np.array([t.done for t in transitions])
        valid_masks = np.array([t.valid_mask for t in transitions])

        return Batch(
            states=torch.FloatTensor(states).to(device),
            actions=torch.LongTensor(actions).to(device),
            rewards=torch.FloatTensor(rewards).to(device),
            next_states=torch.FloatTensor(next_states).to(device),
            dones=torch.BoolTensor(dones).to(device),
            valid_masks=torch.BoolTensor(valid_masks).to(device)
        )

    def __len__(self) -> int:
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples"""
        return len(self.buffer) >= batch_size


class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer"""

    def __init__(self, capacity: int = 100000, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.position = 0

    def push(self,
             state: np.ndarray,
             action: int,
             reward: float,
             next_state: np.ndarray,
             done: bool,
             valid_mask: np.ndarray):
        """Add transition with max priority"""
        max_priority = max(self.priorities) if self.priorities else 1.0

        transition = Transition(
            state=state.copy(),
            action=action,
            reward=reward,
            next_state=next_state.copy(),
            done=done,
            valid_mask=valid_mask.copy()
        )

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(max_priority)
        else:
            self.buffer[self.position] = transition
            self.priorities[self.position] = max_priority

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4, device: str = 'cpu') -> Tuple[Batch, np.ndarray, List[int]]:
        """Sample batch with importance sampling weights"""
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)),
                                   p=probs, replace=False)

        # Importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()

        transitions = [self.buffer[i] for i in indices]

        states = np.array([t.state for t in transitions])
        actions = np.array([t.action for t in transitions])
        rewards = np.array([t.reward for t in transitions])
        next_states = np.array([t.next_state for t in transitions])
        dones = np.array([t.done for t in transitions])
        valid_masks = np.array([t.valid_mask for t in transitions])

        batch = Batch(
            states=torch.FloatTensor(states).to(device),
            actions=torch.LongTensor(actions).to(device),
            rewards=torch.FloatTensor(rewards).to(device),
            next_states=torch.FloatTensor(next_states).to(device),
            dones=torch.BoolTensor(dones).to(device),
            valid_masks=torch.BoolTensor(valid_masks).to(device)
        )

        return batch, torch.FloatTensor(weights).to(device), list(indices)

    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities for sampled transitions"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6

    def __len__(self) -> int:
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        return len(self.buffer) >= batch_size
