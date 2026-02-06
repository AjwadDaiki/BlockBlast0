
import numpy as np
import random
from collections import deque
from typing import NamedTuple
import torch


class Transition(NamedTuple):
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    valid_mask: np.ndarray


class Batch(NamedTuple):
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor
    valid_masks: torch.Tensor


class ReplayBuffer:

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
        self.buffer.append(Transition(
            state=state.copy(),
            action=action,
            reward=reward,
            next_state=next_state.copy(),
            done=done,
            valid_mask=valid_mask.copy()
        ))

    def sample(self, batch_size: int, device: str = 'cpu') -> Batch:
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
        return len(self.buffer) >= batch_size
