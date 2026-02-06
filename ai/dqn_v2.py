
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

from .replay_buffer import ReplayBuffer, Batch


@dataclass
class DQNv2Config:

    hidden_dims: List[int] = None


    learning_rate: float = 3e-5
    gamma: float = 0.99
    batch_size: int = 128
    buffer_size: int = 200000


    target_update_freq: int = 100
    tau: float = 0.005


    epsilon_start: float = 1.0
    epsilon_end: float = 0.02
    epsilon_decay: float = 0.9998


    double_dqn: bool = True


    dueling: bool = True

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [512, 512, 256]


class DuelingQNetwork(nn.Module):

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int]):
        super().__init__()


        self.feature_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.1),
        )


        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[2], 1)
        )


        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[2], action_dim)
        )


        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.feature_layers(state)

        value = self.value_stream(features)
        advantage = self.advantage_stream(features)


        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q_values


class StandardQNetwork(nn.Module):

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int]):
        super().__init__()

        layers = []
        prev_dim = state_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            if i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, action_dim))
        self.network = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class DQNv2Agent:

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 config: DQNv2Config = None,
                 device: str = None):

        self.config = config or DQNv2Config()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.name = "DQN_v2"


        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device


        NetworkClass = DuelingQNetwork if self.config.dueling else StandardQNetwork

        self.q_network = NetworkClass(
            state_dim, action_dim, self.config.hidden_dims
        ).to(self.device)

        self.target_network = NetworkClass(
            state_dim, action_dim, self.config.hidden_dims
        ).to(self.device)

        self.target_network.load_state_dict(self.q_network.state_dict())


        self.optimizer = optim.AdamW(
            self.q_network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=1e-5
        )


        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=2000, gamma=0.95
        )


        self.replay_buffer = ReplayBuffer(self.config.buffer_size)


        self.epsilon = self.config.epsilon_start
        self.train_steps = 0
        self.episodes = 0


        self.state_mean = None
        self.state_std = None
        self.state_count = 0

    def _update_state_stats(self, state: np.ndarray):
        if self.state_mean is None:
            self.state_mean = np.zeros_like(state, dtype=np.float64)
            self.state_std = np.ones_like(state, dtype=np.float64)

        self.state_count += 1
        delta = state - self.state_mean
        self.state_mean += delta / self.state_count
        delta2 = state - self.state_mean
        self.state_std += delta * delta2

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        if self.state_mean is None or self.state_count < 100:
            return state

        std = np.sqrt(self.state_std / self.state_count + 1e-8)
        return (state - self.state_mean) / std

    def select_action(self,
                      state: np.ndarray,
                      valid_mask: np.ndarray,
                      training: bool = True) -> int:

        if training:
            self._update_state_stats(state)


        if training and np.random.random() < self.epsilon:
            valid_actions = np.where(valid_mask)[0]
            if len(valid_actions) == 0:
                return 0
            return np.random.choice(valid_actions)


        with torch.no_grad():
            norm_state = self._normalize_state(state)
            state_tensor = torch.FloatTensor(norm_state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor).squeeze(0)


            mask_tensor = torch.BoolTensor(valid_mask).to(self.device)
            q_values[~mask_tensor] = float('-inf')

            return q_values.argmax().item()

    def get_q_values(self, state: np.ndarray, valid_mask: np.ndarray = None) -> np.ndarray:
        with torch.no_grad():
            norm_state = self._normalize_state(state)
            state_tensor = torch.FloatTensor(norm_state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor).squeeze(0).cpu().numpy()

            if valid_mask is not None:
                q_values[~valid_mask] = float('-inf')

            return q_values

    def get_top_actions(self,
                        state: np.ndarray,
                        valid_mask: np.ndarray,
                        top_k: int = 5) -> List[Dict]:
        q_values = self.get_q_values(state, valid_mask)

        valid_indices = np.where(valid_mask)[0]
        valid_q = [(i, q_values[i]) for i in valid_indices]
        valid_q.sort(key=lambda x: x[1], reverse=True)

        results = []
        for i, (action_id, q) in enumerate(valid_q[:top_k]):
            piece_idx = action_id // 64
            remainder = action_id % 64
            y = remainder // 8
            x = remainder % 8

            results.append({
                "action_id": int(action_id),
                "q": float(q),
                "desc": f"p{piece_idx}@({x},{y})",
                "piece_idx": piece_idx,
                "x": x,
                "y": y
            })

        return results

    def store_transition(self,
                         state: np.ndarray,
                         action: int,
                         reward: float,
                         next_state: np.ndarray,
                         done: bool,
                         valid_mask: np.ndarray):

        norm_state = self._normalize_state(state)
        norm_next = self._normalize_state(next_state)
        self.replay_buffer.push(norm_state, action, reward, norm_next, done, valid_mask)

    def update(self) -> Optional[float]:
        if not self.replay_buffer.is_ready(self.config.batch_size):
            return None


        batch = self.replay_buffer.sample(self.config.batch_size, self.device)


        current_q = self.q_network(batch.states)
        current_q = current_q.gather(1, batch.actions.unsqueeze(1)).squeeze(1)


        with torch.no_grad():
            if self.config.double_dqn:

                next_q_online = self.q_network(batch.next_states)

                next_q_online[~batch.valid_masks] = float('-inf')
                next_actions = next_q_online.argmax(1)


                next_q_target = self.target_network(batch.next_states)
                next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                next_q = self.target_network(batch.next_states)
                next_q[~batch.valid_masks] = float('-inf')
                next_q = next_q.max(1)[0]


            target_q = batch.rewards + self.config.gamma * next_q * (~batch.dones)


        loss = F.smooth_l1_loss(current_q, target_q)


        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()


        self.train_steps += 1
        if self.train_steps % self.config.target_update_freq == 0:
            self._soft_update_target()

        return loss.item()

    def _soft_update_target(self):
        for target_param, param in zip(
            self.target_network.parameters(),
            self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.config.tau * param.data +
                (1 - self.config.tau) * target_param.data
            )

    def decay_epsilon(self):
        self.epsilon = max(
            self.config.epsilon_end,
            self.epsilon * self.config.epsilon_decay
        )

    def end_episode(self):
        self.episodes += 1
        self.decay_epsilon()
        self.scheduler.step()

    def reset(self):
        pass

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'train_steps': self.train_steps,
            'episodes': self.episodes,
            'config': self.config,
            'state_mean': self.state_mean,
            'state_std': self.state_std,
            'state_count': self.state_count,
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.epsilon = checkpoint['epsilon']
        self.train_steps = checkpoint['train_steps']
        self.episodes = checkpoint['episodes']
        if 'state_mean' in checkpoint:
            self.state_mean = checkpoint['state_mean']
            self.state_std = checkpoint['state_std']
            self.state_count = checkpoint['state_count']

    def get_stats(self) -> Dict:
        return {
            "epsilon": self.epsilon,
            "train_steps": self.train_steps,
            "episodes": self.episodes,
            "buffer_size": len(self.replay_buffer),
            "lr": self.optimizer.param_groups[0]['lr'],
        }
