
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
class DQNConfig:

    hidden_dims: List[int] = None


    learning_rate: float = 1e-4
    gamma: float = 0.99
    batch_size: int = 64
    buffer_size: int = 100000


    target_update_freq: int = 1000
    tau: float = 1.0


    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.9995


    double_dqn: bool = True

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 256, 128]


class QNetwork(nn.Module):

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int]):
        super().__init__()

        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, action_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class DQNAgent:

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 config: DQNConfig = None,
                 device: str = None):

        self.config = config or DQNConfig()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.name = "DQN"


        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device


        self.q_network = QNetwork(
            state_dim, action_dim, self.config.hidden_dims
        ).to(self.device)

        self.target_network = QNetwork(
            state_dim, action_dim, self.config.hidden_dims
        ).to(self.device)

        self.target_network.load_state_dict(self.q_network.state_dict())


        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=self.config.learning_rate
        )


        self.replay_buffer = ReplayBuffer(self.config.buffer_size)


        self.epsilon = self.config.epsilon_start
        self.train_steps = 0
        self.episodes = 0

    def select_action(self,
                      state: np.ndarray,
                      valid_mask: np.ndarray,
                      training: bool = True) -> int:


        if training and np.random.random() < self.epsilon:

            valid_actions = np.where(valid_mask)[0]
            if len(valid_actions) == 0:
                return 0
            return np.random.choice(valid_actions)


        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor).squeeze(0)


            mask_tensor = torch.BoolTensor(valid_mask).to(self.device)
            q_values[~mask_tensor] = float('-inf')

            return q_values.argmax().item()

    def get_q_values(self, state: np.ndarray, valid_mask: np.ndarray = None) -> np.ndarray:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
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
        self.replay_buffer.push(state, action, reward, next_state, done, valid_mask)

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


        loss = F.mse_loss(current_q, target_q)


        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()


        self.train_steps += 1
        if self.train_steps % self.config.target_update_freq == 0:
            self._update_target_network()

        return loss.item()

    def _update_target_network(self):
        if self.config.tau >= 1.0:

            self.target_network.load_state_dict(self.q_network.state_dict())
        else:

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

    def reset(self):
        pass

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_steps': self.train_steps,
            'episodes': self.episodes,
            'config': self.config,
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.train_steps = checkpoint['train_steps']
        self.episodes = checkpoint['episodes']

    def get_stats(self) -> Dict:
        return {
            "epsilon": self.epsilon,
            "train_steps": self.train_steps,
            "episodes": self.episodes,
            "buffer_size": len(self.replay_buffer),
        }
