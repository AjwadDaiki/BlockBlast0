
import numpy as np
import json
import os
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

from .pieces import Piece, sample_pieces, get_piece_by_id, PIECES_DATA, PIECE_IDS


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


GRID_SIZE = 8
NUM_PIECES = 3
ACTION_SPACE_SIZE = NUM_PIECES * GRID_SIZE * GRID_SIZE


@dataclass
class StepInfo:
    t: int
    grid: List[List[int]]
    pieces: List[Optional[str]]
    piece_colors: List[Optional[str]]
    action: Dict[str, Any]
    valid_mask_summary: Dict[str, int]
    k_clears: int
    cleared_rows: List[int]
    cleared_cols: List[int]
    combo_streak: int
    score_total: int
    score_delta: int
    reward: float
    q_values_top: Optional[List[Dict]] = None
    decision_tags: Optional[List[str]] = None


class EpisodeRecorder:

    def __init__(self, base_dir: str = "outputs/replays"):
        self.base_dir = Path(base_dir)
        self.current_run = None
        self.current_episode = None
        self.steps: List[Dict] = []
        self.initial_pieces = None

    def start_episode(self, run_name: str, episode_id: int, initial_pieces: List[Piece]):
        self.current_run = run_name
        self.current_episode = episode_id
        self.steps = []
        self.initial_pieces = [(p.id, p.color) for p in initial_pieces]

    def record_step(self, info: StepInfo, q_values_top: List[Dict] = None, decision_tags: List[str] = None):
        step_dict = asdict(info)
        if q_values_top:
            step_dict['q_values_top'] = q_values_top
        if decision_tags:
            step_dict['decision_tags'] = decision_tags
        self.steps.append(step_dict)

    def end_episode(self) -> str:
        if not self.current_run or self.current_episode is None:
            return ""

        out_dir = self.base_dir / self.current_run
        out_dir.mkdir(parents=True, exist_ok=True)

        replay_data = {
            "run_name": self.current_run,
            "episode_id": self.current_episode,
            "total_steps": len(self.steps),
            "final_score": self.steps[-1]["score_total"] if self.steps else 0,
            "initial_pieces": self.initial_pieces,
            "steps": self.steps
        }

        filename = f"episode_{self.current_episode:06d}.json"
        filepath = out_dir / filename

        with open(filepath, 'w') as f:
            json.dump(replay_data, f, cls=NumpyEncoder)


        self._update_index(out_dir, replay_data)

        return str(filepath)

    def _update_index(self, out_dir: Path, replay_data: Dict):
        index_path = out_dir / "index.json"
        if index_path.exists():
            with open(index_path, 'r') as f:
                index = json.load(f)
        else:
            index = {"episodes": []}


        summary = {
            "episode_id": replay_data["episode_id"],
            "total_steps": replay_data["total_steps"],
            "final_score": replay_data["final_score"],
            "best_combo": max((s.get("combo_streak", 0) for s in replay_data["steps"]), default=0),
            "max_clears": max((s.get("k_clears", 0) for s in replay_data["steps"]), default=0),
        }
        index["episodes"].append(summary)

        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2, cls=NumpyEncoder)


class BlockBlastEnv:

    def __init__(self, record_episodes: bool = False, run_name: str = "default"):
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
        self.pieces: List[Optional[Piece]] = [None, None, None]
        self.score = 0
        self.step_count = 0
        self.combo_streak = 0
        self.done = False

        self.record_episodes = record_episodes
        self.run_name = run_name
        self.episode_count = 0
        self.recorder = EpisodeRecorder() if record_episodes else None


        self.last_cleared_rows = []
        self.last_cleared_cols = []
        self.last_k_clears = 0
        self.last_score_delta = 0
        self.last_reward = 0.0

    def reset(self, seed: int = None) -> Tuple[Dict, Dict]:
        if seed is not None:
            np.random.seed(seed)

        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
        self.pieces = sample_pieces(3)
        self.score = 0
        self.step_count = 0
        self.combo_streak = 0
        self.done = False

        self.last_cleared_rows = []
        self.last_cleared_cols = []
        self.last_k_clears = 0
        self.last_score_delta = 0
        self.last_reward = 0.0

        if self.recorder:
            self.recorder.start_episode(self.run_name, self.episode_count, self.pieces)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action_id: int) -> Tuple[Dict, float, bool, bool, Dict]:
        if self.done:
            return self._get_obs(), 0.0, True, False, self._get_info()

        piece_idx, x, y = self.decode_action(action_id)
        piece = self.pieces[piece_idx]


        if piece is None or not self._can_place(piece, x, y):

            self.done = True
            self.last_reward = -10.0
            info = self._get_info(action_id, piece_idx, x, y)
            if self.recorder:
                self._record_step(info)
                self.recorder.end_episode()
                self.episode_count += 1
            return self._get_obs(), self.last_reward, True, False, info


        self._place_piece(piece, x, y)
        self.pieces[piece_idx] = None
        piece_score = piece.num_cells()


        cleared_rows, cleared_cols = self._check_clears()
        k_clears = len(cleared_rows) + len(cleared_cols)

        if k_clears > 0:
            self._clear_lines(cleared_rows, cleared_cols)
            self.combo_streak += 1
        else:
            self.combo_streak = 0


        clear_score = self._calculate_clear_score(piece_score, k_clears)
        self.score += clear_score

        self.last_cleared_rows = cleared_rows
        self.last_cleared_cols = cleared_cols
        self.last_k_clears = k_clears
        self.last_score_delta = clear_score


        if all(p is None for p in self.pieces):
            self.pieces = sample_pieces(3)


        self.done = not self._has_valid_moves()


        self.last_reward = self._compute_reward(clear_score, k_clears, self.done)

        self.step_count += 1

        info = self._get_info(action_id, piece_idx, x, y)

        if self.recorder:
            self._record_step(info)
            if self.done:
                self.recorder.end_episode()
                self.episode_count += 1

        return self._get_obs(), self.last_reward, self.done, False, info

    def decode_action(self, action_id: int) -> Tuple[int, int, int]:
        piece_idx = action_id // 64
        remainder = action_id % 64
        y = remainder // 8
        x = remainder % 8
        return piece_idx, x, y

    def encode_action(self, piece_idx: int, x: int, y: int) -> int:
        return piece_idx * 64 + y * 8 + x

    def get_valid_action_mask(self) -> np.ndarray:
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)

        for piece_idx, piece in enumerate(self.pieces):
            if piece is None:
                continue
            for y in range(GRID_SIZE):
                for x in range(GRID_SIZE):
                    if self._can_place(piece, x, y):
                        action_id = self.encode_action(piece_idx, x, y)
                        mask[action_id] = True

        return mask

    def get_num_valid_actions(self) -> int:
        return int(self.get_valid_action_mask().sum())

    def get_action_info(self, action_id: int) -> Dict:
        piece_idx, x, y = self.decode_action(action_id)
        piece = self.pieces[piece_idx]

        info = {
            "action_id": action_id,
            "piece_idx": piece_idx,
            "x": x,
            "y": y,
            "valid": False,
            "piece_id": None,
            "would_clear": 0,
            "cleared_rows": [],
            "cleared_cols": [],
        }

        if piece is None:
            return info

        info["piece_id"] = piece.id
        info["valid"] = self._can_place(piece, x, y)

        if info["valid"]:

            sim_result = self.simulate_action(action_id)
            info["would_clear"] = sim_result["k_clears"]
            info["cleared_rows"] = sim_result["cleared_rows"]
            info["cleared_cols"] = sim_result["cleared_cols"]

        return info

    def simulate_action(self, action_id: int) -> Dict:
        piece_idx, x, y = self.decode_action(action_id)
        piece = self.pieces[piece_idx]

        if piece is None or not self._can_place(piece, x, y):
            return {"valid": False, "k_clears": 0, "cleared_rows": [], "cleared_cols": [], "score": 0}


        temp_grid = self.grid.copy()


        for dx, dy in piece.shape:
            temp_grid[y + dy, x + dx] = 1


        cleared_rows = [i for i in range(GRID_SIZE) if temp_grid[i, :].all()]
        cleared_cols = [i for i in range(GRID_SIZE) if temp_grid[:, i].all()]
        k_clears = len(cleared_rows) + len(cleared_cols)

        score = self._calculate_clear_score(piece.num_cells(), k_clears)

        return {
            "valid": True,
            "k_clears": k_clears,
            "cleared_rows": cleared_rows,
            "cleared_cols": cleared_cols,
            "score": score,
            "piece_id": piece.id,
        }

    def get_all_valid_actions_with_info(self) -> List[Dict]:
        valid_actions = []
        mask = self.get_valid_action_mask()

        for action_id in np.where(mask)[0]:
            info = self.simulate_action(action_id)
            info["action_id"] = action_id
            piece_idx, x, y = self.decode_action(action_id)
            info["piece_idx"] = piece_idx
            info["x"] = x
            info["y"] = y
            valid_actions.append(info)

        return valid_actions

    def _can_place(self, piece: Piece, x: int, y: int) -> bool:
        for dx, dy in piece.shape:
            px, py = x + dx, y + dy
            if px < 0 or px >= GRID_SIZE or py < 0 or py >= GRID_SIZE:
                return False
            if self.grid[py, px] != 0:
                return False
        return True

    def _place_piece(self, piece: Piece, x: int, y: int):
        for dx, dy in piece.shape:
            self.grid[y + dy, x + dx] = 1

    def _check_clears(self) -> Tuple[List[int], List[int]]:
        cleared_rows = [i for i in range(GRID_SIZE) if self.grid[i, :].all()]
        cleared_cols = [i for i in range(GRID_SIZE) if self.grid[:, i].all()]
        return cleared_rows, cleared_cols

    def _clear_lines(self, rows: List[int], cols: List[int]):
        for r in rows:
            self.grid[r, :] = 0
        for c in cols:
            self.grid[:, c] = 0

    def _calculate_clear_score(self, piece_size: int, k_clears: int) -> int:
        if k_clears == 0:
            return piece_size
        elif k_clears == 1:
            return piece_size + 10
        else:
            base = k_clears * 10
            bonus = (k_clears - 1) * 20
            return piece_size + base + bonus

    def _has_valid_moves(self) -> bool:
        for piece in self.pieces:
            if piece is None:
                continue
            for y in range(GRID_SIZE):
                for x in range(GRID_SIZE):
                    if self._can_place(piece, x, y):
                        return True
        return False

    def _compute_reward(self, score_delta: int, k_clears: int, done: bool) -> float:
        reward = score_delta / 10.0

        if k_clears >= 2:
            reward += k_clears * 2.0

        if done:
            reward -= 5.0

        return reward

    def _get_obs(self) -> Dict:

        piece_grids = []
        for piece in self.pieces:
            grid = np.zeros((5, 5), dtype=np.float32)
            if piece is not None:
                for dx, dy in piece.shape:
                    if dx < 5 and dy < 5:
                        grid[dy, dx] = 1.0
            piece_grids.append(grid)

        return {
            "grid": self.grid.astype(np.float32),
            "pieces": np.stack(piece_grids),
            "piece_ids": [p.id if p else None for p in self.pieces],
        }

    def _get_info(self, action_id: int = None, piece_idx: int = None, x: int = None, y: int = None) -> Dict:
        return {
            "t": self.step_count,
            "grid": self.grid.tolist(),
            "pieces": [p.id if p else None for p in self.pieces],
            "piece_colors": [p.color if p else None for p in self.pieces],
            "action": {
                "piece_i": piece_idx,
                "x": x,
                "y": y,
                "action_id": action_id
            } if action_id is not None else None,
            "valid_mask_summary": {"num_valid": self.get_num_valid_actions()},
            "k_clears": self.last_k_clears,
            "cleared_rows": self.last_cleared_rows,
            "cleared_cols": self.last_cleared_cols,
            "combo_streak": self.combo_streak,
            "score_total": self.score,
            "score_delta": self.last_score_delta,
            "reward": self.last_reward,
        }

    def _record_step(self, info: Dict):
        if self.recorder:
            step_info = StepInfo(**{k: v for k, v in info.items()
                                    if k in StepInfo.__dataclass_fields__})
            self.recorder.record_step(step_info)

    def render(self, mode: str = "ansi") -> Optional[str]:
        if mode == "ansi":
            return self._render_ansi()
        return None

    def _render_ansi(self) -> str:
        lines = []
        lines.append(f"Score: {self.score}  Step: {self.step_count}  Combo: {self.combo_streak}")
        lines.append("+" + "-" * (GRID_SIZE * 2 + 1) + "+")

        for y in range(GRID_SIZE):
            row = "| "
            for x in range(GRID_SIZE):
                row += "█ " if self.grid[y, x] else ". "
            row += "|"
            lines.append(row)

        lines.append("+" + "-" * (GRID_SIZE * 2 + 1) + "+")


        lines.append("Pieces:")
        for i, piece in enumerate(self.pieces):
            if piece:
                lines.append(f"  [{i}] {piece.id} ({piece.color})")
            else:
                lines.append(f"  [{i}] (used)")

        lines.append(f"Valid moves: {self.get_num_valid_actions()}")

        return "\n".join(lines)

    def get_state_for_nn(self) -> np.ndarray:

        grid_flat = self.grid.flatten().astype(np.float32)


        piece_flat = []
        for piece in self.pieces:
            grid = np.zeros(25, dtype=np.float32)
            if piece is not None:
                for dx, dy in piece.shape:
                    if dx < 5 and dy < 5:
                        grid[dy * 5 + dx] = 1.0
            piece_flat.extend(grid)

        return np.concatenate([grid_flat, np.array(piece_flat)])

    @property
    def state_dim(self) -> int:
        return 64 + 75

    @property
    def action_dim(self) -> int:
        return ACTION_SPACE_SIZE
