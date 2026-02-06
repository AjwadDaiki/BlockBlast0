"""
Highlight Detection for Block Blast
Automatically detect interesting moments for YouTube
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from .replay import load_replay, render_highlight_clip


@dataclass
class Highlight:
    episode_id: int
    type: str
    start_step: int
    end_step: int
    before_score: int
    after_score: int
    short_caption: str
    metadata: Dict = None

    def to_dict(self) -> Dict:
        return asdict(self)


class HighlightDetector:
    """Detects interesting moments in Block Blast episodes"""

    def __init__(self,
                 min_big_combo: int = 3,
                 near_death_threshold: int = 3,
                 survival_steps: int = 5):
        self.min_big_combo = min_big_combo
        self.near_death_threshold = near_death_threshold
        self.survival_steps = survival_steps

    def analyze_episode(self, replay: Dict) -> List[Highlight]:
        """Analyze replay and return all detected highlights"""
        highlights = []

        steps = replay.get('steps', [])
        episode_id = replay.get('episode_id', 0)

        if not steps:
            return highlights

        # Detect each type
        highlights.extend(self._detect_immediate_clear_refused(steps, episode_id))
        highlights.extend(self._detect_big_combo(steps, episode_id))
        highlights.extend(self._detect_near_death_save(steps, episode_id))

        return highlights

    def _detect_immediate_clear_refused(self, steps: List[Dict], episode_id: int) -> List[Highlight]:
        """
        Detect when agent refuses immediate clear for a better setup.

        Conditions:
        - At step t, there's at least one action that would clear (k>=1)
        - Agent chose action with k==0 or lower k
        - At step t+1 or t+2, agent achieves higher clear (k>=2)
        """
        highlights = []

        for i, step in enumerate(steps[:-2]):
            # Check if there was a clear option available but not taken
            k_chosen = step.get('k_clears', 0)

            # Look for "q_values_top" to see if there were clear options
            q_values = step.get('q_values_top', [])
            had_clear_option = False

            for qv in q_values:
                if qv.get('would_clear', 0) > k_chosen:
                    had_clear_option = True
                    break

            if not had_clear_option or k_chosen > 0:
                continue

            # Check next steps for bigger clear
            for j in range(1, min(3, len(steps) - i)):
                next_step = steps[i + j]
                next_k = next_step.get('k_clears', 0)

                if next_k >= 2:
                    highlight = Highlight(
                        episode_id=episode_id,
                        type="IMMEDIATE_CLEAR_REFUSED",
                        start_step=max(0, i - 1),
                        end_step=min(len(steps) - 1, i + j + 1),
                        before_score=step.get('score_total', 0),
                        after_score=next_step.get('score_total', 0),
                        short_caption=f"REFUSE CLEAR → {next_k}x CLEAR!",
                        metadata={"setup_steps": j, "final_clears": next_k}
                    )
                    highlights.append(highlight)
                    break

        return highlights

    def _detect_big_combo(self, steps: List[Dict], episode_id: int) -> List[Highlight]:
        """Detect big combos (k >= min_big_combo) or combo streak records"""
        highlights = []
        max_combo_seen = 0

        for i, step in enumerate(steps):
            k_clears = step.get('k_clears', 0)
            combo_streak = step.get('combo_streak', 0)

            is_big_clear = k_clears >= self.min_big_combo
            is_combo_record = combo_streak > max_combo_seen and combo_streak >= 2

            if is_big_clear or is_combo_record:
                if is_combo_record:
                    max_combo_seen = combo_streak
                    caption = f"COMBO x{combo_streak}!"
                else:
                    caption = f"{k_clears}x MULTI-CLEAR!"

                highlight = Highlight(
                    episode_id=episode_id,
                    type="BIG_COMBO",
                    start_step=max(0, i - 2),
                    end_step=min(len(steps) - 1, i + 2),
                    before_score=steps[max(0, i - 1)].get('score_total', 0),
                    after_score=step.get('score_total', 0),
                    short_caption=caption,
                    metadata={"k_clears": k_clears, "combo_streak": combo_streak}
                )
                highlights.append(highlight)

        return highlights

    def _detect_near_death_save(self, steps: List[Dict], episode_id: int) -> List[Highlight]:
        """Detect near-death situations where agent survives"""
        highlights = []

        for i, step in enumerate(steps[:-self.survival_steps]):
            num_valid = step.get('valid_mask_summary', {}).get('num_valid', 100)

            if num_valid <= self.near_death_threshold:
                # Check if survived for enough steps
                survived = True
                for j in range(1, self.survival_steps + 1):
                    if i + j >= len(steps):
                        survived = False
                        break

                if survived:
                    highlight = Highlight(
                        episode_id=episode_id,
                        type="NEAR_DEATH_SAVE",
                        start_step=max(0, i - 1),
                        end_step=min(len(steps) - 1, i + self.survival_steps),
                        before_score=step.get('score_total', 0),
                        after_score=steps[min(len(steps) - 1, i + self.survival_steps)].get('score_total', 0),
                        short_caption=f"NEAR DEATH! Only {num_valid} moves left!",
                        metadata={"num_valid_at_crisis": num_valid}
                    )
                    highlights.append(highlight)

        return highlights

    def analyze_run(self, replays_dir: str) -> Dict:
        """
        Analyze all episodes in a run directory.

        Returns dict with:
        - all_highlights: List of all highlights
        - best_episode: Episode with highest score
        - highlight_counts: Count by type
        """
        replays_path = Path(replays_dir)
        all_highlights = []
        best_score = 0
        best_episode = None

        # Load index if exists
        index_path = replays_path / "index.json"
        if index_path.exists():
            with open(index_path) as f:
                index = json.load(f)

            # Find best episode
            for ep in index.get('episodes', []):
                if ep.get('final_score', 0) > best_score:
                    best_score = ep['final_score']
                    best_episode = ep['episode_id']

        # Analyze each replay
        for replay_file in sorted(replays_path.glob("episode_*.json")):
            replay = load_replay(str(replay_file))
            highlights = self.analyze_episode(replay)
            all_highlights.extend(highlights)

            # Track best
            if replay.get('final_score', 0) > best_score:
                best_score = replay['final_score']
                best_episode = replay['episode_id']

        # Add best episode highlight
        if best_episode is not None:
            best_highlight = Highlight(
                episode_id=best_episode,
                type="BEST_EPISODE",
                start_step=0,
                end_step=-1,  # Full episode
                before_score=0,
                after_score=best_score,
                short_caption=f"BEST SCORE: {best_score}!",
                metadata={"is_best": True}
            )
            all_highlights.append(best_highlight)

        # Count by type
        counts = {}
        for h in all_highlights:
            counts[h.type] = counts.get(h.type, 0) + 1

        return {
            "all_highlights": [h.to_dict() for h in all_highlights],
            "best_episode": best_episode,
            "best_score": best_score,
            "highlight_counts": counts,
            "total_highlights": len(all_highlights)
        }


def export_highlights(replays_dir: str,
                      out_dir: str,
                      max_highlights: int = 20,
                      types: List[str] = None) -> List[str]:
    """
    Export highlights as GIF clips.

    Args:
        replays_dir: Directory containing replay JSONs
        out_dir: Output directory for highlight clips
        max_highlights: Maximum number of highlights to export
        types: Filter by highlight types (None = all)

    Returns:
        List of generated file paths
    """
    detector = HighlightDetector()
    analysis = detector.analyze_run(replays_dir)

    highlights = analysis['all_highlights']

    # Filter by type
    if types:
        highlights = [h for h in highlights if h['type'] in types]

    # Limit count
    highlights = highlights[:max_highlights]

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    generated_files = []

    for i, h in enumerate(highlights):
        episode_id = h['episode_id']
        replay_path = Path(replays_dir) / f"episode_{episode_id:06d}.json"

        if not replay_path.exists():
            continue

        # Generate clip
        clip_path = out_path / f"highlight_{i:04d}_{h['type'].lower()}.gif"
        result = render_highlight_clip(
            str(replay_path),
            h['start_step'],
            h['end_step'] if h['end_step'] >= 0 else 9999,
            str(clip_path),
            caption=h['short_caption']
        )

        if result:
            generated_files.append(result)

            # Save metadata
            meta_path = out_path / f"highlight_{i:04d}_{h['type'].lower()}.json"
            with open(meta_path, 'w') as f:
                json.dump(h, f, indent=2)

    return generated_files
