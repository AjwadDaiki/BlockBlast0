
import json
from pathlib import Path
from typing import Dict, List, Optional
from .renderer import BlockBlastRenderer

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False


def load_replay(path: str) -> Dict:
    with open(path, 'r') as f:
        return json.load(f)


def render_replay(replay_path: str,
                  out_dir: str,
                  fps: int = 10,
                  to_mp4: bool = True,
                  to_gif: bool = False,
                  cell_size: int = 50,
                  step_duration: float = 0.3) -> Dict:
    replay = load_replay(replay_path)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    renderer = BlockBlastRenderer(cell_size=cell_size)

    frames_dir = out_path / "frames"
    frames_dir.mkdir(exist_ok=True)

    steps = replay.get('steps', [])
    frame_paths = []
    images = []


    frames_per_step = max(1, int(step_duration * fps))

    for i, step in enumerate(steps):

        img = renderer.render_frame(
            step,
            q_values_top=step.get('q_values_top'),
            decision_tags=step.get('decision_tags')
        )


        frame_path = frames_dir / f"frame_{i:06d}.png"
        img.save(str(frame_path))
        frame_paths.append(str(frame_path))

        if to_mp4 or to_gif:
            import numpy as np
            img_array = np.array(img)

            for _ in range(frames_per_step):
                images.append(img_array)

    result = {
        "frames_dir": str(frames_dir),
        "num_frames": len(frame_paths),
        "frame_paths": frame_paths,
    }


    if HAS_IMAGEIO and (to_mp4 or to_gif):
        if to_mp4:
            mp4_path = out_path / "episode.mp4"
            try:
                imageio.mimsave(str(mp4_path), images, fps=fps)
                result["mp4_path"] = str(mp4_path)
            except Exception as e:
                print(f"Warning: Could not create MP4: {e}")
                print("Try: pip install imageio-ffmpeg")

        if to_gif:
            gif_path = out_path / "episode.gif"
            try:

                subsample = max(1, len(images) // 100)
                gif_images = images[::subsample]
                imageio.mimsave(str(gif_path), gif_images, fps=fps//2, loop=0)
                result["gif_path"] = str(gif_path)
            except Exception as e:
                print(f"Warning: Could not create GIF: {e}")

    return result


def render_highlight_clip(replay_path: str,
                          start_step: int,
                          end_step: int,
                          out_path: str,
                          fps: int = 15,
                          caption: str = None) -> str:
    replay = load_replay(replay_path)
    steps = replay.get('steps', [])[start_step:end_step + 1]

    if not steps:
        return ""

    renderer = BlockBlastRenderer(cell_size=40, width=600, height=450)
    images = []

    for step in steps:
        img = renderer.render_frame(
            step,
            q_values_top=step.get('q_values_top'),
            decision_tags=step.get('decision_tags')
        )


        if caption:
            from PIL import ImageDraw
            draw = ImageDraw.Draw(img)
            draw.rectangle([0, 0, 600, 30], fill=(0, 0, 0, 180))
            draw.text((300, 15), caption, fill=(255, 255, 100), anchor="mm",
                      font=renderer.font_medium)

        import numpy as np
        img_array = np.array(img)

        for _ in range(3):
            images.append(img_array)

    if HAS_IMAGEIO and images:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            imageio.mimsave(out_path, images, fps=fps, loop=0)
            return out_path
        except Exception as e:
            print(f"Warning: Could not create highlight clip: {e}")

    return ""


def get_replay_summary(replay_path: str) -> Dict:
    replay = load_replay(replay_path)
    steps = replay.get('steps', [])

    if not steps:
        return {}

    return {
        "episode_id": replay.get('episode_id'),
        "total_steps": len(steps),
        "final_score": steps[-1].get('score_total', 0) if steps else 0,
        "max_combo": max((s.get('combo_streak', 0) for s in steps), default=0),
        "total_clears": sum(s.get('k_clears', 0) for s in steps),
        "max_single_clear": max((s.get('k_clears', 0) for s in steps), default=0),
    }
