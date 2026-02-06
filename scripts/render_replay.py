
import argparse
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from render.replay import render_replay, get_replay_summary


def main():
    parser = argparse.ArgumentParser(description='Render Block Blast replay')
    parser.add_argument('--replay', type=str, required=True, help='Path to replay JSON')
    parser.add_argument('--out', type=str, required=True, help='Output directory')
    parser.add_argument('--fps', type=int, default=10, help='Frames per second')
    parser.add_argument('--to_mp4', type=int, default=1, help='Export MP4 (1=yes, 0=no)')
    parser.add_argument('--to_gif', type=int, default=0, help='Export GIF (1=yes, 0=no)')
    parser.add_argument('--cell_size', type=int, default=50, help='Cell size in pixels')

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"BLOCK BLAST - RENDER REPLAY")
    print(f"{'='*60}")
    print(f"Replay: {args.replay}")
    print(f"Output: {args.out}")
    print(f"{'='*60}\n")


    summary = get_replay_summary(args.replay)
    print(f"Episode {summary.get('episode_id', '?')}")
    print(f"  Steps: {summary.get('total_steps', 0)}")
    print(f"  Final score: {summary.get('final_score', 0)}")
    print(f"  Max combo: {summary.get('max_combo', 0)}")
    print(f"  Total clears: {summary.get('total_clears', 0)}")
    print()


    print("Rendering frames...")
    result = render_replay(
        args.replay,
        args.out,
        fps=args.fps,
        to_mp4=bool(args.to_mp4),
        to_gif=bool(args.to_gif),
        cell_size=args.cell_size
    )

    print(f"\nGenerated {result['num_frames']} frames")
    print(f"Frames directory: {result['frames_dir']}")

    if 'mp4_path' in result:
        print(f"MP4: {result['mp4_path']}")

    if 'gif_path' in result:
        print(f"GIF: {result['gif_path']}")

    print("\nDone!")


if __name__ == "__main__":
    main()
