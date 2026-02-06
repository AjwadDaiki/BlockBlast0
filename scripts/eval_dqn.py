
import argparse
import json
from pathlib import Path
from typing import Dict, List
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from game.blockblast_env import BlockBlastEnv
from ai.dqn import DQNAgent, DQNConfig
from render.replay import render_replay
from render.highlights import HighlightDetector, export_highlights


def evaluate(args):
    print(f"\n{'='*60}")
    print(f"BLOCK BLAST - DQN EVALUATION")
    print(f"{'='*60}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Episodes: {args.episodes}")
    print(f"Run name: {args.run_name}")
    print(f"{'='*60}\n")


    out_base = Path("outputs")
    replays_dir = out_base / "replays" / args.run_name
    videos_dir = out_base / "videos" / args.run_name
    highlights_dir = out_base / "highlights" / args.run_name

    for d in [replays_dir, videos_dir, highlights_dir]:
        d.mkdir(parents=True, exist_ok=True)


    env = BlockBlastEnv(record_episodes=True, run_name=args.run_name)


    config = DQNConfig()
    agent = DQNAgent(env.state_dim, env.action_dim, config)
    agent.load(args.checkpoint)
    agent.epsilon = 0.0

    print(f"Loaded agent: {agent.get_stats()}")


    scores = []
    steps_list = []
    best_score = 0
    best_episode = 0

    print("\nEvaluating...")
    print(f"{'Episode':>8} {'Score':>8} {'Steps':>8} {'Best':>8}")
    print("-" * 40)

    for episode in range(args.episodes):
        obs, info = env.reset()
        state = env.get_state_for_nn()
        done = False
        step = 0


        while not done:
            valid_mask = env.get_valid_action_mask()
            action = agent.select_action(state, valid_mask, training=False)


            q_values_top = agent.get_top_actions(state, valid_mask, top_k=5)

            obs, reward, done, truncated, info = env.step(action)


            if env.recorder and env.recorder.steps:
                env.recorder.steps[-1]['q_values_top'] = q_values_top

            state = env.get_state_for_nn()
            step += 1

            if step > 1000:
                break

        scores.append(env.score)
        steps_list.append(step)

        if env.score > best_score:
            best_score = env.score
            best_episode = episode

        if (episode + 1) % 10 == 0:
            print(f"{episode+1:>8} {env.score:>8} {step:>8} {best_score:>8}")


    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Episodes: {args.episodes}")
    print(f"Mean score: {np.mean(scores):.1f} (+/- {np.std(scores):.1f})")
    print(f"Max score: {max(scores)}")
    print(f"Min score: {min(scores)}")
    print(f"Median score: {np.median(scores):.1f}")
    print(f"Mean steps: {np.mean(steps_list):.1f}")
    print(f"Best episode: {best_episode}")
    print(f"{'='*60}\n")


    results = {
        'checkpoint': args.checkpoint,
        'episodes': args.episodes,
        'scores': scores,
        'steps': steps_list,
        'mean_score': float(np.mean(scores)),
        'std_score': float(np.std(scores)),
        'max_score': int(max(scores)),
        'min_score': int(min(scores)),
        'median_score': float(np.median(scores)),
        'best_episode': best_episode
    }

    results_path = out_base / "logs" / args.run_name / "eval_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {results_path}")


    if args.render_best:
        print("\nRendering best episode...")
        best_replay = replays_dir / f"episode_{best_episode:06d}.json"
        if best_replay.exists():
            try:
                result = render_replay(
                    str(best_replay),
                    str(videos_dir / "best"),
                    to_mp4=True,
                    to_gif=True
                )
                print(f"  Video saved: {result.get('mp4_path', 'N/A')}")
            except Exception as e:
                print(f"  Warning: Could not render: {e}")


    if args.export_highlights:
        print("\nDetecting highlights...")
        detector = HighlightDetector()
        analysis = detector.analyze_run(str(replays_dir))

        print(f"  Found {analysis['total_highlights']} highlights")
        for htype, count in analysis['highlight_counts'].items():
            print(f"    {htype}: {count}")


        print("\nExporting highlight clips...")
        try:
            clips = export_highlights(
                str(replays_dir),
                str(highlights_dir),
                max_highlights=20
            )
            print(f"  Exported {len(clips)} clips to {highlights_dir}")
        except Exception as e:
            print(f"  Warning: Could not export highlights: {e}")

    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(description='Evaluate DQN on Block Blast')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--episodes', type=int, default=50, help='Number of episodes')
    parser.add_argument('--run_name', type=str, default='dqn_eval', help='Run name')
    parser.add_argument('--render_best', action='store_true', default=True,
                        help='Render best episode')
    parser.add_argument('--export_highlights', action='store_true', default=True,
                        help='Export highlight clips')

    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
