
import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from game.blockblast_env import BlockBlastEnv
from ai.baselines import get_agent, RandomAgent, ValidRandomAgent, HeuristicAgent, GreedyAgent, MaxClearAgent
from render.replay import render_replay


def run_agent(agent, env: BlockBlastEnv, num_episodes: int, record: bool = True) -> Dict:
    scores = []
    steps_list = []
    episode_data = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        agent.reset()
        done = False
        total_reward = 0
        ep_steps = 0

        while not done:
            state = env.get_state_for_nn()
            valid_mask = env.get_valid_action_mask()
            action = agent.select_action(env)

            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            ep_steps += 1

            if ep_steps > 1000:
                break

        scores.append(env.score)
        steps_list.append(ep_steps)
        episode_data.append({
            'episode': ep,
            'score': env.score,
            'steps': ep_steps,
            'reward': total_reward
        })

        if (ep + 1) % 50 == 0:
            print(f"  {agent.name}: Episode {ep + 1}/{num_episodes}, "
                  f"Avg Score: {np.mean(scores):.1f}")

    return {
        'agent': agent.name,
        'scores': scores,
        'steps': steps_list,
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'max_score': max(scores),
        'min_score': min(scores),
        'median_score': np.median(scores),
        'mean_steps': np.mean(steps_list),
        'episodes': episode_data
    }


def find_representative_episode(results: Dict) -> int:
    scores = results['scores']
    median = np.median(scores)
    closest_idx = min(range(len(scores)), key=lambda i: abs(scores[i] - median))
    return closest_idx


def main():
    parser = argparse.ArgumentParser(description='Compare Block Blast agents')
    parser.add_argument('--episodes', type=int, default=100, help='Episodes per agent')
    parser.add_argument('--run_name', type=str, default='compare', help='Run name for outputs')
    parser.add_argument('--agents', type=str, nargs='+',
                        default=['validrandom', 'heuristic', 'greedy', 'maxclear'],
                        help='Agents to compare')
    parser.add_argument('--render_best', action='store_true', help='Render best episodes')

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"BLOCK BLAST - AGENT COMPARISON")
    print(f"{'='*60}")
    print(f"Episodes per agent: {args.episodes}")
    print(f"Agents: {args.agents}")
    print(f"Run name: {args.run_name}")
    print(f"{'='*60}\n")


    out_base = Path("outputs")
    logs_dir = out_base / "logs" / args.run_name
    plots_dir = out_base / "plots" / args.run_name
    videos_dir = out_base / "videos" / "compare"
    replays_dir = out_base / "replays" / args.run_name

    for d in [logs_dir, plots_dir, videos_dir, replays_dir]:
        d.mkdir(parents=True, exist_ok=True)


    all_results = []
    start_time = time.time()

    for agent_name in args.agents:
        print(f"\nRunning {agent_name}...")
        agent = get_agent(agent_name)


        env = BlockBlastEnv(record_episodes=True, run_name=f"{args.run_name}_{agent_name}")

        results = run_agent(agent, env, args.episodes)
        all_results.append(results)

        print(f"  {agent_name}: Mean={results['mean_score']:.1f} "
              f"Max={results['max_score']} Median={results['median_score']:.1f}")

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f}s")


    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Agent':<15} {'Mean':>10} {'Std':>10} {'Max':>8} {'Median':>10} {'Steps':>8}")
    print(f"{'-'*60}")

    for r in all_results:
        print(f"{r['agent']:<15} {r['mean_score']:>10.1f} {r['std_score']:>10.1f} "
              f"{r['max_score']:>8} {r['median_score']:>10.1f} {r['mean_steps']:>8.1f}")

    print(f"{'='*60}\n")


    csv_path = logs_dir / "compare.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['agent', 'mean_score', 'std_score', 'max_score',
                         'min_score', 'median_score', 'mean_steps'])
        for r in all_results:
            writer.writerow([
                r['agent'], r['mean_score'], r['std_score'], r['max_score'],
                r['min_score'], r['median_score'], r['mean_steps']
            ])
    print(f"Saved: {csv_path}")


    json_path = logs_dir / "compare_detailed.json"
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"Saved: {json_path}")


    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))


        agents = [r['agent'] for r in all_results]
        means = [r['mean_score'] for r in all_results]
        stds = [r['std_score'] for r in all_results]

        ax1 = axes[0]
        bars = ax1.bar(agents, means, yerr=stds, capsize=5, color=['#3498db', '#2ecc71', '#e74c3c', '#9b59b6'])
        ax1.set_ylabel('Score')
        ax1.set_title('Mean Score by Agent')
        ax1.set_ylim(0, max(means) * 1.3)


        for bar, mean in zip(bars, means):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                     f'{mean:.0f}', ha='center', va='bottom', fontsize=10)


        ax2 = axes[1]
        box_data = [r['scores'] for r in all_results]
        bp = ax2.boxplot(box_data, labels=agents, patch_artist=True)
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax2.set_ylabel('Score')
        ax2.set_title('Score Distribution by Agent')

        plt.tight_layout()
        plot_path = plots_dir / "compare.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved: {plot_path}")

    except ImportError:
        print("Warning: matplotlib not available, skipping plots")


    if args.render_best:
        print("\nRendering best episodes...")
        for r in all_results:
            agent_name = r['agent']
            best_ep = np.argmax(r['scores'])
            rep_ep = find_representative_episode(r)

            replay_base = out_base / "replays" / f"{args.run_name}_{agent_name}"

            for ep_idx, ep_type in [(best_ep, 'best'), (rep_ep, 'representative')]:
                replay_path = replay_base / f"episode_{ep_idx:06d}.json"
                if replay_path.exists():
                    out_dir = videos_dir / f"{agent_name}_{ep_type}"
                    print(f"  Rendering {agent_name} {ep_type} (ep {ep_idx})...")
                    try:
                        render_replay(str(replay_path), str(out_dir), to_mp4=True, to_gif=False)
                    except Exception as e:
                        print(f"    Warning: Could not render: {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
