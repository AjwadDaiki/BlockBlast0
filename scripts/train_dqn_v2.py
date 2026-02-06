"""
DQN V2 Training Script for Block Blast
Uses improved DQN architecture and reward shaping
"""

import argparse
import csv
import time
from pathlib import Path
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from game.blockblast_env import BlockBlastEnv
from ai.dqn_v2 import DQNv2Agent, DQNv2Config
from ai.rewards_v2 import compute_reward_v2, RewardV2Config, compute_terminal_reward_v2


def train(args):
    """Main training loop for DQN V2"""
    print(f"\n{'='*60}")
    print(f"BLOCK BLAST - DQN V2 TRAINING (IMPROVED)")
    print(f"{'='*60}")
    print(f"Episodes: {args.episodes}")
    print(f"Run name: {args.run_name}")
    print(f"Improvements: Dueling DQN, LayerNorm, Soft updates, Better rewards")
    print(f"{'='*60}\n")

    # Create directories
    out_base = Path("outputs")
    logs_dir = out_base / "logs" / args.run_name
    plots_dir = out_base / "plots" / args.run_name
    replays_dir = out_base / "replays" / args.run_name
    checkpoints_dir = Path("checkpoints") / args.run_name

    for d in [logs_dir, plots_dir, replays_dir, checkpoints_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Create environment
    env = BlockBlastEnv(record_episodes=True, run_name=args.run_name)

    # Create V2 agent with improved config
    config = DQNv2Config(
        hidden_dims=[512, 512, 256],
        learning_rate=args.lr,
        batch_size=128,
        buffer_size=200000,
        target_update_freq=100,
        tau=0.005,  # Soft update
        epsilon_start=1.0,
        epsilon_end=0.02,
        epsilon_decay=0.9998,
        gamma=0.99,
        double_dqn=True,
        dueling=True
    )

    agent = DQNv2Agent(env.state_dim, env.action_dim, config)
    print(f"Agent config: {config}")
    print(f"Device: {agent.device}")

    # Load checkpoint if resuming
    if args.resume:
        checkpoint_path = checkpoints_dir / "latest.pt"
        if checkpoint_path.exists():
            print(f"Resuming from {checkpoint_path}")
            agent.load(str(checkpoint_path))

    # Training metrics
    reward_config = RewardV2Config()
    episode_scores = []
    episode_steps = []
    episode_rewards = []
    losses = []
    best_score = 0
    best_avg = 0
    start_time = time.time()

    # CSV logging
    csv_path = logs_dir / "train.csv"
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['episode', 'steps', 'score', 'reward', 'epsilon',
                         'avg_score_100', 'best_score', 'loss', 'lr', 'time_elapsed'])

    print("Training started...")
    print(f"{'Episode':>8} {'Score':>8} {'Avg100':>8} {'Best':>8} {'Eps':>8} {'Loss':>10} {'LR':>10}")
    print("-" * 75)

    for episode in range(args.episodes):
        obs, info = env.reset()
        state = env.get_state_for_nn()
        done = False
        episode_reward = 0
        episode_loss = []
        step = 0

        while not done:
            valid_mask = env.get_valid_action_mask()
            action = agent.select_action(state, valid_mask, training=True)

            obs, reward, done, truncated, info = env.step(action)
            next_state = env.get_state_for_nn()
            next_valid_mask = env.get_valid_action_mask()

            # Compute improved reward
            if done:
                shaped_reward = compute_terminal_reward_v2(env.score, step, reward_config)
            else:
                shaped_reward = compute_reward_v2(info, reward_config)

            # Store transition
            agent.store_transition(state, action, shaped_reward, next_state, done, next_valid_mask)

            # Multiple updates per step for faster learning
            for _ in range(2):
                loss = agent.update()
                if loss is not None:
                    episode_loss.append(loss)

            state = next_state
            episode_reward += shaped_reward
            step += 1

            if step > 1000:
                break

        # End episode
        agent.end_episode()
        episode_scores.append(env.score)
        episode_steps.append(step)
        episode_rewards.append(episode_reward)

        avg_loss = np.mean(episode_loss) if episode_loss else 0
        losses.append(avg_loss)

        # Update best score
        if env.score > best_score:
            best_score = env.score
            agent.save(str(checkpoints_dir / "best.pt"))

        # Calculate metrics
        avg_score_100 = np.mean(episode_scores[-100:])

        # Save best average model too
        if len(episode_scores) >= 100 and avg_score_100 > best_avg:
            best_avg = avg_score_100
            agent.save(str(checkpoints_dir / "best_avg.pt"))

        elapsed = time.time() - start_time
        current_lr = agent.optimizer.param_groups[0]['lr']

        # Log to CSV
        csv_writer.writerow([
            episode, step, env.score, episode_reward, agent.epsilon,
            avg_score_100, best_score, avg_loss, current_lr, elapsed
        ])
        csv_file.flush()

        # Print progress
        if (episode + 1) % args.log_interval == 0:
            print(f"{episode+1:>8} {env.score:>8} {avg_score_100:>8.1f} "
                  f"{best_score:>8} {agent.epsilon:>8.4f} {avg_loss:>10.4f} {current_lr:>10.6f}")

        # Save checkpoint
        if (episode + 1) % args.save_interval == 0:
            agent.save(str(checkpoints_dir / "latest.pt"))
            agent.save(str(checkpoints_dir / f"checkpoint_{episode+1}.pt"))

        # Update live plot
        if (episode + 1) % args.plot_interval == 0:
            _update_plots(episode_scores, losses, agent.epsilon, plots_dir)

    csv_file.close()

    # Final save
    agent.save(str(checkpoints_dir / "final.pt"))
    _update_plots(episode_scores, losses, agent.epsilon, plots_dir, final=True)

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Best score: {best_score}")
    print(f"Best avg (100): {best_avg:.1f}")
    print(f"Final avg (100): {np.mean(episode_scores[-100:]):.1f}")
    print(f"Total time: {time.time() - start_time:.1f}s")
    print(f"Checkpoints: {checkpoints_dir}")
    print(f"Logs: {logs_dir}")
    print(f"{'='*60}\n")


def _update_plots(scores, losses, epsilon, plots_dir, final=False):
    """Update training plots"""
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Score over time
        ax1 = axes[0, 0]
        ax1.plot(scores, alpha=0.3, color='blue', label='Raw')
        if len(scores) >= 100:
            rolling = np.convolve(scores, np.ones(100)/100, mode='valid')
            ax1.plot(range(99, len(scores)), rolling, color='blue', linewidth=2, label='Avg 100')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Score')
        ax1.set_title('Score over Training (V2)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Best score so far
        ax2 = axes[0, 1]
        best_so_far = np.maximum.accumulate(scores)
        ax2.plot(best_so_far, color='green', linewidth=2)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Best Score')
        ax2.set_title('Best Score Progress')
        ax2.grid(True, alpha=0.3)

        # Loss (log scale)
        ax3 = axes[1, 0]
        if losses:
            ax3.semilogy(losses, alpha=0.5, color='red')
            if len(losses) >= 100:
                rolling_loss = np.convolve(losses, np.ones(100)/100, mode='valid')
                ax3.semilogy(range(99, len(losses)), rolling_loss, color='red', linewidth=2)
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Loss (log scale)')
        ax3.set_title('Training Loss')
        ax3.grid(True, alpha=0.3)

        # Score distribution comparison
        ax4 = axes[1, 1]
        if len(scores) >= 200:
            first_100 = scores[:100]
            last_100 = scores[-100:]
            ax4.hist(first_100, bins=20, alpha=0.5, color='red', label=f'First 100 (avg={np.mean(first_100):.0f})')
            ax4.hist(last_100, bins=20, alpha=0.5, color='green', label=f'Last 100 (avg={np.mean(last_100):.0f})')
            ax4.axvline(np.mean(first_100), color='red', linestyle='--')
            ax4.axvline(np.mean(last_100), color='green', linestyle='--')
        else:
            ax4.hist(scores, bins=20, color='purple', alpha=0.7)
            ax4.axvline(np.mean(scores), color='red', linestyle='--', label=f'Mean: {np.mean(scores):.1f}')
        ax4.set_xlabel('Score')
        ax4.set_ylabel('Count')
        ax4.set_title('Score Distribution (First vs Last 100)')
        ax4.legend()

        plt.suptitle(f'DQN V2 Training - Epsilon: {epsilon:.4f}', fontsize=14)
        plt.tight_layout()

        # Save
        if final:
            plt.savefig(plots_dir / "progress.png", dpi=150)
        plt.savefig(plots_dir / "live.png", dpi=100)
        plt.close()

    except ImportError:
        pass


def main():
    parser = argparse.ArgumentParser(description='Train DQN V2 on Block Blast')
    parser.add_argument('--episodes', type=int, default=10000, help='Number of episodes')
    parser.add_argument('--run_name', type=str, default='dqn_v2', help='Run name')
    parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--log_interval', type=int, default=50, help='Log interval')
    parser.add_argument('--save_interval', type=int, default=500, help='Save interval')
    parser.add_argument('--plot_interval', type=int, default=100, help='Plot update interval')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
