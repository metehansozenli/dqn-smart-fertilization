"""
Full DQN training with comprehensive metrics for IEEE paper figures.
Logs: reward, yield, loss, epsilon, action distribution.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
import random

from env.agriculture_env import AgricultureEnv
from agent.dqn_agent import DQNAgent


def train_full(episodes=800, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    env = AgricultureEnv(use_real_data=True, max_days=30, seed=seed)
    state_dim = 9
    action_dim = env.n_actions
    device = 'cpu'

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=1e-4,
        gamma=0.97,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.994,
        buffer_size=60000,
        batch_size=128,
        target_update=300,
        device=device
    )

    rewards, yields, losses, epsilons = [], [], [], []
    all_actions = []
    action_names = [env.actions[i]['name'] for i in range(action_dim)]

    print(f"Training Hybrid DQN | {episodes} episodes | {action_dim} actions")
    for ep in tqdm(range(episodes)):
        state, _ = env.reset()
        ep_reward = 0.0
        ep_losses = []
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _, info = env.step(action)
            agent.store(state, action, reward, next_state, float(done))
            loss = agent.train_step()
            if loss is not None and loss > 0:
                ep_losses.append(loss)
            agent.update_epsilon()
            state = next_state
            ep_reward += reward
            all_actions.append(action)

        rewards.append(ep_reward)
        yields.append(info.get('current_yield', 0))
        losses.append(np.mean(ep_losses) if ep_losses else 0.0)
        epsilons.append(agent.epsilon)

        if (ep + 1) % 100 == 0:
            print(f"Ep {ep+1:4d} | R: {np.mean(rewards[-50:]):7.1f} | "
                  f"Y: {np.mean(yields[-50:]):6.0f} | Loss: {np.mean(losses[-50:]):.4f} | eps: {agent.epsilon:.3f}")

    # Save model
    os.makedirs('results', exist_ok=True)
    agent.save('results/hybrid_dqn_model.pth')
    print("Model saved.")

    # ========== COMPREHENSIVE FIGURE ==========
    fig = plt.figure(figsize=(14, 12))

    # 1. Reward
    ax1 = fig.add_subplot(3, 2, 1)
    ax1.plot(rewards, alpha=0.25, color='#4C72B0', linewidth=0.8)
    if len(rewards) > 30:
        ma = np.convolve(rewards, np.ones(30)/30, mode='valid')
        ax1.plot(range(29, len(rewards)), ma, color='#C44E52', lw=2.2, label='MA-30')
    ax1.set_title('(a) Episode Reward', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Cumulative Reward')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    # 2. Yield
    ax2 = fig.add_subplot(3, 2, 2)
    ax2.plot(yields, alpha=0.25, color='#55A868', linewidth=0.8)
    if len(yields) > 30:
        ma = np.convolve(yields, np.ones(30)/30, mode='valid')
        ax2.plot(range(29, len(yields)), ma, color='#DD8452', lw=2.2, label='MA-30')
    ax2.axhline(3800, color='gray', ls='--', lw=1.5, label='Data Median')
    ax2.set_title('(b) Final Yield (kg/ha)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Yield (kg/ha)')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)

    # 3. Loss
    ax3 = fig.add_subplot(3, 2, 3)
    ax3.plot(losses, alpha=0.4, color='#8172B2', linewidth=0.8)
    if len(losses) > 30:
        ma = np.convolve(losses, np.ones(30)/30, mode='valid')
        ax3.plot(range(29, len(losses)), ma, color='#4C72B0', lw=2.0)
    ax3.set_title('(c) TD Loss (Smooth L1)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Loss')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)

    # 4. Epsilon
    ax4 = fig.add_subplot(3, 2, 4)
    ax4.plot(epsilons, color='#C44E52', lw=2.0)
    ax4.set_title('(d) Exploration Rate (ε)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('ε')
    ax4.set_ylim(0, 1.05)
    ax4.grid(True, alpha=0.3)

    # 5. Action Distribution (last 30% of training)
    ax5 = fig.add_subplot(3, 2, 5)
    late_actions = all_actions[int(len(all_actions)*0.7):]
    counts = Counter(late_actions)
    labels = [action_names[i] for i in range(action_dim)]
    values = [counts.get(i, 0) for i in range(action_dim)]
    colors = plt.cm.Set2(np.linspace(0, 1, action_dim))
    bars = ax5.bar(range(action_dim), values, color=colors, edgecolor='black', linewidth=0.5)
    ax5.set_xticks(range(action_dim))
    ax5.set_xticklabels(labels, rotation=35, ha='right', fontsize=8)
    ax5.set_title('(e) Action Distribution (Last 30% Episodes)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Count')
    ax5.grid(True, alpha=0.3, axis='y')

    # 6. Yield vs Fertilizer efficiency (scatter from late episodes)
    ax6 = fig.add_subplot(3, 2, 6)
    # Simulate efficiency trend
    late_y = yields[int(len(yields)*0.5):]
    ax6.hist(late_y, bins=25, color='#64B5CD', edgecolor='black', alpha=0.75)
    ax6.axvline(np.mean(late_y), color='red', ls='--', lw=2, label=f'Mean: {np.mean(late_y):.0f}')
    ax6.set_title('(f) Yield Distribution (Second Half)', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Yield (kg/ha)')
    ax6.set_ylabel('Frequency')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    out = 'results/full_dqn_paper_figures.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Full figure saved → {out}")

    # Summary
    print("\n========== FINAL METRICS ==========")
    print(f"Episodes              : {episodes}")
    print(f"Final Avg Reward (50) : {np.mean(rewards[-50:]):.1f}")
    print(f"Final Avg Yield (50)  : {np.mean(yields[-50:]):.0f} kg/ha")
    print(f"Best Yield            : {np.max(yields):.0f} kg/ha")
    print(f"Final Avg Loss        : {np.mean(losses[-50:]):.5f}")
    print(f"Final Epsilon         : {agent.epsilon:.3f}")
    print(f"Most used action      : {action_names[np.argmax(values)]}")

    return agent


if __name__ == "__main__":
    train_full(episodes=800)
