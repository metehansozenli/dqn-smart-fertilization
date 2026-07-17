import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from env.agriculture_env import AgricultureEnv
from agent.dqn_agent import DQNAgent

def train(episodes=800, max_days=30, seed=42, save_path="results/hybrid_dqn_model.pth"):
    env = AgricultureEnv(use_real_data=True, max_days=max_days, seed=seed)
    state_dim = 9
    action_dim = env.n_actions

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device} | Actions: {action_dim}")

    agent = DQNAgent(state_dim, action_dim, device=device,
                     lr=1.5e-4, gamma=0.97, epsilon_decay=0.993,
                     batch_size=128, buffer_size=60000)

    rewards_hist, yields_hist = [], []

    print(f"Training Hybrid DQN for {episodes} episodes...")
    for ep in tqdm(range(episodes)):
        state, _ = env.reset()
        ep_reward = 0.0
        final_yield = 0.0

        for t in range(max_days):
            action = agent.select_action(state)
            next_state, reward, done, _, info = env.step(action)
            agent.store(state, action, reward, next_state, float(done))
            agent.train_step()
            state = next_state
            ep_reward += reward
            final_yield = info['current_yield']
            if done:
                break

        agent.update_epsilon()
        rewards_hist.append(ep_reward)
        yields_hist.append(final_yield)

        if (ep + 1) % 100 == 0:
            print(f"Ep {ep+1} | AvgR(100): {np.mean(rewards_hist[-100:]):.1f} | "
                  f"AvgYield: {np.mean(yields_hist[-100:]):.0f} | eps: {agent.epsilon:.3f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    agent.save(save_path)
    print(f"Model saved → {save_path}")

    # Training curves
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(rewards_hist, alpha=0.4, color='#4C72B0')
    if len(rewards_hist) > 40:
        ma = np.convolve(rewards_hist, np.ones(40)/40, mode='valid')
        axes[0].plot(range(39, len(rewards_hist)), ma, color='#C44E52', lw=2)
    axes[0].set_title('Episode Reward')
    axes[0].set_xlabel('Episode')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(yields_hist, alpha=0.4, color='#55A868')
    if len(yields_hist) > 40:
        ma = np.convolve(yields_hist, np.ones(40)/40, mode='valid')
        axes[1].plot(range(39, len(yields_hist)), ma, color='#DD8452', lw=2)
    axes[1].axhline(3800, color='gray', ls='--', label='Baseline')
    axes[1].set_title('Final Yield (kg/ha)')
    axes[1].set_xlabel('Episode')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    curve_path = save_path.replace('.pth', '_curves.png')
    plt.savefig(curve_path, dpi=140, bbox_inches='tight')
    plt.close()
    print(f"Curves saved → {curve_path}")

    return agent, rewards_hist, yields_hist

if __name__ == "__main__":
    train(episodes=700)
