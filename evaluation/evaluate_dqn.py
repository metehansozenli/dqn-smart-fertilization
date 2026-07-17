import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from env.agriculture_env import AgricultureEnv
from agent.dqn_agent import DQNAgent

def evaluate_dqn(model_path, use_real_data=False, data_path="data/real_fertilization_data.csv",
                 episodes=100, max_steps=30, seed=42, device='cpu', render=False):

    env = AgricultureEnv(use_real_data=use_real_data, data_path=data_path, max_days=max_steps, seed=seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim, device=device)
    agent.load(model_path)
    agent.epsilon = 0.0  # Pure exploitation

    rewards = []
    yields = []
    action_counts = np.zeros(action_dim)

    print(f"Evaluating DQN model from {model_path} on {'real' if use_real_data else 'sim'} data...")

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        final_yield = 0

        for step in range(max_steps):
            action = agent.select_action(state, evaluate=True)
            action_counts[action] += 1
            next_state, reward, terminated, truncated, info = env.step(action)
            state = next_state
            total_reward += reward
            final_yield = info.get('current_yield', 0)
            if terminated or truncated:
                break

        rewards.append(total_reward)
        yields.append(final_yield)

        if render and ep < 3:
            env.render()

    print(f"\n=== Evaluation Results ({episodes} episodes) ===")
    print(f"Average Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Average Final Yield: {np.mean(yields):.1f} ± {np.std(yields):.1f} kg/ha")
    print(f"Action Distribution: {action_counts / episodes:.1f} (per episode avg)")
    print(f"Best Yield: {max(yields):.1f} | Worst: {min(yields):.1f}")

    return {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'mean_yield': np.mean(yields),
        'std_yield': np.std(yields),
        'yields': yields,
        'action_dist': action_counts / episodes
    }

if __name__ == "__main__":
    # Example evaluations
    if os.path.exists("results/dqn_sim_model.pth"):
        evaluate_dqn("results/dqn_sim_model.pth", use_real_data=False, episodes=50)
    if os.path.exists("results/dqn_real_model.pth"):
        evaluate_dqn("results/dqn_real_model.pth", use_real_data=True, episodes=50)