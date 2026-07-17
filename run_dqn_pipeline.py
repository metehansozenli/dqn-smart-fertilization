#!/usr/bin/env python3
"""
Main pipeline for DQN Auto-Fertilization project.
Trains and evaluates on both simulated and real data.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training.train_dqn import train_dqn
from evaluation.evaluate_dqn import evaluate_dqn

def main():
    print("=" * 60)
    print("DQN Auto-Fertilization Pipeline")
    print("=" * 60)

    os.makedirs("results", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # 1. Train on simulated data
    print("\n[1/3] Training DQN on Simulated Environment...")
    train_dqn(use_real_data=False, episodes=2000, save_path="results/dqn_sim_model.pth")

    # 2. Train on real data
    real_data_path = "data/real_fertilization_data.csv"
    if os.path.exists(real_data_path):
        print("\n[2/3] Training DQN on Real Dataset...")
        train_dqn(use_real_data=True, data_path=real_data_path, episodes=2000,
                  save_path="results/dqn_real_model.pth")
    else:
        print(f"\n[2/3] Real data not found at {real_data_path}. Skipping real-data training.")

    # 3. Evaluate both
    print("\n[3/3] Evaluating models...")
    if os.path.exists("results/dqn_sim_model.pth"):
        evaluate_dqn("results/dqn_sim_model.pth", use_real_data=False, episodes=100)

    if os.path.exists("results/dqn_real_model.pth"):
        evaluate_dqn("results/dqn_real_model.pth", use_real_data=True, episodes=100)

    print("\n" + "=" * 60)
    print("Pipeline completed! Check results/ folder for models and plots.")
    print("=" * 60)

if __name__ == "__main__":
    main()