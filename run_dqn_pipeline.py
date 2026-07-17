#!/usr/bin/env python3
"""
DQN Akıllı Gübreleme projesi için ana işlem hattı (pipeline).
Hem simüle edilmiş hem de gerçek veriler üzerinde modeli eğitir ve değerlendirir.
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

    # 1. Simüle edilmiş veri üzerinde eğitim
    print("\n[1/3] Simüle Edilmiş Ortamda DQN Eğitimi Başlıyor...")
    train_dqn(use_real_data=False, episodes=2000, save_path="results/dqn_sim_model.pth")

    # 2. Gerçek veri üzerinde eğitim
    real_data_path = "data/real_fertilization_data.csv"
    if os.path.exists(real_data_path):
        print("\n[2/3] Gerçek Veri Setinde DQN Eğitimi Başlıyor...")
        train_dqn(use_real_data=True, data_path=real_data_path, episodes=2000,
                  save_path="results/dqn_real_model.pth")
    else:
        print(f"\n[2/3] {real_data_path} yolunda gerçek veri bulunamadı. Bu adım atlanıyor.")

    # 3. Her iki modelin değerlendirilmesi
    print("\n[3/3] Modeller Değerlendiriliyor...")
    if os.path.exists("results/dqn_sim_model.pth"):
        evaluate_dqn("results/dqn_sim_model.pth", use_real_data=False, episodes=100)

    if os.path.exists("results/dqn_real_model.pth"):
        evaluate_dqn("results/dqn_real_model.pth", use_real_data=True, episodes=100)

    print("\n" + "=" * 60)
    print("İşlem hattı tamamlandı! Modeller ve grafikler için results/ klasörünü kontrol edin.")
    print("=" * 60)

if __name__ == "__main__":
    main()