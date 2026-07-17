import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

# DQN için kullanılacak yapay sinir ağı (Sinir Ağı) mimarisi
class DQNNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=[256, 128, 64]):
        super(DQNNetwork, self).__init__()
        layers = []
        prev = state_dim
        # Gizli katmanları (hidden layers) oluştur
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU()) # Aktivasyon fonksiyonu
            layers.append(nn.LayerNorm(h)) # Eğitim istikrarı için LayerNorm (Katman Normalizasyonu)
            prev = h
        # Çıktı katmanı: her eylem için bir Q-değeri döndürür
        layers.append(nn.Linear(prev, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# DQN Ajan (Agent) sınıfı
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.97,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.992,
                 buffer_size=80000, batch_size=128, target_update=500, device='cpu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma # İndirim faktörü (discount factor)
        self.epsilon = epsilon_start # Keşif (exploration) oranı
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay # Keşif oranının azalma katsayısı
        self.batch_size = batch_size
        self.target_update = target_update # Hedef ağı (target net) güncelleme sıklığı
        self.device = device
        self.steps = 0

        # Politika (Policy) ve Hedef (Target) ağlarının oluşturulması
        self.policy_net = DQNNetwork(state_dim, action_dim).to(device)
        self.target_net = DQNNetwork(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss() # Huber loss (Hata fonksiyonu)
        self.buffer = deque(maxlen=buffer_size) # Deneyim tekrar (replay) belleği

    def select_action(self, state, evaluate=False):
        # Epsilon-greedy stratejisiyle eylem seçimi
        if (not evaluate) and random.random() < self.epsilon:
            return random.randrange(self.action_dim) # Rastgele eylem seçimi (Keşif)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q = self.policy_net(s)
            return int(q.argmax(dim=1).item()) # En yüksek Q-değerine sahip eylem (Sömürü)

    def store(self, s, a, r, ns, done):
        # Deneyimi belleğe kaydet
        self.buffer.append((s, a, r, ns, done))

    def update_epsilon(self):
        # Epsilon değerini zamanla azalt (Keşfi azalt, sömürüyü artır)
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def train_step(self):
        # Yeterli deneyim yoksa eğitimi atla
        if len(self.buffer) < self.batch_size:
            return 0.0

        # Bellekten rastgele bir mini-batch (örneklem) al
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Tensörlere dönüştür
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Mevcut durum için ağın tahmin ettiği Q-değerlerini hesapla
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Sonraki durumlar için hedef ağdan (target_net) maksimum Q-değerlerini al
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q # Bellman denklemi

        # Kayıp (loss) hesaplaması ve ağın güncellenmesi
        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradyan patlamasını önlemek için kırpma (clipping)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5.0)
        self.optimizer.step()

        self.steps += 1
        # Belirli adım aralıklarıyla hedef ağı politikaya eşitle (güncelle)
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def save(self, path):
        # Model ağırlıklarını ve diğer durumları belirtilen yola kaydet
        torch.save({
            'policy': self.policy_net.state_dict(),
            'target': self.target_net.state_dict(),
            'optim': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)

    def load(self, path):
        # Kaydedilmiş modeli yoldan yükle
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt['policy'])
        self.target_net.load_state_dict(ckpt['target'])
        self.optimizer.load_state_dict(ckpt['optim'])
        self.epsilon = ckpt.get('epsilon', self.epsilon_end)
