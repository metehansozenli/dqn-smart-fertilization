# Derin Pekiştirmeli Öğrenme ile Akıllı Gübreleme ve Sulama

**DQN Tabanlı Hassas Tarım Sistemi**

Bu proje, bitki yetiştirme sezonunda günlük sulama ve gübreleme kararlarını öğrenen bir **Deep Q-Network (DQN)** ajanı sunar. Ortam, gerçek ürün verim veri seti istatistikleri ile kalibre edilmiş hibrit bir simülasyondur.

---

## Ana Sonuç (DQN vs Rastgele)

Aynı ortamda eğitilmiş DQN politikası ile rastgele politikanın karşılaştırması:

![DQN vs Rastgele](results/compare_dqn_vs_random.gif)

| Metrik | DQN (eğitilmiş) | Rastgele |
|--------|-----------------|----------|
| Verim | **~3500+ kg/ha** | ~1800 kg/ha |
| Sezon gübre kullanımı | **~120 kg/ha** | ~2000+ kg/ha |
| Toksisite | Düşük / kontrollü | Yüksek (bitki ölümü) |

- **Üst panel:** DQN ajanı — duruma bakarak sulama / ölçülü gübre seçer  
- **Alt panel:** Rastgele politika — aşırı gübre → toksisite → verim çöküşü  
- Tüm sayılar `env.step()` ve eğitilmiş model çıktısından gelir  

---

## Eğitim Grafikleri

![Eğitim Tanıları](results/full_dqn_paper_figures.png)

Ödül, verim, TD loss, keşif oranı (ε), aksiyon dağılımı ve verim histogramı.

---

## Proje Ne Yapıyor?

1. **Durum:** nem, besin, pH, sıcaklık, sağlık, toksisite vb. (9 boyut)  
2. **Aksiyon:** 8 ayrık seçenek (bekle, sulama, gübre 40/70/140 kg, kombinasyonlar)  
3. **Öğrenme:** DQN (experience replay + target network)  
4. **Hedef:** yüksek verim + gerçekçi gübre kullanımı + düşük toksisite  

Gübre dozları gerçek veri yüzdeliklerine (medyan ~137 kg/ha) hizalanmıştır.

---

## Hızlı Başlangıç

```bash
# 1) Bağımlılıklar
pip install -r requirements.txt

# 2) Eğit + grafikleri üret
python training/train_full_metrics.py

# 3) DQN vs Rastgele GIF
python visualization/compare_gif.py
```

Eğitilmiş model: `results/hybrid_dqn_model.pth`

---

## Dizin Yapısı

```
.
├── agent/dqn_agent.py           # DQN ajanı (PyTorch)
├── env/agriculture_env.py       # Hibrit tarım ortamı
├── training/train_full_metrics.py
├── visualization/compare_gif.py # Ana karşılaştırma GIF
├── data/                        # Kalibrasyon veri seti
├── results/
│   ├── compare_dqn_vs_random.gif   ← ana görsel
│   ├── full_dqn_paper_figures.png
│   ├── hybrid_dqn_model.pth
│   └── ieee_paper.pdf
├── paper/ieee_paper.tex|.pdf    # IEEE tarzı makale (TR, 6 sayfa)
├── requirements.txt
└── README.md
```

---

## Akademik Makale

- **Dosya:** `paper/ieee_paper.pdf` / `results/ieee_paper.pdf`  
- **Dil:** Türkçe  
- **İçerik:** literatür, MDP formülasyonu, DQN mimarisi, gerçek veri entegrasyonu, deneysel sonuçlar  
- **Referanslar:** DQN (Mnih), AquaCrop, FAO Ky, CropGym vb.

---

## Öğrenilen Politika (Özet)

Ajan şunu öğrenmiştir:

1. Nemı korumak için düzenli **hafif sulama**  
2. Sezon içinde birkaç kez **40 kg gübre** (toplam gerçek veri medyanına yakın)  
3. Yüksek tek seferlik dozlardan kaçınma (toksisite)  

Rastgele politika ise aşırı gübreleyip toksisite ile verimi düşürür. Karşılaştırma GIF’i bu farkı gösterir.

---

## Notlar

- GIF ve grafikteki metrikler model/ortam çıktısıdır; bitki çizimi görselleştirmedir (boy ≈ büyüme, renk ≈ sağlık).  
- Seasonal denemeler proje kapsamı dışında bırakılmıştır; sunulan sistem günlük hibrit modeldir.

---

## Lisans

MIT
