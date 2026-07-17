"""
Gerçek Indian Crop Yield veri setinin istatistiklerine dayalı
yüksek kaliteli sentetik veri üretici.
Fertilizer sayısal (kg), Yield, Rainfall vb. içerir.
"""
import numpy as np
import pandas as pd
import os

np.random.seed(42)

n_samples = 5000  # Eğitim ve kalibrasyon için yeterli

crops = ['Rice', 'Wheat', 'Maize', 'Cotton', 'Sugarcane', 'Soybean', 'Groundnut', 'Potato']
seasons = ['Kharif', 'Rabi', 'Whole Year']
states = ['Maharashtra', 'Uttar Pradesh', 'Madhya Pradesh', 'Punjab', 'Karnataka', 
          'Andhra Pradesh', 'Tamil Nadu', 'Gujarat', 'Bihar', 'Rajasthan']

data = {
    'Crop': np.random.choice(crops, n_samples),
    'Crop_Year': np.random.randint(2005, 2021, n_samples),
    'Season': np.random.choice(seasons, n_samples),
    'State': np.random.choice(states, n_samples),
    'Area': np.random.lognormal(mean=5.5, sigma=1.2, size=n_samples).clip(1, 50000),
}

# Gerçekçi yağış (mm)
data['Annual_Rainfall'] = np.random.normal(1050, 450, n_samples).clip(200, 3500)

# Fertilizer (kg) - gerçekçi dağılım (alan ile ilişkili)
base_fert = np.random.lognormal(mean=9.5, sigma=0.9, size=n_samples)
data['Fertilizer'] = (base_fert * (data['Area'] / 100) * np.random.uniform(0.7, 1.4, n_samples)).clip(50, 2_000_000)

# Pesticide
data['Pesticide'] = (data['Fertilizer'] * np.random.uniform(0.02, 0.12, n_samples)).clip(5, 100000)

# Production ve Yield (gübre ve yağış ile ilişkili)
fert_per_ha = data['Fertilizer'] / data['Area']
rain_factor = np.clip(data['Annual_Rainfall'] / 1000, 0.4, 1.8)
fert_factor = np.clip(fert_per_ha / 80, 0.5, 2.2)

base_yield = np.random.normal(2800, 800, n_samples)
data['Yield'] = (base_yield * rain_factor * fert_factor * np.random.uniform(0.75, 1.25, n_samples)).clip(400, 12000)
data['Production'] = data['Yield'] * data['Area'] / 1000  # ton civarı

df = pd.DataFrame(data)

# Ekstra kalibrasyon sütunları (simülasyon için)
df['Fertilizer_per_ha'] = df['Fertilizer'] / df['Area']
df['Rainfall_norm'] = df['Annual_Rainfall'] / 1500

os.makedirs(os.path.dirname(__file__), exist_ok=True)
out_path = os.path.join(os.path.dirname(__file__), 'indian_crop_yield_synthetic.csv')
df.to_csv(out_path, index=False)

print(f"Sentetik veri seti oluşturuldu: {out_path}")
print(f"Şekil: {df.shape}")
print("\n--- Temel İstatistikler ---")
print(df[['Area', 'Annual_Rainfall', 'Fertilizer', 'Fertilizer_per_ha', 'Yield']].describe().round(1))
print("\nFertilizer_per_ha (kg/ha) dağılımı kullanılarak aksiyon aralıkları belirlenecek.")