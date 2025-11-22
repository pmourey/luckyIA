"""
Script pour générer des données d'exemple pour tester le module IA
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Définir le seed pour la reproductibilité
np.random.seed(42)

# Paramètres
n_samples = 1000
n_features = 5

# Générer des données synthétiques
# Simulation de données clients: âge, revenu, ancienneté, score crédit, dépenses
data = {
    'age': np.random.randint(18, 80, n_samples),
    'revenu': np.random.normal(50000, 20000, n_samples).clip(15000, 200000),
    'anciennete': np.random.randint(0, 30, n_samples),
    'score_credit': np.random.randint(300, 850, n_samples),
    'depenses_mensuelles': np.random.normal(2000, 800, n_samples).clip(500, 8000)
}

# Créer une variable cible (par exemple: risque de défaut)
# Calcul basé sur une formule combinant les features
df = pd.DataFrame(data)
df['risque'] = (
    0.3 * (800 - df['score_credit']) / 500 +
    0.2 * (df['depenses_mensuelles'] / df['revenu']) +
    0.15 * (70 - df['age']) / 50 +
    0.1 * np.random.random(n_samples)
).clip(0, 1)

# Créer le dossier data si nécessaire
data_dir = Path(__file__).parent.parent / 'data' / 'raw'
data_dir.mkdir(parents=True, exist_ok=True)

# Sauvegarder les données
output_path = data_dir / 'client_data.csv'
df.to_csv(output_path, index=False)

print(f"Données générées et sauvegardées: {output_path}")
print(f"Nombre d'échantillons: {n_samples}")
print(f"Nombre de features: {n_features}")
print("\nAperçu des données:")
print(df.head())
print("\nStatistiques descriptives:")
print(df.describe())
