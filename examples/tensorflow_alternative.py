"""
Exemple alternatif utilisant TensorFlow au lieu de PyTorch

Ce script montre comment le même type de modèle pourrait être implémenté
avec TensorFlow/Keras comme alternative à PyTorch.

Note: TensorFlow n'est pas inclus dans requirements.txt par défaut.
Pour utiliser cette alternative, installez TensorFlow:
    pip install tensorflow>=2.13.0
"""

def train_with_tensorflow_example():
    """
    Exemple de code pour entraîner un modèle similaire avec TensorFlow
    
    Ce code n'est pas exécuté par défaut car TensorFlow n'est pas installé,
    mais il montre comment adapter le module pour utiliser TensorFlow.
    """
    
    print("=" * 70)
    print("Exemple d'implémentation avec TensorFlow/Keras")
    print("=" * 70)
    print("\nNote: Ce code nécessite TensorFlow. Installez-le avec:")
    print("  pip install tensorflow>=2.13.0")
    print("=" * 70)
    
    example_code = """
# Exemple d'implémentation avec TensorFlow

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Charger les données
data = pd.read_csv('data/raw/client_data.csv')

# 2. Séparer features et target
X = data[['age', 'revenu', 'anciennete', 'score_credit', 'depenses_mensuelles']].values
y = data['risque'].values

# 3. Séparer train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Normaliser
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Créer le modèle avec Keras
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(5,)),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(16, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1)
])

# 6. Compiler le modèle
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

# 7. Entraîner
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    verbose=1
)

# 8. Évaluer
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss:.4f}')
print(f'Test MAE: {test_mae:.4f}')

# 9. Sauvegarder
model.save('models/tensorflow_model.h5')

# 10. Charger et prédire
loaded_model = keras.models.load_model('models/tensorflow_model.h5')
predictions = loaded_model.predict(X_test)
"""
    
    print("\n" + example_code)
    
    print("\n" + "=" * 70)
    print("Comparaison PyTorch vs TensorFlow")
    print("=" * 70)
    print("""
PyTorch (implémentation actuelle):
  ✓ Plus flexible pour la recherche
  ✓ Plus pythonique
  ✓ Meilleur pour le débogage
  ✓ Communauté académique forte
  
TensorFlow/Keras:
  ✓ Plus simple pour débuter
  ✓ Meilleur pour la production
  ✓ TensorFlow Lite pour mobile
  ✓ Meilleur support Google Cloud
  
Les deux sont d'excellents choix et le module actuel pourrait être
facilement adapté pour utiliser TensorFlow si nécessaire.
""")
    print("=" * 70)

def main():
    """Fonction principale"""
    train_with_tensorflow_example()
    
    print("\nLe module actuel utilise PyTorch car:")
    print("  1. PyTorch offre plus de flexibilité pour personnaliser le modèle")
    print("  2. Le code est plus facile à comprendre et à modifier")
    print("  3. PyTorch a une excellente communauté et documentation")
    print("\nMais vous pouvez facilement adapter ce code pour TensorFlow!")

if __name__ == "__main__":
    main()
