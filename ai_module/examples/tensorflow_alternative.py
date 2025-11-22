"""Module avec l'exemple TensorFlow (non exécuté si TensorFlow absent).
Ce fichier est une reprise de examples/tensorflow_alternative.py
"""

def train_with_tensorflow_example():
    """Affiche un exemple prêt-à-copier pour entraîner un modèle avec TensorFlow/Keras."""
    from textwrap import dedent
    example_code = dedent('''
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
    ''')

    print(example_code)


if __name__ == '__main__':
    train_with_tensorflow_example()

