"""
Script d'exemple pour faire des prédictions avec un modèle entraîné
"""

import sys
from pathlib import Path
import torch

# Ajouter le répertoire parent au path pour importer le module
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_module import ClientDataLoader, ClientModel, ModelTrainer

def main():
    """Fonction principale pour faire des prédictions"""
    
    print("=" * 60)
    print("Prédictions avec le modèle IA entraîné")
    print("=" * 60)
    
    # Vérifier que le modèle existe
    model_path = Path(__file__).parent.parent / 'models' / 'client_model.pth'
    if not model_path.exists():
        print(f"\nERREUR: Modèle non trouvé: {model_path}")
        print("Veuillez d'abord entraîner le modèle avec: python examples/train_model.py")
        return
    
    # Charger les données
    print("\n[1/4] Chargement des données...")
    data_path = Path(__file__).parent.parent / 'data' / 'raw' / 'client_data.csv'
    
    if not data_path.exists():
        print(f"ERREUR: Fichier de données non trouvé: {data_path}")
        print("Veuillez d'abord exécuter: python examples/generate_sample_data.py")
        return
    
    loader = ClientDataLoader(data_path=str(data_path), test_size=0.2, random_state=42)
    data = loader.load_data()
    print(f"   ✓ {len(data)} échantillons chargés")
    
    # Prétraiter les données
    print("\n[2/4] Prétraitement des données...")
    target_column = 'risque'
    feature_columns = ['age', 'revenu', 'anciennete', 'score_credit', 'depenses_mensuelles']
    
    X_train, X_test, y_train, y_test = loader.preprocess_data(
        data, target_column, feature_columns
    )
    
    # Créer les DataLoaders
    train_loader, test_loader = loader.create_dataloaders(
        X_train, X_test, y_train, y_test, batch_size=32
    )
    print(f"   ✓ Données prétraitées")
    
    # Charger le modèle
    print("\n[3/4] Chargement du modèle...")
    input_dim = len(feature_columns)
    model = ClientModel(
        input_dim=input_dim,
        hidden_dims=[64, 32, 16],
        output_dim=1,
        dropout_rate=0.2
    )
    
    trainer = ModelTrainer(model)
    trainer.load_model(str(model_path))
    print(f"   ✓ Modèle chargé avec {model.get_num_parameters()} paramètres")
    
    # Faire des prédictions
    print("\n[4/4] Prédictions sur l'ensemble de test...")
    predictions = trainer.predict(test_loader)
    
    # Afficher quelques exemples
    print("\n" + "=" * 60)
    print("Exemples de prédictions vs valeurs réelles:")
    print("=" * 60)
    print(f"{'Prédiction':<15} {'Réel':<15} {'Différence':<15}")
    print("-" * 60)
    
    for i in range(min(10, len(predictions))):
        pred = predictions[i][0]
        real = y_test[i][0]
        diff = abs(pred - real)
        print(f"{pred:<15.4f} {real:<15.4f} {diff:<15.4f}")
    
    # Calculer la performance
    import numpy as np
    mae = np.mean(np.abs(predictions - y_test))
    mse = np.mean((predictions - y_test) ** 2)
    rmse = np.sqrt(mse)
    
    print("\n" + "=" * 60)
    print("Métriques de performance:")
    print("=" * 60)
    print(f"MAE (Mean Absolute Error):  {mae:.4f}")
    print(f"MSE (Mean Squared Error):   {mse:.4f}")
    print(f"RMSE (Root Mean Squared):   {rmse:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
