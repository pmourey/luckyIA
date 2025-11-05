"""
Script d'exemple pour entraîner un modèle avec le module IA
"""

import sys
from pathlib import Path

# Ajouter le répertoire parent au path pour importer le module
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_module import ClientDataLoader, ClientModel, ModelTrainer
import torch

def main():
    """Fonction principale pour entraîner le modèle"""
    
    print("=" * 60)
    print("Formation d'un modèle IA avec PyTorch")
    print("=" * 60)
    
    # 1. Charger les données
    print("\n[1/5] Chargement des données...")
    data_path = Path(__file__).parent.parent / 'data' / 'raw' / 'client_data.csv'
    
    if not data_path.exists():
        print(f"ERREUR: Fichier de données non trouvé: {data_path}")
        print("Veuillez d'abord exécuter: python examples/generate_sample_data.py")
        return
    
    loader = ClientDataLoader(data_path=str(data_path), test_size=0.2, random_state=42)
    data = loader.load_data()
    print(f"   ✓ {len(data)} échantillons chargés")
    
    # 2. Prétraiter les données
    print("\n[2/5] Prétraitement des données...")
    target_column = 'risque'
    feature_columns = ['age', 'revenu', 'anciennete', 'score_credit', 'depenses_mensuelles']
    
    X_train, X_test, y_train, y_test = loader.preprocess_data(
        data, target_column, feature_columns
    )
    print(f"   ✓ Train: {len(X_train)} échantillons")
    print(f"   ✓ Test: {len(X_test)} échantillons")
    
    # 3. Créer les DataLoaders
    print("\n[3/5] Création des DataLoaders...")
    train_loader, test_loader = loader.create_dataloaders(
        X_train, X_test, y_train, y_test, batch_size=32
    )
    print(f"   ✓ DataLoaders créés (batch_size=32)")
    
    # 4. Créer et entraîner le modèle
    print("\n[4/5] Création et entraînement du modèle...")
    input_dim = len(feature_columns)
    model = ClientModel(
        input_dim=input_dim,
        hidden_dims=[64, 32, 16],
        output_dim=1,
        dropout_rate=0.2
    )
    
    print(f"   ✓ Modèle créé avec {model.get_num_parameters()} paramètres")
    print(f"   ✓ Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    
    trainer = ModelTrainer(model)
    
    print("\n   Entraînement en cours...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=50,
        learning_rate=0.001,
        verbose=True
    )
    
    # 5. Évaluer le modèle
    print("\n[5/5] Évaluation du modèle...")
    test_loss = trainer.evaluate(test_loader)
    print(f"   ✓ Perte finale sur le test: {test_loss:.4f}")
    
    # Sauvegarder le modèle
    model_path = Path(__file__).parent.parent / 'models' / 'client_model.pth'
    trainer.save_model(str(model_path))
    
    print("\n" + "=" * 60)
    print("✓ Entraînement terminé avec succès!")
    print("=" * 60)
    print(f"\nModèle sauvegardé: {model_path}")
    print(f"Perte d'entraînement finale: {history['train_losses'][-1]:.4f}")
    print(f"Perte de validation finale: {history['val_losses'][-1]:.4f}")

if __name__ == "__main__":
    main()
