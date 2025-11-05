"""
Script d'exemple pour utiliser le module avec des données personnalisées
Ce script montre comment entraîner un modèle avec vos propres données CSV
"""

import sys
from pathlib import Path

# Ajouter le répertoire parent au path pour importer le module
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_module import ClientDataLoader, ClientModel, ModelTrainer

def train_with_custom_data(csv_path, target_col, feature_cols=None, 
                          epochs=50, batch_size=32, learning_rate=0.001):
    """
    Entraîne un modèle avec des données personnalisées
    
    Args:
        csv_path: Chemin vers le fichier CSV de données
        target_col: Nom de la colonne cible à prédire
        feature_cols: Liste des colonnes de features (None = toutes sauf target)
        epochs: Nombre d'époques d'entraînement
        batch_size: Taille des batches
        learning_rate: Taux d'apprentissage
    """
    
    print("=" * 70)
    print("Entraînement avec données personnalisées")
    print("=" * 70)
    
    # 1. Charger les données
    print(f"\n[1/5] Chargement des données depuis {csv_path}...")
    loader = ClientDataLoader(data_path=csv_path, test_size=0.2, random_state=42)
    data = loader.load_data()
    print(f"   ✓ {len(data)} échantillons chargés")
    print(f"   ✓ Colonnes disponibles: {list(data.columns)}")
    
    # 2. Prétraiter les données
    print("\n[2/5] Prétraitement des données...")
    X_train, X_test, y_train, y_test = loader.preprocess_data(
        data, target_col, feature_cols
    )
    
    # Déterminer les colonnes de features utilisées
    if feature_cols is None:
        feature_cols = [col for col in data.columns if col != target_col]
    
    print(f"   ✓ Train: {len(X_train)} échantillons")
    print(f"   ✓ Test: {len(X_test)} échantillons")
    print(f"   ✓ Features utilisées: {feature_cols}")
    print(f"   ✓ Colonne cible: {target_col}")
    
    # 3. Créer les DataLoaders
    print("\n[3/5] Création des DataLoaders...")
    train_loader, test_loader = loader.create_dataloaders(
        X_train, X_test, y_train, y_test, batch_size=batch_size
    )
    print(f"   ✓ DataLoaders créés (batch_size={batch_size})")
    
    # 4. Créer et entraîner le modèle
    print("\n[4/5] Création et entraînement du modèle...")
    input_dim = len(feature_cols)
    model = ClientModel(
        input_dim=input_dim,
        hidden_dims=[64, 32, 16],
        output_dim=1,
        dropout_rate=0.2
    )
    
    print(f"   ✓ Modèle créé: {input_dim} inputs -> [64,32,16] -> 1 output")
    print(f"   ✓ Paramètres: {model.get_num_parameters()}")
    
    trainer = ModelTrainer(model)
    
    print(f"\n   Entraînement en cours ({epochs} époques)...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=epochs,
        learning_rate=learning_rate,
        verbose=True
    )
    
    # 5. Évaluer le modèle
    print("\n[5/5] Évaluation finale...")
    test_loss = trainer.evaluate(test_loader)
    
    # Sauvegarder le modèle
    model_path = Path(__file__).parent.parent / 'models' / 'custom_model.pth'
    trainer.save_model(str(model_path))
    
    print("\n" + "=" * 70)
    print("✓ Entraînement terminé!")
    print("=" * 70)
    print(f"Modèle sauvegardé: {model_path}")
    print(f"Perte d'entraînement finale: {history['train_losses'][-1]:.6f}")
    print(f"Perte de validation finale: {history['val_losses'][-1]:.6f}")
    print(f"Perte de test finale: {test_loss:.6f}")
    print("=" * 70)

def main():
    """Fonction principale"""
    
    print("\n" + "=" * 70)
    print("Exemple d'utilisation avec données personnalisées")
    print("=" * 70)
    print("\nCe script montre comment utiliser le module AI avec vos propres données.")
    print("\nFormat attendu:")
    print("  - Fichier CSV avec en-têtes")
    print("  - Une colonne cible (variable à prédire)")
    print("  - Une ou plusieurs colonnes de features")
    print("\nPour cet exemple, nous utilisons les données de démonstration.")
    print("=" * 70)
    
    # Utiliser les données d'exemple pour la démonstration
    data_path = Path(__file__).parent.parent / 'data' / 'raw' / 'client_data.csv'
    
    if not data_path.exists():
        print(f"\nERREUR: Fichier de données non trouvé: {data_path}")
        print("Veuillez d'abord exécuter: python examples/generate_sample_data.py")
        return
    
    # Entraîner avec les données
    train_with_custom_data(
        csv_path=str(data_path),
        target_col='risque',
        feature_cols=['age', 'revenu', 'anciennete', 'score_credit', 'depenses_mensuelles'],
        epochs=30,
        batch_size=32,
        learning_rate=0.001
    )
    
    print("\n" + "=" * 70)
    print("Pour utiliser vos propres données:")
    print("=" * 70)
    print("""
from ai_module import ClientDataLoader, ClientModel, ModelTrainer

# Charger vos données
loader = ClientDataLoader(data_path='mes_donnees.csv')
data = loader.load_data()

# Prétraiter
X_train, X_test, y_train, y_test = loader.preprocess_data(
    data, 
    target_column='ma_cible',
    feature_columns=['feature1', 'feature2', 'feature3']
)

# Créer DataLoaders
train_loader, test_loader = loader.create_dataloaders(
    X_train, X_test, y_train, y_test, batch_size=32
)

# Créer et entraîner le modèle
model = ClientModel(input_dim=3, hidden_dims=[32, 16], output_dim=1)
trainer = ModelTrainer(model)
trainer.train(train_loader, test_loader, epochs=50)

# Sauvegarder
trainer.save_model('models/mon_modele.pth')
""")

if __name__ == "__main__":
    main()
