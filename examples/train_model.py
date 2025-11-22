"""
Script d'exemple pour entraîner un modèle avec le module IA
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from ai_module import examples as ex
except Exception as e:
    print('ai_module.examples not available:', e)
    raise SystemExit(1)

def main():
    """Fonction principale pour entraîner le modèle"""
    
    print("=" * 60)
    print("Formation d'un modèle IA avec PyTorch")
    print("=" * 60)
    
    # 1. Charger les données
    print("\n[1/5] Chargement des données...")
    data_path = str(Path(__file__).parent.parent / 'data' / 'raw' / 'client_data.csv')

    if not Path(data_path).exists():
        print(f"ERREUR: Fichier de données non trouvé: {data_path}")
        print("Veuillez d'abord exécuter: python examples/generate_sample_data.py")
        return
    
    # 2. Créer et entraîner le modèle
    print("\n[2/5] Création et entraînement du modèle...")
    res = ex.train_model_from_csv(data_path, target_column='risque', epochs=30)

    # 3. Évaluer le modèle
    print("\n[3/5] Évaluation du modèle...")
    test_loader = res.get('test_loader')
    trainer = res.get('trainer')
    test_loss = trainer.evaluate(test_loader)
    print(f"   ✓ Perte finale sur le test: {test_loss:.4f}")
    
    # Sauvegarder le modèle
    model_path = Path(__file__).parent.parent / 'models' / 'client_model.pth'
    trainer.save_model(str(model_path))
    
    print("\n" + "=" * 60)
    print("✓ Entraînement terminé avec succès!")
    print("=" * 60)
    print(f"\nModèle sauvegardé: {model_path}")
    print(f"Perte d'entraînement finale: {res.get('train_loss'):.4f}")
    print(f"Perte de validation finale: {res.get('val_loss'):.4f}")

if __name__ == "__main__":
    main()
