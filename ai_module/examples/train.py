"""Fonctions utilitaires pour entraîner des modèles à partir de CSV.
Regroupe la logique de `examples/train_model.py` et `examples/custom_data_example.py`.
"""
from pathlib import Path
from typing import List, Optional

from ..data_loader import ClientDataLoader
from ..model import ClientModel
from ..trainer import ModelTrainer


def train_model_from_csv(data_path: str, target_column: str = 'risque',
                         feature_columns: Optional[List[str]] = None,
                         model_output_path: Optional[str] = None,
                         epochs: int = 50, batch_size: int =32, learning_rate: float = 0.001):
    """Charge un CSV, prétraite, crée un modèle, l'entraîne et le sauvegarde.

    Args:
        data_path: chemin vers le CSV
        target_column: nom de la colonne cible
        feature_columns: liste des colonnes à utiliser; si None, toutes sauf target
        model_output_path: chemin de sauvegarde du modèle (si None -> models/client_model.pth)
        epochs, batch_size, learning_rate: hyperparamètres

    Returns:
        dict avec info sur l'entraînement
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Fichier de données non trouvé: {data_path}")

    loader = ClientDataLoader(data_path=str(data_path))
    df = loader.load_data()

    if feature_columns is None:
        feature_columns = [c for c in df.columns if c != target_column]

    X_train, X_test, y_train, y_test = loader.preprocess_data(df, target_column, feature_columns)
    train_loader, test_loader = loader.create_dataloaders(X_train, X_test, y_train, y_test, batch_size=batch_size)

    input_dim = len(feature_columns)
    model = ClientModel(input_dim=input_dim, hidden_dims=[64,32,16], output_dim=1, dropout_rate=0.2)
    trainer = ModelTrainer(model)
    history = trainer.train(train_loader, val_loader=test_loader, epochs=epochs, learning_rate=learning_rate, verbose=True)

    if model_output_path is None:
        model_output_path = Path(__file__).parent.parent / 'models' / 'client_model.pth'
    else:
        model_output_path = Path(model_output_path)

    trainer.save_model(str(model_output_path))

    return {
        'model_path': str(model_output_path),
        'history': history,
        'test_loss': trainer.evaluate(test_loader, verbose=False)
    }


def train_with_custom_data(csv_path, target_col, feature_cols=None, epochs=30, batch_size=32, learning_rate=0.001):
    """Wrapper compatible avec examples/custom_data_example.train_with_custom_data signature"""
    return train_model_from_csv(data_path=csv_path, target_column=target_col, feature_columns=feature_cols,
                                epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)

