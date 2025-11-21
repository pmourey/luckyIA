"""Fonctions utilitaires pour faire des prédictions à partir d'un CSV et d'un modèle sauvegardé.
Regroupe la logique de `examples/predict_example.py`.
"""
from pathlib import Path
from typing import List, Optional

from ..data_loader import ClientDataLoader
from ..model import ClientModel
from ..trainer import ModelTrainer


def predict_from_csv(data_path: str, model_path: str, feature_columns: Optional[List[str]] = None):
    """Charge les données, prétraite, charge le modèle et retourne les prédictions et métriques.

    Args:
        data_path: chemin vers le CSV de données brutes
        model_path: chemin vers le modèle sauvegardé (.pth)
        feature_columns: colonnes à utiliser; si None, déduit automatiquement

    Returns:
        dict avec 'predictions' (numpy array), 'y_test' (numpy), et métriques
    """
    data_path = Path(data_path)
    model_path = Path(model_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Fichier de données non trouvé: {data_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Modèle non trouvé: {model_path}")

    loader = ClientDataLoader(data_path=str(data_path))
    df = loader.load_data()

    if feature_columns is None:
        # déduire toutes les colonnes non cibles; si 'risque' existe, on l'utilise comme target
        if 'risque' in df.columns:
            target = 'risque'
        else:
            # si pas de target, on retourne les features bruts
            target = None
        feature_columns = [c for c in df.columns if c != target]

    if 'risque' in df.columns:
        target = 'risque'
    else:
        target = None

    X_train, X_test, y_train, y_test = loader.preprocess_data(df, target_column=target if target else feature_columns[0], feature_columns=feature_columns)
    _, test_loader = loader.create_dataloaders(X_train, X_test, y_train, y_test, batch_size=32)

    input_dim = len(feature_columns)
    model = ClientModel(input_dim=input_dim, hidden_dims=[64,32,16], output_dim=1, dropout_rate=0.2)
    trainer = ModelTrainer(model)
    trainer.load_model(str(model_path))

    preds = trainer.predict(test_loader)

    # calculer métriques si y_test disponible
    import numpy as _np
    metrics = {}
    if y_test is not None and len(y_test) > 0:
        mae = _np.mean(_np.abs(preds - y_test))
        mse = _np.mean((preds - y_test) ** 2)
        rmse = _np.sqrt(mse)
        metrics = {'mae': float(mae), 'mse': float(mse), 'rmse': float(rmse)}

    return {'predictions': preds, 'y_test': y_test, 'metrics': metrics}

