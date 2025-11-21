"""Fonctions utilitaires pour générer des données et modèles de démonstration.
Regroupe la logique de `examples/generate_sample_data.py` et `examples/generate_demo_models.py`.
"""

from pathlib import Path
import numpy as np

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except Exception:
    pd = None
    PANDAS_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    SKLEARN_AVAILABLE = True
except Exception:
    RandomForestClassifier = None
    make_classification = None
    SKLEARN_AVAILABLE = False

try:
    from joblib import dump
    JOBLIB_AVAILABLE = True
except Exception:
    dump = None
    JOBLIB_AVAILABLE = False

# Définir SimpleNet au niveau module pour éviter les problèmes de pickling
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    class SimpleNet(nn.Module):
        def __init__(self, n_in, n_out=1):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_in, 16),
                nn.ReLU(),
                nn.Linear(16, n_out)
            )
        def forward(self, x):
            return self.net(x)


def generate_sample_data(output_path=None, n_samples=1000, random_state=42):
    """Génère des données clients synthétiques et sauvegarde un CSV dans data/raw.

    Args:
        output_path: chemin de sortie (Path ou str). Si None, utilise data/raw/client_data.csv
        n_samples: nombre d'échantillons
        random_state: seed

    Returns:
        Path vers le fichier CSV créé
    """
    np.random.seed(random_state)

    data = {
        'age': np.random.randint(18, 80, n_samples),
        'revenu': np.random.normal(50000, 20000, n_samples).clip(15000, 200000),
        'anciennete': np.random.randint(0, 30, n_samples),
        'score_credit': np.random.randint(300, 850, n_samples),
        'depenses_mensuelles': np.random.normal(2000, 800, n_samples).clip(500, 8000)
    }

    df = None
    if PANDAS_AVAILABLE:
        df = pd.DataFrame(data)
        df['risque'] = (
            0.3 * (800 - df['score_credit']) / 500 +
            0.2 * (df['depenses_mensuelles'] / df['revenu']) +
            0.15 * (70 - df['age']) / 50 +
            0.1 * np.random.random(n_samples)
        ).clip(0, 1)

    # Chemin par défaut : utiliser la racine du projet pour que l'emplacement soit montable en Docker
    if output_path is None:
        # __file__ = ai_module/examples/generate.py -> parents[2] = project root
        project_root = Path(__file__).resolve().parents[2]
        output_path = project_root / 'data' / 'raw' / 'client_data.csv'
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if PANDAS_AVAILABLE and df is not None:
        df.to_csv(output_path, index=False)
    else:
        # fallback : sauvegarder un CSV simple (sans la colonne cible)
        cols = ['age','revenu','anciennete','score_credit','depenses_mensuelles']
        header = ",".join(cols)
        X = np.vstack([data[c] for c in cols]).T
        np.savetxt(output_path, X, delimiter=",", header=header, comments='')

    return output_path


def generate_demo_models(root=None):
    """Génère un CSV d'exemple et un modèle scikit-learn (optionnel) + petit modèle PyTorch si disponible.

    Args:
        root: Racine du projet (Path ou str). Si None, déduit automatiquement.

    Returns:
        dict avec chemins créés
    """
    # Par défaut, choisir la racine du projet (parents[2]) pour écrire dans /models et /data
    if root is None:
        root = Path(__file__).resolve().parents[2]
    else:
        root = Path(root)

    models_dir = root / 'models'
    data_dir = root / 'data' / 'processed'
    models_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Générer/charger X,y
    if SKLEARN_AVAILABLE and make_classification is not None:
        X, y = make_classification(n_samples=200, n_features=5, n_informative=3, random_state=42)
    else:
        rng = np.random.default_rng(42)
        X = rng.normal(size=(200, 5))
        linear = X[:, 0] * 0.5 + X[:, 1] * -0.2 + X[:, 2] * 0.3
        y = (linear + rng.normal(scale=0.5, size=linear.shape) > 0).astype(int)

    cols = [f'feat_{i}' for i in range(X.shape[1])]

    # Sauver CSV d'entrée
    input_csv = data_dir / 'demo_input.csv'
    if PANDAS_AVAILABLE:
        df = pd.DataFrame(X, columns=cols)
        df.to_csv(input_csv, index=False)
    else:
        header = ",".join(cols)
        np.savetxt(input_csv, X, delimiter=",", header=header, comments='')

    results = {'input_csv': str(input_csv)}

    # Entraîner/save sklearn model
    if SKLEARN_AVAILABLE and RandomForestClassifier is not None and JOBLIB_AVAILABLE and dump is not None:
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        model_path = models_dir / 'demo_sklearn.joblib'
        dump(model, model_path)
        results['sklearn_model'] = str(model_path)

    # Essayer PyTorch
    try:
        if not TORCH_AVAILABLE:
            raise RuntimeError('torch indisponible')
        import torch
        import torch.nn as nn

        # Utiliser la classe SimpleNet définie au niveau module
        net = SimpleNet(X.shape[1], n_out=1)
        X_t = torch.from_numpy(X.astype('float32'))
        y_t = torch.from_numpy(y.astype('float32')).unsqueeze(1)
        opt = torch.optim.Adam(net.parameters(), lr=1e-2)
        loss_fn = nn.BCEWithLogitsLoss()
        net.train()
        for _ in range(10):
            opt.zero_grad()
            out = net(X_t)
            loss = loss_fn(out, y_t)
            loss.backward()
            opt.step()

        net.eval()
        torch_path = models_dir / 'demo_torch.pt'
        # Tenter de sauvegarder un module scriptable pour éviter les problèmes de pickling
        try:
            scripted = torch.jit.script(net)
            torch.jit.save(scripted, torch_path)
            results['torch_model'] = str(torch_path)
        except Exception:
            # Fallback: sauvegarder seulement le state_dict (plus sûr à recharger si on reconstruit l'arch)
            sd_path = models_dir / 'demo_torch_state_dict.pth'
            torch.save(net.state_dict(), sd_path)
            results['torch_model_state_dict'] = str(sd_path)
    except Exception as e:
        results['torch_error'] = str(e)

    return results
