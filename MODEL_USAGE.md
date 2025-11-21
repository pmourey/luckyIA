# Mémento — Contexte du projet et réutilisation des modèles entraînés

Ce document résume le projet, explique comment réutiliser les modèles entraînés, et décrit une petite application Flask fournie pour faciliter l'inférence par un utilisateur final.

Raccourci des fichiers importants
- `ai_module/data_loader.py` : classes `ClientDataset`, `ClientDataLoader` (chargement, prétraitement, création de `DataLoader`, `StandardScaler`).
- `ai_module/model.py` : `ClientModel` et `ClientClassificationModel` (définition du réseau, `predict`, `predict_proba`).
- `ai_module/trainer.py` : `ModelTrainer` (entraînement, évaluation, prédiction, sauvegarde/chargement du checkpoint).
- `models/` : emplacement attendu des fichiers modèles (ex : `models/client_model.pth`) et du scaler (recommandé `models/scaler.joblib`).
- `examples/predict_example.py` : script d'exemple pour charger un modèle et faire des prédictions.

Contrat minimal (inputs / outputs)
- Entrée : CSV de données clients (ex : `data/raw/client_data.csv`). Le CSV doit contenir les colonnes features utilisées lors de l'entraînement. Par convention du dépôt d'exemple, la colonne cible s'appelle `risque` et les features d'exemple sont : `['age','revenu','anciennete','score_credit','depenses_mensuelles']`.
- Sortie : fichier checkpoint PyTorch (ex: `models/client_model.pth`) contenant un dict :
  - `model_state_dict` : poids du modèle
  - `train_losses`, `val_losses` (optionnel)
- Important : le transformateur de features (`StandardScaler`) doit être conservé pour assurer une inférence cohérente. Recommandation : sauvegarder `models/scaler.joblib` via `joblib`.

Installation & compatibilité PyTorch
- PyTorch publie des roues pour des versions spécifiques de Python. Si vous êtes sur Python 3.13 (comme dans votre environnement actuel), il est probable qu'il n'existe pas de wheel prêt à l'emploi.
- Recommandation : utiliser Python 3.11 (ou 3.10) via Miniconda/conda ou pyenv.
  - Avec conda :
    ```bash
    conda create -n luckyia python=3.11 -y
    conda activate luckyia
    conda install pytorch cpuonly -c pytorch -y
    pip install -r requirements.txt
    ```
  - Ou créer un venv basé sur `python3.11` puis installer PyTorch compatible.
- Si vous ne pouvez pas installer PyTorch, vous ne pourrez pas exécuter l'entraînement ni l'inférence PyTorch localement.

Sauvegarde recommandée du pipeline
- Sauvegarder ensemble : modèle + scaler
  - `trainer.save_model('models/client_model.pth')` pour le modèle
  - `joblib.dump(loader.scaler, 'models/scaler.joblib')` pour le scaler
- Lors du chargement, reconstruire l'architecture du modèle (même `input_dim`, `hidden_dims`) puis `model.load_state_dict(checkpoint['model_state_dict'])`.

Exemple rapide de code pour l'inférence (snippet)
```python
from pathlib import Path
import joblib
import numpy as np
import torch
from ai_module.model import ClientModel
from ai_module.trainer import ModelTrainer

# chemins
model_path = Path('models/client_model.pth')
scaler_path = Path('models/scaler.joblib')

# reconstruire le modèle : il faut connaître input_dim / hidden_dims
# simple méthode : déduire input_dim depuis le checkpoint (voir l'app Flask fournie)

checkpoint = torch.load(str(model_path), map_location='cpu')
# déduire la dimension d'entrée et des couches depuis checkpoint['model_state_dict'] si nécessaire
# ... (voir utilitaire web_app/app.py pour un exemple complet)

scaler = joblib.load(str(scaler_path))

# préparer les données (exemple one-row)
sample = np.array([[45, 45000, 3, 680, 1200]], dtype=float)
sample_scaled = scaler.transform(sample)

model = ClientModel(input_dim=5, hidden_dims=[64,32,16], output_dim=1)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
with torch.no_grad():
    out = model(torch.FloatTensor(sample_scaled))
    print('Prediction:', out.numpy())
```

Application Flask fournie
- Un prototype d'application est placé dans `web_app/` (serveur minimal + templates) :
  - `web_app/app.py` : application Flask
  - `web_app/templates/` : `layout.html`, `index.html`, `results.html`
  - `web_app/requirements.txt` : dépendances minimales pour l'app (Flask, joblib, pandas, scikit-learn)
- Fonctionnalités :
  - Charger un fichier CSV utilisateur
  - Lister les modèles disponibles dans `models/` et permettre d'en choisir un
  - Tenter de reconstruire automatiquement l'architecture du modèle depuis le checkpoint
  - Appliquer le scaler `models/scaler.joblib` s'il existe, sinon alerter l'utilisateur
  - Afficher les prédictions (et la colonne cible si présente)

Pièges courants et solutions
- Absence de `scaler.joblib` : l'inférence risque d'être incorrecte si vous normalisez différemment, sauvegardez toujours le scaler.
- Mismatch `input_dim` : reconstruisez l'architecture depuis le checkpoint ou enregistrez les paramètres d'architecture dans le checkpoint.
- Versions PyTorch différentes : gardez la même version en prod/entraînement quand possible.

Prochaines améliorations recommandées
- Ajouter dans les checkpoints les métadonnées d'architecture (input_dim, hidden_dims, feature_columns) pour une reconstruction fiable.
- Ajouter un utilitaire `ai_module/inference.py` qui encapsule la logique de chargement de modèle + scaler + prédiction.
- Ajouter tests unitaires pour la transformation et l'inférence.


---
Voir `web_app/README.md` pour des instructions d'exécution de l'application Flask fournie.

