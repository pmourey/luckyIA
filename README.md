# luckyIA

Module d'Intelligence Artificielle en Python pour la formation de modèles avec PyTorch à partir de données clients.

## 📋 Description

Ce projet fournit un module IA complet pour:
- Charger et prétraiter des données clients
- Créer et entraîner des modèles de réseaux de neurones avec PyTorch
- Évaluer et sauvegarder les modèles entraînés

## 🚀 Installation

### Prérequis
- Python 3.8 ou supérieur
- pip

### Installation des dépendances

```bash
pip install -r requirements.txt
```

## 📁 Structure du projet

```
luckyIA/
├── ai_module/              # Module IA principal
│   ├── __init__.py        # Initialisation du module
│   ├── data_loader.py     # Chargement et prétraitement des données
│   ├── model.py           # Définition des modèles PyTorch
│   └── trainer.py         # Entraînement et évaluation
├── examples/              # Scripts d'exemple
│   ├── generate_sample_data.py  # Génération de données de test
│   └── train_model.py     # Exemple d'entraînement
├── data/                  # Répertoire des données
│   ├── raw/              # Données brutes
│   └── processed/        # Données traitées
├── models/               # Modèles sauvegardés
├── requirements.txt      # Dépendances Python
└── README.md            # Ce fichier
```

## 🎯 Utilisation

### 1. Générer des données d'exemple

```bash
python examples/generate_sample_data.py
```

Cela créera un fichier `data/raw/client_data.csv` avec 1000 échantillons de données clients synthétiques.

### 2. Entraîner un modèle

```bash
python examples/train_model.py
```

Ce script va:
- Charger les données clients
- Prétraiter et normaliser les features
- Créer un modèle de réseau de neurones
- Entraîner le modèle pendant 50 *epoc*
- Évaluer les performances
- Sauvegarder le modèle dans `models/client_model.pth`

### 3. Utiliser le module dans votre code

```python
from ai_module import ClientDataLoader, ClientModel, ModelTrainer

# Charger les données
loader = ClientDataLoader(data_path='data/raw/client_data.csv')
data = loader.load_data()

# Prétraiter
X_train, X_test, y_train, y_test = loader.preprocess_data(
    data, 
    target_column='risque',
    feature_columns=['age', 'revenu', 'anciennete', 'score_credit', 'depenses_mensuelles']
)

# Créer les DataLoaders
train_loader, test_loader = loader.create_dataloaders(
    X_train, X_test, y_train, y_test, batch_size=32
)

# Créer le modèle
model = ClientModel(input_dim=5, hidden_dims=[64, 32, 16], output_dim=1)

# Entraîner
trainer = ModelTrainer(model)
history = trainer.train(
    train_loader=train_loader,
    val_loader=test_loader,
    epochs=50,
    learning_rate=0.001
)

# Sauvegarder
trainer.save_model('models/my_model.pth')
```

## 🔧 Composants du module

### ClientDataLoader
Gestionnaire de chargement et prétraitement des données:
- Chargement depuis CSV
- Normalisation des features (StandardScaler)
- Séparation train/test
- Création de DataLoaders PyTorch

### ClientModel
Modèle de réseau de neurones configurable:
- Architecture personnalisable (couches cachées)
- Dropout pour la régularisation
- Support pour régression et classification

### ModelTrainer
Gestionnaire d'entraînement:
- Entraînement avec suivi des métriques
- Évaluation sur données de validation
- Sauvegarde/chargement de modèles
- Support GPU automatique

## 📊 Exemple de données

Le script `generate_sample_data.py` génère des données clients simulées avec:
- **age**: Âge du client (18-80 ans)
- **revenu**: Revenu annuel (15k-200k)
- **anciennete**: Ancienneté en années (0-30)
- **score_credit**: Score de crédit (300-850)
- **depenses_mensuelles**: Dépenses mensuelles (500-8000)
- **risque**: Variable cible - risque de défaut (0-1)

### Détail de la colonne `risque`

La colonne `risque` correspond à la variable cible (target) utilisée pour l'entraînement et l'évaluation des modèles. Voici les précisions importantes :

- Rôle : indique le niveau de risque associé à un client (par ex. risque de défaut de paiement). Le modèle apprend à prédire `risque` à partir des features (age, revenu, anciennete, score_credit, depenses_mensuelles, ...).
- Formats acceptés :
  - Binaire (0 / 1) — classification binaire (1 = événement à risque). Utilisé avec des classifieurs (RandomForestClassifier, LogisticRegression, réseaux avec sortie sigmoïde).
  - Probabilité (float entre 0.0 et 1.0) — sortie probabiliste. On évaluera la calibration et on appliquera un seuil pour la décision finale.
  - Score continu (ex. 0..100) — possible mais nécessite normalisation/transformations selon l'objectif.
- Dans l'UI / pipeline :
  - Pour l'entraînement, le CSV doit contenir la colonne `risque` afin que le script d'entraînement puisse séparer target/features.
  - Pour la prédiction (inférence), l'utilisateur final doit téléverser un CSV contenant uniquement les features (sans `risque`). Si la colonne `risque` est présente dans un CSV envoyé pour prédiction, l'application essaie de la détecter et de la retirer automatiquement (colonnes candidates détectées : `risque`, `target`, `label`, `Unnamed: 0`, `index`).
- Bonnes pratiques :
  - Stocker `risque` en numérique (int 0/1 pour classes, float pour probabilités).
  - Vérifier les valeurs manquantes et appliquer une politique d'imputation ou de suppression.
  - Si vous utilisez des probabilités, vérifier la calibration et choisir un seuil métier pour la conversion en classe.

## 🛠️ Technologies utilisées

- **PyTorch**: Framework de deep learning
- **NumPy**: Calcul numérique
- **Pandas**: Manipulation de données
- **Scikit-learn**: Prétraitement et métriques
- **Matplotlib**: Visualisation
- **tqdm**: Barres de progression

## 📝 Licence

Ce projet est open source et disponible sous licence MIT.

## 👥 Contribution

Les contributions sont les bienvenues! N'hésitez pas à ouvrir une issue ou une pull request.

## 🕸️ Interface Web (Flask)

Une application Flask minimale est fournie dans `web_app/` pour permettre à un utilisateur final de :
- téléverser un fichier CSV client,
- choisir un modèle parmi ceux présents dans le dossier `models/`,
- lancer une prédiction et voir les résultats sous forme de tableau,
- générer des données d'exemple et des modèles de démonstration depuis l'UI,
- entraîner un modèle directement depuis un CSV uploadé.

Commandes rapides :

```bash
# Démarrer l'application (depuis la racine du projet)
cd web_app
python app.py
```

URL par défaut : http://127.0.0.1:5000/

Pages disponibles :
- / : page d'accueil pour téléverser un CSV et choisir un modèle
- /generate-data : générer et sauvegarder un CSV d'exemple
- /generate-model : générer des modèles de démonstration (sklearn + PyTorch si disponible)
- /train-model : téléverser un CSV et entraîner un modèle (sauvegardé dans `models/`)


## 🧰 Utilitaires `ai_module.examples`

Le package expose des utilitaires prêts à l'emploi pour la génération, l'entraînement et la prédiction. Exemple d'utilisation depuis Python :

```python
from ai_module import examples as ex

# Générer des données clients synthétiques (sauvegarde par défaut dans data/raw/client_data.csv)
ex.generate_sample_data(n_samples=1000, random_state=42)

# Créer des modèles et un CSV d'entrée de démonstration (models/demo_sklearn.joblib, models/demo_torch.pt si torch disponible)
res = ex.generate_demo_models()
print(res)

# Entraîner un modèle à partir d'un CSV
res = ex.train_model_from_csv('data/raw/client_data.csv', target_column='risque', epochs=30)
print('Model saved at', res['model_path'])

# Faire une prédiction depuis Python
out = ex.predict_from_csv('data/raw/client_data.csv', model_path='models/client_model.pth')
print(out['metrics'])
```

Remarques importantes :
- Certains utilitaires nécessitent `scikit-learn`, `joblib` et/ou `torch` : si ces paquets manquent, les fonctions feront des fallbacks (ou retourneront des messages d'erreur clairs). Installez-les via pip si nécessaire.
- Pour PyTorch, suivez les instructions officielles si une roue binaire n'est pas disponible pour votre version de Python : https://pytorch.org
