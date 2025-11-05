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
- Entraîner le modèle pendant 50 époques
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