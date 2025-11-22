"""
AI Module pour la formation de modèles avec PyTorch
"""

from .data_loader import ClientDataLoader
from .model import ClientModel
from .trainer import ModelTrainer

# Exposer les utilitaires d'exemples
from . import examples

__version__ = "1.0.0"
__all__ = ["ClientDataLoader", "ClientModel", "ModelTrainer", "examples"]
