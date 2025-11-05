"""
Module de définition du modèle PyTorch
"""

import torch
import torch.nn as nn


class ClientModel(nn.Module):
    """Modèle de réseau de neurones pour les données clients"""
    
    def __init__(self, input_dim, hidden_dims=None, output_dim=1, dropout_rate=0.2):
        """
        Initialise le modèle
        
        Args:
            input_dim: Dimension d'entrée (nombre de features)
            hidden_dims: Liste des dimensions des couches cachées (défaut: [64, 32])
            output_dim: Dimension de sortie (nombre de classes ou valeurs à prédire)
            dropout_rate: Taux de dropout pour la régularisation
        """
        super(ClientModel, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [64, 32]
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Construire les couches
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Couche de sortie
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Propagation avant
        
        Args:
            x: Tensor d'entrée
            
        Returns:
            Prédictions du modèle
        """
        return self.network(x)
    
    def predict(self, x):
        """
        Fait des prédictions (mode évaluation)
        
        Args:
            x: Tensor d'entrée
            
        Returns:
            Prédictions
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x)
    
    def get_num_parameters(self):
        """Retourne le nombre de paramètres du modèle"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ClientClassificationModel(ClientModel):
    """Modèle spécialisé pour la classification"""
    
    def __init__(self, input_dim, hidden_dims=None, num_classes=2, dropout_rate=0.2):
        """
        Initialise le modèle de classification
        
        Args:
            input_dim: Dimension d'entrée
            hidden_dims: Dimensions des couches cachées (défaut: [64, 32])
            num_classes: Nombre de classes
            dropout_rate: Taux de dropout
        """
        if hidden_dims is None:
            hidden_dims = [64, 32]
        
        super(ClientClassificationModel, self).__init__(
            input_dim, hidden_dims, num_classes, dropout_rate
        )
        
        # Ajouter softmax pour la classification multi-classe
        if num_classes > 1:
            self.softmax = nn.Softmax(dim=1)
    
    def predict_proba(self, x):
        """
        Prédit les probabilités de classe
        
        Args:
            x: Tensor d'entrée
            
        Returns:
            Probabilités pour chaque classe
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            if self.output_dim > 1:
                return self.softmax(logits)
            else:
                return torch.sigmoid(logits)
