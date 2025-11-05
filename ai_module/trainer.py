"""
Module pour l'entraînement du modèle
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from pathlib import Path


class ModelTrainer:
    """Gestionnaire d'entraînement du modèle"""
    
    def __init__(self, model, device=None):
        """
        Initialise le trainer
        
        Args:
            model: Modèle PyTorch à entraîner
            device: Device (cpu ou cuda)
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.train_losses = []
        self.val_losses = []
        
    def train(self, train_loader, val_loader=None, epochs=100, learning_rate=0.001, 
              criterion=None, optimizer=None, verbose=True):
        """
        Entraîne le modèle
        
        Args:
            train_loader: DataLoader pour les données d'entraînement
            val_loader: DataLoader pour les données de validation (optionnel)
            epochs: Nombre d'époques
            learning_rate: Taux d'apprentissage
            criterion: Fonction de perte (défaut: MSELoss)
            optimizer: Optimiseur (défaut: Adam)
            verbose: Afficher la progression
            
        Returns:
            Historique d'entraînement (dict avec train_losses et val_losses)
        """
        # Initialiser le critère et l'optimiseur
        if criterion is None:
            criterion = nn.MSELoss()
        
        if optimizer is None:
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Entraînement
        for epoch in range(epochs):
            # Mode entraînement
            self.model.train()
            train_loss = 0.0
            
            # Barre de progression
            iterator = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}') if verbose else train_loader
            
            for features, labels in iterator:
                # Déplacer vers le device
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Perte moyenne d'entraînement
            avg_train_loss = train_loss / len(train_loader)
            self.train_losses.append(avg_train_loss)
            
            # Validation
            if val_loader is not None:
                avg_val_loss = self.evaluate(val_loader, criterion, verbose=False)
                self.val_losses.append(avg_val_loss)
                
                if verbose:
                    print(f'Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}')
            else:
                if verbose:
                    print(f'Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}')
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
    
    def evaluate(self, data_loader, criterion=None, verbose=True):
        """
        Évalue le modèle sur un ensemble de données
        
        Args:
            data_loader: DataLoader pour l'évaluation
            criterion: Fonction de perte
            verbose: Afficher les résultats
            
        Returns:
            Perte moyenne
        """
        if criterion is None:
            criterion = nn.MSELoss()
        
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for features, labels in data_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(data_loader)
        
        if verbose:
            print(f'Evaluation Loss: {avg_loss:.4f}')
        
        return avg_loss
    
    def predict(self, data_loader):
        """
        Fait des prédictions sur un ensemble de données
        
        Args:
            data_loader: DataLoader pour les prédictions
            
        Returns:
            Numpy array avec les prédictions
        """
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for features, _ in data_loader:
                features = features.to(self.device)
                outputs = self.model(features)
                predictions.append(outputs.cpu().numpy())
        
        return np.vstack(predictions)
    
    def save_model(self, path):
        """
        Sauvegarde le modèle
        
        Args:
            path: Chemin de sauvegarde
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, path)
        print(f'Modèle sauvegardé: {path}')
    
    def load_model(self, path):
        """
        Charge un modèle sauvegardé
        
        Args:
            path: Chemin du modèle
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        print(f'Modèle chargé: {path}')
