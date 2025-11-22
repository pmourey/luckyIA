"""
Module pour charger et prétraiter les données clients
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader


class ClientDataset(Dataset):
    """Dataset PyTorch pour les données clients"""
    
    def __init__(self, features, labels):
        """
        Initialise le dataset
        
        Args:
            features: Caractéristiques (numpy array)
            labels: Étiquettes (numpy array)
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class ClientDataLoader:
    """Gestionnaire de chargement et prétraitement des données clients"""
    
    def __init__(self, data_path=None, test_size=0.2, random_state=42):
        """
        Initialise le chargeur de données
        
        Args:
            data_path: Chemin vers le fichier CSV de données
            test_size: Proportion des données de test (défaut: 0.2)
            random_state: Graine aléatoire pour la reproductibilité
        """
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        
    def load_data(self, data_path=None):
        """
        Charge les données depuis un fichier CSV
        
        Args:
            data_path: Chemin vers le fichier CSV (optionnel)
            
        Returns:
            DataFrame pandas avec les données
        """
        path = data_path or self.data_path
        if path is None:
            raise ValueError("Chemin de données non spécifié")
        
        data = pd.read_csv(path)
        return data
    
    def preprocess_data(self, data, target_column, feature_columns=None):
        """
        Prétraite les données (normalisation, séparation features/labels)
        
        Args:
            data: DataFrame pandas
            target_column: Nom de la colonne cible
            feature_columns: Liste des colonnes de features (None = toutes sauf target)
            
        Returns:
            Tuple (X_train, X_test, y_train, y_test)
        """
        # Sélectionner les features
        if feature_columns is None:
            feature_columns = [col for col in data.columns if col != target_column]
        
        X = data[feature_columns].values
        y = data[target_column].values.reshape(-1, 1)
        
        # Séparer train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Normaliser les features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test
    
    def create_dataloaders(self, X_train, X_test, y_train, y_test, batch_size=32):
        """
        Crée les DataLoaders PyTorch
        
        Args:
            X_train, X_test, y_train, y_test: Données prétraitées
            batch_size: Taille des batches
            
        Returns:
            Tuple (train_loader, test_loader)
        """
        train_dataset = ClientDataset(X_train, y_train)
        test_dataset = ClientDataset(X_test, y_test)
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )
        
        return train_loader, test_loader
    
    def get_feature_dim(self, data, target_column, feature_columns=None):
        """
        Retourne le nombre de features
        
        Args:
            data: DataFrame pandas
            target_column: Nom de la colonne cible
            feature_columns: Liste des colonnes de features
            
        Returns:
            Nombre de features
        """
        if feature_columns is None:
            feature_columns = [col for col in data.columns if col != target_column]
        return len(feature_columns)
