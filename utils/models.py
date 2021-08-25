"""
Models module

This module contains the predictor class.
"""

from __future__ import annotations
import logging
import xgboost
import pandas as pd
import numpy as np

class GDPGrowthPredictor:
    """
    GPDGrowthPredictor.
    
    Model to make the predictions of the GDPGrowth.
    """

    def __init__(self):
        self._model = xgboost.XGBRegressor()

    def train(self, X: pd.DataFrame, y: pd.DataFrame, *args, **kwargs):
        """
        Train the model.
        
        Args:
            X (DataFrame): Features data.
            y (DataFrame): Target values.
        """
        
        self._model.fit(X, y, *args, **kwargs)
        logging.getLogger(__name__).info('Model trained')

    def predict(self, X: pd.DataFrame, *args, **kwargs) -> np.array:
        """
        Make predictions.
        
        Args:
            X (DataFrame): Features data.
        
        Returns:
            np.array: Predictions
        """
        
        logging.getLogger(__name__).info('Making predictions')
        return self._model.predict( X, *args, **kwargs)

    @staticmethod
    def load(filename: str) -> GDPGrowthPredictor:
        """
        Load model from file.
        
        Args:
            filename (str): Filename where the model is saved.
            
        Returns:
            GDPGrowthPredictor: The model loaded
        """
        
        model = GDPGrowthPredictor()
        model._model.load_model(filename)
        logging.getLogger(__name__).info(f"Model in '{filename}' loaded")
        return model

    def save(self, filename: str):
        """
        Save model to a file.
        
        Args:
            filename (str): Path to file where the model will be stored.
        """
        
        self._model.save_model(filename)
        logging.getLogger(__name__).info(f"Model saved to '{filename}'")
        
    def get_internal_model(self) -> xgboost.XGBRegressor:
        """
        Returns the internal model, i.e. the gradient boosting regressor.
        
        Returns:
            XGBRegressor: The internal model
        """
        
        return self._model
