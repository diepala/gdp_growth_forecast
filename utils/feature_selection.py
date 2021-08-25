"""
Module to select the most important features for a model.
"""

from typing import List
import logging
import pandas as pd
import shap

def select_features_with_shap(k: int, model, model_type: str, X: pd.DataFrame, y) -> List[str]:
    """
    Select the most relevant features for the `model`.
    
    This computes the shap values to see which features are more important.
    Selects the features whose shap values have greater variance.
    
    Args:
        k (int): Number of features to select
        model: Model that will be used to predict. Must implement scikit-learn API.
        model_type (str): Type of the model. At the moment only ``'tree'`` is supported.
            This includes decision trees, random-forests, and gradient boosting methods based on trees.
        X (DataFrame): Features data.
        y (array-like): Target variable data.
    
    Returns:
        List[str]: List with the names of the selected features.
    
    Raises:
        ValueError: If `k` is not positive.
            If `model_type` is not correct.
    """
    
    if k < 0:
        raise ValueError("Number of features k must be positive")
    if model_type not in ['tree']:
        raise ValueError(f"model_type '{model_type}' is not currently supported")

    model.fit(X, y) # train model
    
    # compute shap values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # select features with higher shap value variance
    features_idx = (-shap_values.var(axis=0)).argsort()[:k]
    feature_names = list(X.columns[features_idx])
    
    logging.getLogger(__name__).info(f"'{k}' most imoprtant features selected")

    return feature_names
    

