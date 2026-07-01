"""
feature_selection.py — Automated feature pruning strategies.

Provides logic for isolating and removing noisy features using
permutation importance prior to the cross-validation and final modeling pipeline.
"""

import logging
from typing import Any, Dict, Tuple, List

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.inspection import permutation_importance

logger = logging.getLogger(__name__)

def perform_feature_selection(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    X_test: pd.DataFrame, 
    cfg: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """
    Evaluates permutation importance via a fast baseline and prunes features
    falling below the configured threshold. Recategorizes remaining features.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target labels.
        X_test (pd.DataFrame): Test features to be pruned symmetrically.
        cfg (Dict[str, Any]): Global configuration dictionary containing thresholds.

    Returns:
        Tuple containing:
            - Pruned X_train (pd.DataFrame)
            - Pruned X_test (pd.DataFrame)
            - continuous_features (List[str])
            - binary_features (List[str])
    """
    fs_cfg = cfg.get("feature_selection", {})
    threshold = fs_cfg.get("importance_threshold", 0.0)
    
    logger.info("Running pre-training feature selection (Threshold: > %s)...", threshold)
    
    # Fit a quick model strictly to extract importances
    fs_model = CatBoostClassifier(**cfg["model"]["catboost"], random_state=cfg["random_state"])
    fs_model.fit(X_train, y_train, verbose=0)
    
    result = permutation_importance(
        fs_model, X_train, y_train, 
        n_repeats=5, random_state=cfg["random_state"], n_jobs=-1
    )
    
    valid_features = []
    for i, col in enumerate(X_train.columns):
        if result.importances_mean[i] > threshold:
            valid_features.append(col)
            
    dropped_count = len(X_train.columns) - len(valid_features)
    logger.info("Dropped %d features due to low importance. Retaining %d features.", dropped_count, len(valid_features))
    
    # Prune datasets symmetrically
    X_train_pruned = X_train[valid_features]
    X_test_pruned = X_test[valid_features]
    
    # Re-categorize features after pruning
    binary_features = [col for col in X_train_pruned.columns if X_train_pruned[col].nunique() <= 2]
    continuous_features = [col for col in X_train_pruned.columns if col not in binary_features]
    
    return X_train_pruned, X_test_pruned, continuous_features, binary_features
