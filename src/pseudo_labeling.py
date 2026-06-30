"""
pseudo_labeling.py — High-confidence test-set augmentation logic.

Implements the pseudo-labeling strategy to bridge the final variance
gap on the leaderboard by injecting highly-calibrated test-set predictions
back into the training pipeline.
"""

import logging
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

def apply_pseudo_labeling(
    trained_pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    cfg: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.Series, bool]:
    """
    Predicts on the test set, filters by high confidence, and augments the training set.

    Args:
        trained_pipeline (Pipeline): The fitted pipeline (e.g. VotingClassifier).
        X_train (pd.DataFrame): The original pruned training features.
        y_train (pd.Series): The original training labels.
        X_test (pd.DataFrame): The pruned test features.
        cfg (Dict[str, Any]): Configuration dictionary.

    Returns:
        Tuple containing:
            - X_train_ext (pd.DataFrame): The augmented training features.
            - y_train_ext (pd.Series): The augmented training labels.
            - was_applied (bool): True if pseudo-labeling actually found and added samples.
    """
    pl_cfg = cfg.get("pseudo_labeling", {})
    if not pl_cfg.get("enabled", False):
        return X_train, y_train, False
        
    confidence_threshold = pl_cfg.get("confidence_threshold", 0.99)
    logger.info("Applying Pseudo-Labeling with confidence threshold: >= %.4f", confidence_threshold)
    
    # 1. Get probabilities for the test set
    test_probs = trained_pipeline.predict_proba(X_test)
    max_probs = np.max(test_probs, axis=1)
    predictions = trained_pipeline.predict(X_test).flatten()
    
    # 2. Filter high confidence
    mask = max_probs >= confidence_threshold
    n_pseudo = mask.sum()
    
    if n_pseudo == 0:
        logger.warning("No test samples met the pseudo-labeling confidence threshold.")
        return X_train, y_train, False
        
    logger.info("Found %d highly confident test samples to inject as pseudo-labels.", n_pseudo)
    
    # 3. Construct Augmented Dataset
    X_pseudo = X_test[mask].copy()
    y_pseudo = pd.Series(predictions[mask], name=y_train.name)
    
    X_train_ext = pd.concat([X_train, X_pseudo], ignore_index=True)
    y_train_ext = pd.concat([y_train, y_pseudo], ignore_index=True)
    
    logger.info("Augmented training set size: %d -> %d", len(X_train), len(X_train_ext))
    
    return X_train_ext, y_train_ext, True
