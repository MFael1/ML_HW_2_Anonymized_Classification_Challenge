"""
training.py — Cross-Validation and evaluation logic.

Provides robust cross-validation mechanics, ensuring stratified splits
and robust evaluation tracking across multiple metrics.
"""

import logging
from typing import Any, Dict, Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, f1_score
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from src.wandb_logger import WandbExperimentLogger

logger = logging.getLogger(__name__)


def run_cross_validation(
    X: pd.DataFrame, 
    y: pd.Series, 
    pipeline_template: Pipeline, 
    cfg: Dict[str, Any], 
    wandb_logger: WandbExperimentLogger
) -> Tuple[float, float]:
    """
    Executes a Stratified K-Fold cross-validation loop.

    Iterates through the data splits, fits a fresh clone of the pipeline on
    each training fold, and evaluates it on the validation fold. It logs
    metrics securely both locally and via Weights & Biases.

    Args:
        X (pd.DataFrame): The feature matrix for training.
        y (pd.Series): The target variable.
        pipeline_template (Pipeline): An unfitted scikit-learn Pipeline 
            that defines the preprocessing and modeling steps.
        cfg (Dict[str, Any]): Global configuration dictionary containing CV settings.
        wandb_logger (WandbExperimentLogger): W&B context manager for logging.

    Returns:
        Tuple[float, float]: A tuple containing the mean Log Loss and mean 
        Macro F1-score across all cross-validation folds.

    Raises:
        ValueError: If the number of splits is less than 2.
    """
    n_splits = cfg["cv"].get("n_splits", 5)
    if n_splits < 2:
        raise ValueError("Number of CV splits must be at least 2.")

    logger.info("Starting %d-fold Stratified CV...", n_splits)
    
    skf = StratifiedKFold(
        n_splits=n_splits, 
        shuffle=cfg["cv"].get("shuffle", True), 
        random_state=cfg["random_state"]
    )
    
    classes = sorted(y.unique())
    cv_logloss: List[float] = []
    cv_macro_f1: List[float] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        # Clone the pipeline to ensure clean state per fold
        fold_pipeline = clone(pipeline_template)
        
        # Fit on training data
        fold_pipeline.fit(X_train, y_train)
        
        # Predict on validation data
        val_preds_proba = fold_pipeline.predict_proba(X_val)
        val_preds_class = fold_pipeline.predict(X_val).flatten()
        
        # Calculate metrics
        fold_logloss = log_loss(y_val, val_preds_proba, labels=classes)
        fold_f1 = f1_score(y_val, val_preds_class, average="macro")
        
        cv_logloss.append(fold_logloss)
        cv_macro_f1.append(fold_f1)
        
        logger.info(
            "Fold %d - Log Loss: %.4f | Macro F1: %.4f", 
            fold + 1, fold_logloss, fold_f1
        )
        
        # Log individual fold metrics to create a progressive chart in W&B
        wandb_logger.log({
            "fold": fold + 1,
            "fold_log_loss": fold_logloss,
            "fold_macro_f1": fold_f1
        })

    mean_logloss = float(np.mean(cv_logloss))
    mean_macro_f1 = float(np.mean(cv_macro_f1))
    std_logloss = float(np.std(cv_logloss))
    std_macro_f1 = float(np.std(cv_macro_f1))

    logger.info(
        "CV Results - Mean Log Loss: %.4f ± %.4f | Mean Macro F1: %.4f ± %.4f",
        mean_logloss, std_logloss, mean_macro_f1, std_macro_f1
    )

    wandb_logger.log({
        "cv/mean_log_loss": mean_logloss,
        "cv/std_log_loss": std_logloss,
        "cv/mean_macro_f1": mean_macro_f1,
        "cv/std_macro_f1": std_macro_f1,
    })

    return mean_logloss, mean_macro_f1
