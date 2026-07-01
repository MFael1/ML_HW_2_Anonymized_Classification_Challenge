"""
tune.py — Optuna Hyperparameter Tuning Script.

Runs a fast Bayesian optimization search to find the optimal
hyperparameters for both CatBoost and LightGBM, explicitly
targeting our pruned dataset and evaluating via Stratified K-Fold.
"""

import os
import logging
import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from config import CONFIG
from feature_selection import perform_feature_selection

# Disable wandb logging during massive parameter sweeps to avoid dashboard spam
os.environ["WANDB_MODE"] = "disabled"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def objective(trial: optuna.Trial) -> float:
    """Optuna objective function for tuning base models."""
    cfg = CONFIG
    
    # 1. Load Data
    df_train_raw = pd.read_csv(cfg["paths"]["train_csv"])
    X_train = df_train_raw.drop(columns=["target", "ID"])
    y_train = df_train_raw["target"]
    X_test = pd.read_csv(cfg["paths"]["test_csv"]).drop(columns=["ID"])
    
    # 2. Prune Data (Ensures we tune on the exact noise-free data we use in prod)
    fs_cfg = cfg.get("feature_selection", {})
    if fs_cfg.get("enabled", False):
        X_train, _, _, _ = perform_feature_selection(X_train, y_train, X_test, cfg)

    # 3. Choose Model Type to Tune
    model_type = trial.suggest_categorical("model_type", ["catboost", "lightgbm"])
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=cfg["random_state"])
    classes = sorted(y_train.unique())
    cv_logloss = []
    
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
        
        if model_type == "catboost":
            params = {
                "iterations": trial.suggest_int("iterations", 500, 2000, step=100),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
                "depth": trial.suggest_int("depth", 4, 10),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-1, 10.0, log=True),
                "auto_class_weights": "Balanced",
                "loss_function": "MultiClass",
                "verbose": 0,
                "random_state": cfg["random_state"]
            }
            model = CatBoostClassifier(**params)
            model.fit(X_tr, y_tr)
            
        elif model_type == "lightgbm":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 500, 2000, step=100),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                "class_weight": "balanced",
                "objective": "multiclass",
                "verbose": -1,
                "random_state": cfg["random_state"]
            }
            model = LGBMClassifier(**params)
            model.fit(X_tr, y_tr)
            
        preds = model.predict_proba(X_val)
        cv_logloss.append(log_loss(y_val, preds, labels=classes))
        
    return float(np.mean(cv_logloss))

def main():
    logger.info("Starting Optuna Hyperparameter Tuning...")
    study = optuna.create_study(direction="minimize")
    
    # Run 30 trials (takes roughly 1-2 minutes on small tabular data)
    study.optimize(objective, n_trials=30)
    
    logger.info("=== Tuning Complete ===")
    logger.info("Best Trial Log Loss: %.4f", study.best_trial.value)
    logger.info("Best Parameters:\n%s", study.best_trial.params)
    
    print("\n[ACTION REQUIRED]: Copy the parameters above into your src/config.py file!")

if __name__ == "__main__":
    main()
