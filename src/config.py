"""
config.py — Global experiment configuration.
"""

from typing import Any, Dict

CONFIG: Dict[str, Any] = {
    "random_state": 42,

    "paths": {
        "train_csv": "./dataset/train_data.csv",
        "test_csv": "./dataset/test_data.csv",
        "submission_csv": "./submission.csv",
        "artifact_path": "./artifacts/classification_pipeline_v2.joblib",
    },

    "cv": {"n_splits": 5, "shuffle": True},

    # Feature selection drops features with permutation importance <= threshold
    "feature_selection": {
        "enabled": True,
        "importance_threshold": 0.0000
    },

    # Pseudo-Labeling Configuration
    "pseudo_labeling": {
        "enabled": False, # Disabled due to confirmation bias
        "confidence_threshold": 0.995
    },

    # Strategy controls how the pipeline is constructed: "single", "voting", or "stacking"
    "ensemble_strategy": "stacking", 
    
    "model": {
        # Used if ensemble_strategy == "single"
        "primary_model": "catboost",
        
        "catboost": {
            "iterations": 1100, 
            "learning_rate": 0.086, 
            "depth": 7,  
            "l2_leaf_reg": 1.98,
            "loss_function": "MultiClass",
            "eval_metric": "MultiClass",
            "auto_class_weights": "Balanced", 
            "verbose": 0,
            "early_stopping_rounds": 50,
        },
        "lightgbm": {
            "n_estimators": 1000,  
            "learning_rate": 0.05, 
            "num_leaves": 31, 
            "objective": "multiclass",
            "metric": "multi_logloss",
            "class_weight": "balanced",
            "verbose": -1,
        },
        "xgboost": {
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "max_depth": 6,
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "verbosity": 0
        },
    },

    "use_wandb": True,
    "wandb_project": "fite-classification-competition",
}
