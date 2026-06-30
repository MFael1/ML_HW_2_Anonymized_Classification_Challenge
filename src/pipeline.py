"""
pipeline.py — Machine Learning Pipeline definition.

This module provides the architectural definition of the preprocessing
and modeling pipeline, strictly utilizing scikit-learn's Pipeline and
ColumnTransformer for reproducible and leakage-free transformations.
"""

import logging
from typing import Any, Dict, List

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


def build_pipeline(cfg: Dict[str, Any], continuous_features: List[str], binary_features: List[str]) -> Pipeline:
    """
    Constructs an end-to-end scikit-learn Pipeline for classification.
    
    Dynamically builds the final estimator based on `cfg["ensemble_strategy"]`,
    allowing seamless switching between single models and ensembles.

    Args:
        cfg (Dict[str, Any]): Global configuration dictionary.
        continuous_features (List[str]): List of continuous feature column names.
        binary_features (List[str]): List of binary/categorical feature column names.

    Returns:
        Pipeline: A scikit-learn Pipeline instance ready to be fitted.
    """
    # Preprocessing: Scale continuous features, pass-through binary features
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), continuous_features),
            ("bin", "passthrough", binary_features)
        ],
        remainder="drop"
    )

    # 1. Define Base Models
    catboost_model = CatBoostClassifier(
        **cfg["model"]["catboost"],
        random_state=cfg["random_state"]
    )
    
    lgbm_model = LGBMClassifier(
        **cfg["model"]["lightgbm"],
        random_state=cfg["random_state"]
    )
    
    xgb_model = XGBClassifier(
        **cfg["model"]["xgboost"],
        random_state=cfg["random_state"]
    )

    strategy = cfg.get("ensemble_strategy", "single")
    logger.info("Pipeline building strategy selected: %s", strategy.upper())

    # 2. Select Final Estimator Architectures
    if strategy == "voting":
        classifier = VotingClassifier(
            estimators=[
                ("catboost", catboost_model),
                ("lightgbm", lgbm_model),
                ("xgboost", xgb_model)
            ],
            voting="soft" # Soft voting averages probabilities
        )
    elif strategy == "stacking":
        # Meta-learner is heavily regularized (C=0.1) to defend against LB overfitting
        meta_learner = LogisticRegression(
            max_iter=1000, 
            class_weight="balanced", 
            random_state=cfg["random_state"],
            C=0.1
        )
        classifier = StackingClassifier(
            estimators=[
                ("catboost", catboost_model),
                ("lightgbm", lgbm_model),
                ("xgboost", xgb_model)
            ],
            final_estimator=meta_learner,
            cv=3, # 3-fold internal CV to generate clean OOF predictions
            stack_method="predict_proba",
            n_jobs=-1
        )
    elif strategy == "single":
        primary = cfg["model"].get("primary_model", "catboost")
        if primary == "lightgbm":
            classifier = lgbm_model
        else:
            classifier = catboost_model
    else:
        raise ValueError(f"Unknown ensemble_strategy: {strategy}")

    # 3. Compile Pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    return pipeline
