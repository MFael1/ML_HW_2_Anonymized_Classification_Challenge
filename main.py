"""
main.py — Application entry point.

Orchestrates the data loading, pipeline execution, cross-validation,
and final model inference for the classification competition.
"""

import logging
import os
import joblib

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss, f1_score
from sklearn.inspection import permutation_importance
from catboost import CatBoostClassifier

from src.config import CONFIG
from src.pipeline import build_pipeline
from src.training import run_cross_validation
from src.feature_selection import perform_feature_selection
from src.pseudo_labeling import apply_pseudo_labeling
from src.wandb_logger import WandbExperimentLogger

log_level = logging.DEBUG if os.environ.get("PIPELINE_DEBUG") == "1" else logging.INFO
logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
logger = logging.getLogger(__name__)

# Replace these with real credentials if necessary, or load from environment variables
os.environ["WANDB_API_KEY"] = "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"

def main() -> None:
    """
    Main orchestration routine.
    
    Responsibilities:
      1. Data ingestion from configured paths.
      2. Feature categorisation (continuous vs binary).
      3. Orchestrating cross-validation via src.training.
      4. Fitting the final pipeline on all training data.
      5. Producing test set predictions and saving artifacts.
    """
    cfg = CONFIG
    logger.info("Initiating classification pipeline execution...")
    
    train_path = cfg["paths"]["train_csv"]
    test_path = cfg["paths"]["test_csv"]
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        logger.error("Required dataset files are missing. Paths: %s, %s", train_path, test_path)
        return

    # 1. Data Ingestion
    df_train_raw = pd.read_csv(train_path)
    df_test_raw = pd.read_csv(test_path)

    X_train = df_train_raw.drop(columns=["target", "ID"])
    y_train = df_train_raw["target"]
    X_test = df_test_raw.drop(columns=["ID"])

    # 2. Initial Feature Categorisation (Fallback)
    binary_features = [col for col in X_train.columns if X_train[col].nunique() <= 2]
    continuous_features = [col for col in X_train.columns if col not in binary_features]
    
    # 2.5 Optional Feature Selection Stage
    fs_cfg = cfg.get("feature_selection", {})
    if fs_cfg.get("enabled", False):
        X_train, X_test, continuous_features, binary_features = perform_feature_selection(
            X_train, y_train, X_test, cfg
        )

    logger.info(
        "Final feature segmentation. Continuous: %d, Binary: %d", 
        len(continuous_features), len(binary_features)
    )

    with WandbExperimentLogger(cfg["use_wandb"], cfg["wandb_project"], cfg) as wandb_logger:
        
        # 3. Pipeline Construction
        pipeline_template = build_pipeline(cfg, continuous_features, binary_features)
        
        # 4. Cross-Validation
        run_cross_validation(X_train, y_train, pipeline_template, cfg, wandb_logger)

        # 5. Final Model Fit (Phase 1)
        logger.info("Fitting the final pipeline on the complete training set...")
        pipeline_template.fit(X_train, y_train)
        
        # 6. Pseudo-Labeling (Avenue 3)
        X_train_ext, y_train_ext, pl_applied = apply_pseudo_labeling(
            pipeline_template, X_train, y_train, X_test, cfg
        )
        
        # If pseudo-labeling augmented our data, we must retrain the pipeline from scratch
        if pl_applied:
            logger.info("Retraining the pipeline entirely on the augmented dataset...")
            # We rebuild the pipeline so it's fresh
            pipeline_template = build_pipeline(cfg, continuous_features, binary_features)
            pipeline_template.fit(X_train_ext, y_train_ext)
            # Switch the reference so permutation importance uses the augmented data
            X_train = X_train_ext
            y_train = y_train_ext
        
        # Extract and log Permutation Importance from the final model
        logger.info("Calculating Permutation Importance (this may take a moment)...")
        result = permutation_importance(
            pipeline_template, X_train, y_train, 
            n_repeats=5, random_state=cfg["random_state"], n_jobs=-1
        )
        
        feature_names = X_train.columns.tolist()
        fi_df = pd.DataFrame({"Feature": feature_names, "Importance": result.importances_mean})
        fi_df = fi_df.sort_values(by="Importance", ascending=False)
        
        logger.info("Top Permutation Importances:\n%s", fi_df.head(10).to_string(index=False))
        
        import wandb
        if wandb_logger.run is not None:
            wandb_logger.run.log({"permutation_importance_table": wandb.Table(dataframe=fi_df)})
            wandb_logger.run.log({
                "permutation_importance_plot": wandb.plot.bar(
                    wandb.Table(dataframe=fi_df),
                    "Feature", "Importance", title="Permutation Importance"
                )
            })

        logger.info("Generating predictions for the test set...")
        test_predictions = pipeline_template.predict(X_test).flatten()
        
        submission = pd.DataFrame({"ID": df_test_raw["ID"], "target": test_predictions})
        submission_path = cfg["paths"]["submission_csv"]
        submission.to_csv(submission_path, index=False)
        logger.info("Submission successfully saved to %s", submission_path)

        # Persist State
        artifact_path = cfg["paths"]["artifact_path"]
        os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
        joblib.dump(pipeline_template, artifact_path)
        logger.info("Fitted pipeline securely persisted to %s", artifact_path)
        
        wandb_logger.log_artifact(submission_path, name="submission", artifact_type="predictions")
        wandb_logger.log_artifact(artifact_path, name="model_pipeline", artifact_type="model")

    logger.info("Pipeline execution completed successfully.")

if __name__ == "__main__":
    main()
