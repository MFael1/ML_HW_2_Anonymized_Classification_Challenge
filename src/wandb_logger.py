"""
wandb_logger.py — Weights & Biases context manager.

One W&B run per main() execution, covering both cross-validation and the
final fit, rather than one run per `.fit()` call on the estimator.
"""

import logging
from typing import Any, Dict

import wandb

logger = logging.getLogger(__name__)


class WandbExperimentLogger:
    """
    A context manager to handle Weights & Biases lifecycle cleanly.

    Usage::

        with WandbExperimentLogger(cfg["use_wandb"], cfg["wandb_project"], cfg) as wl:
            wl.log({"metric": value})
            wl.log_artifact("submission.csv", name="submission", artifact_type="predictions")
    """

    def __init__(self, enable: bool, project_name: str, config: Dict[str, Any]):
        self.enable = enable
        self.project_name = project_name
        self.config = config
        self.run = None

    def __enter__(self) -> "WandbExperimentLogger":
        if self.enable:
            logger.info("Initialising Weights & Biases run...")
            self.run = wandb.init(project=self.project_name, config=self.config, reinit=True)
        return self

    def log(self, metrics: Dict[str, Any]) -> None:
        """Logs metrics only if W&B is enabled and a run is active."""
        if self.enable and self.run:
            wandb.log(metrics)

    def log_artifact(self, path: str, name: str, artifact_type: str) -> None:
        """Registers a file (submission CSV, fitted pipeline) as a W&B artifact."""
        if self.enable and self.run:
            artifact = wandb.Artifact(name=name, type=artifact_type)
            artifact.add_file(path)
            self.run.log_artifact(artifact)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.enable and self.run:
            logger.info("Finishing Weights & Biases run.")
            wandb.finish()
