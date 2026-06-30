# Anonymized Classification Challenge — Project Context

This document describes the full problem, requirements, and plan for this project.
Read this fully before doing any work — it should be treated as the source of truth
for what needs to be built.

## 1. Competition Overview

This is a multi-class classification problem on a Kaggle-style competition. The
dataset contains 21 fully anonymized features (mixed numeric and categorical types,
not labeled as such — we have to infer it). The true meaning, units, and scale of the
features are unknown and not meant to be reverse-engineered. The target is categorical
with multiple classes (e.g. `class1`, `class2`, `class3`).

Because nothing is known about what the features represent, success depends on:
- Careful exploratory data analysis to infer feature types and structure
- Solid, leak-free validation strategy (to avoid overfitting to the public leaderboard)
- Strong tabular modeling (gradient boosting is the expected approach)
- Proper experiment tracking and reproducibility (MLOps requirements, see below)

## 2. Data Files

| File | Description |
|---|---|
| `train_data.csv` | Training set: `ID`, 21 anonymized features, `target` (ground truth label) |
| `test_data.csv` | Test set: `ID`, the same 21 features. Target must be predicted. |
| `true_values.csv` | Ground truth for test set, used internally by Kaggle to score the leaderboard. Has an extra `Usage` column marking each row as `Public` or `Private`. **We do not have/use this file for training — it's for reference only and is not present on our side in real competition conditions.** |
| `random_submission.csv` | Example of the exact submission format required. |

Total data size is small (~520 KB), so this is not a big-data problem — no need for
distributed processing, chunked loading, or GPU acceleration for data handling.

## 3. Submission Format

A CSV with exactly two columns, header included:

```csv
ID,target
1,class1
2,class3
3,class2
```

One row per test `ID`, predicting the most likely class for the target.

## 4. Public vs Private Leaderboard (how scoring works)

The test set is split into two hidden groups via the `Usage` column in
`true_values.csv` (`Public` or `Private`). We never see this column ourselves.

- During the competition: every submission is scored only against the `Public` rows,
  and that score is what shows on the live leaderboard.
- After the competition ends: the final ranking is computed using the `Private` rows
  instead.
- We always predict for **all** test rows in every submission — we don't know which
  row belongs to which group. This setup discourages overfitting to the public
  leaderboard score.

Implication for modeling: don't chase the public leaderboard score by overfitting
hyperparameters to it. Trust a solid local cross-validation score more than small
public leaderboard score differences.

## 5. Evaluation Metric

Standard multi-class classification metric (accuracy / log loss / macro-F1 —
confirm exact metric from the competition rules page if available; default to
optimizing macro-F1 or log loss alongside accuracy during local validation, since
class balance is not guaranteed to be even).

## 6. Environment

Developing **locally**, not on Colab. Reasoning:
- Data is ~520 KB — trivially small.
- Expected modeling approach (LightGBM / XGBoost / CatBoost, possibly scikit-learn
  baselines) runs on CPU, so the local GPU (GTX 1650Ti) is not needed.
- Local development makes MLflow's UI (`mlflow ui`, localhost dashboard) and DVC
  (expects a normal local git repo) much simpler to use than on Colab.

## 7. MLOps Requirement 1 — MLflow (Required)

Must integrate MLflow into the training code to track experiments. Concretely:

- Run **at least 3 different experiments** (e.g. different models, hyperparameters,
  or preprocessing choices).
- For every run, log:
  - **Parameters** (inputs): model type, hyperparameters (learning rate, depth,
    number of estimators, encoding strategy, etc.)
  - **Metrics** (outputs): evaluation scores such as accuracy, log loss, F1 — ideally
    per-fold and averaged across the cross-validation folds.
- Save the **best model and its preprocessing steps together** as a single MLflow
  artifact (e.g. an sklearn `Pipeline` containing both preprocessing and the model,
  logged via `mlflow.sklearn.log_model` or equivalent).
- Deliverable: a screenshot of the MLflow UI comparison table showing all runs
  side by side. This means the experiments need clear, distinguishable run names
  and consistent logged parameters/metrics so the comparison table is meaningful.

MLflow stores everything locally by default in an `mlruns/` folder created inside
the project directory — no external server needed. View results by running
`mlflow ui` in the terminal and opening the local dashboard in a browser.

## 8. MLOps Requirement 2 — DVC (Bonus / Optional, +10%)

Must track the dataset with DVC (Data Version Control) instead of committing raw
CSVs to git.

- Initialize DVC in the project folder.
- Add the dataset files with `dvc add` — this moves the real data into a local DVC
  cache and creates small `.dvc` pointer files (containing a hash of the data).
- Commit only the `.dvc` pointer files to git. The raw CSVs must **not** be pushed
  to GitHub.
- Set up a DVC remote (e.g. a Google Drive folder) so the data can actually be
  pulled by someone else.
- Deliverable: the GitHub repo must contain the `.dvc` file, and the `README.md`
  must include a short (2-sentence) instruction on how a teammate would fetch the
  data, i.e. using `dvc pull`.

DVC should be set up **before** any real EDA or modeling work starts, so the
project's data versioning history is clean from the very first commit.

## 9. Full Project Plan (Phases)

1. **Project setup** — folder structure, virtual environment, dependencies
   (pandas, numpy, scikit-learn, lightgbm, mlflow, dvc, matplotlib, seaborn),
   git init.
2. **DVC setup** — `dvc init`, `dvc add` on the data files, commit `.dvc` files,
   configure a remote, before any data work begins.
3. **EDA** — shapes, dtypes, missing values, target class balance, infer which of
   the 21 columns are categorical vs numeric (e.g. via cardinality), check for
   train/test distribution shift.
4. **Preprocessing + validation strategy** — build a clean preprocessing pipeline
   (missing value handling, categorical encoding) and a stratified k-fold CV setup
   that will be reused consistently across all experiments.
5. **MLflow integration** — wire logging of parameters, metrics, and model +
   preprocessing artifacts into the training script.
6. **Run ≥3 experiments** — e.g. logistic regression baseline → LightGBM default
   params → LightGBM tuned params (or similar progression), each logged as a
   separate MLflow run.
7. **Model selection + submission** — compare runs in the MLflow UI, pick the
   best one, retrain if needed, predict on `test_data.csv`, write the submission
   CSV in the exact `ID,target` format matching `random_submission.csv`.
8. **Reporting** — MLflow UI comparison table screenshot, README with DVC pull
   instructions, summary of findings/approach.

## 10. Suggested Folder Structure

```
project/
├── data/              # CSV files, tracked by DVC (not committed to git directly)
├── notebooks/         # EDA notebooks
├── src/                # preprocessing, training, prediction scripts
├── models/             # local model artifacts (if not fully handled by mlflow)
├── reports/            # screenshots, writeups
├── mlruns/             # MLflow's local tracking store (auto-created)
├── requirements.txt
├── README.md
└── .dvc/                # DVC internal config/cache
```

## 11. Working Conventions

- Don't push raw data CSVs to git — only `.dvc` pointer files.
- Every training experiment must go through MLflow logging, no exceptions —
  this is graded.
- Use a consistent stratified k-fold CV split across all experiments so scores
  are comparable apples-to-apples.
- Prefer local CV score over small public leaderboard differences when deciding
  which model is "best."
