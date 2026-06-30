# Anonymized Classification Challenge — ML HW 2

A Kaggle-style multi-class classification competition on a fully anonymized dataset
with 21 mixed numeric/categorical features. The goal is to predict a categorical target
class for each row in the test set.

---

## Project Structure

```
ML_HW_2/
├── data/               # Dataset files (tracked by DVC, not committed to git)
├── notebooks/          # EDA and exploration notebooks
├── src/                # Preprocessing, training, and prediction scripts
├── models/             # Local model artifacts
├── reports/            # Screenshots, writeups, and figures
├── requirements.txt    # Python dependencies
└── .dvc/               # DVC configuration
```

---

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd ML_HW_2
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # macOS / Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Fetch the data (DVC)

The raw CSV files are not stored in git. They are tracked with DVC and stored on
Google Drive. To download them, run:

```bash
dvc pull
```

> **Note:** You will need Google OAuth credentials (Client ID and Secret) and access
> to the shared Drive folder. Contact the repository owner to get both.

---

## Data Files

| File | Description |
|---|---|
| `train_data.csv` | Training set: `ID`, 21 anonymized features, `target` label |
| `test_data.csv` | Test set: `ID` and 21 features. Target must be predicted. |
| `sample_submission.csv` | Example of the required submission format |

---

## Experiment Tracking (MLflow)

All training experiments are tracked with MLflow. To view the experiment dashboard:

```bash
mlflow ui
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

---

## Submission Format

```
ID,target
1,class1
2,class3
3,class2
```

One row per test `ID`, predicting the most likely class.
