# DNN Assignment 1

This repository contains a beginner-friendly, from-scratch implementation workflow for Assignment 1 (baseline linear model vs MLP).

## Current Progress
- Milestone 1 complete: dataset selection + preprocessing pipeline for the Adult Income dataset (OpenML).
- Milestone 2 complete: baseline logistic regression from scratch with NumPy.
- Milestone 3 complete: MLP from scratch with forward/backward propagation.
- Milestone 4 complete: final result packaging + comparison plots.

## Milestone 1 (Assignment 4.1 + 4.2)
- **Dataset**: Adult Income (UCI ML Repository), 32,561 samples, 14 raw features.
- **Data Source**: Direct fetch from `https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data`
- **Problem type**: binary classification (`<=50K` vs `>50K`).
- **Primary metric**: F1-score, because class distribution is imbalanced (~24% positive class).
- **Split strategy**: stratified 80/20 train-test split (`random_state=42`) for reproducibility.
- **Preprocessing**:
  - Numeric features: `StandardScaler`
  - Categorical features: `OneHotEncoder(handle_unknown="ignore")`
  - Unified transformation using `ColumnTransformer`

## Project Layout
- `sample.ipynb`: notebook entry point for assignment submission.
- `pyproject.toml`: package metadata + CLI entrypoint (`dnn`).
- `src/data_pipeline.py`: reusable data loading and preprocessing logic.
- `src/cli.py`: centralized command runner for milestone steps.
- `run_step1.py`: tiny runner to verify step-1 data pipeline.
- `run_step2.py`: runner for step 2 to train and evaluate the baseline model.
- `run_step3.py`: runner for step 3 to train and evaluate the MLP model.
- `run_step4.py`: runner for final result packaging and comparison plots.
- `src/mlp.py`: from-scratch MLP implementation (ReLU hidden layers + sigmoid output).
- `src/assignment_results.py`: `get_assignment_results()` function expected by the assignment.
- `tests/test_smoke.py`: fast sanity check for imports + data contract.
- `data/`: local storage for datasets/artifacts.
- `models/`: saved checkpoints/weights.

## Setup
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
```

`pip install -e .` makes `src` importable from Jupyter and scripts without manual path hacks.

## Run with Unified CLI
```powershell
python -m src.cli step1
python -m src.cli step2
python -m src.cli step3
python -m src.cli step4
```

After editable install, you can also use:

```powershell
dnn step1
dnn step2
dnn step3
dnn step4
```

## Legacy Step Runners
```powershell
python run_step1.py
python run_step2.py
python run_step3.py
python run_step4.py
```

### Customizing Data Source (Advanced)

To load the Adult dataset from a custom URL:

```python
from src.data_pipeline import load_adult_income_data

# Default: UCI ML Repository
data = load_adult_income_data()

# Custom URL
custom_url = 'https://your-server.com/adult.data'
data = load_adult_income_data(url=custom_url)
```

**Supported URLs:**
- UCI ML Repository (default): `https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data`
- Any mirror or hosted copy of the adult.data file with the same format

**Data format requirements:**
- CSV format with spaces after commas
- Missing values marked as `?`
- Column order: age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country, income

If you still see "No module named 'src'" in Jupyter after install, verify the notebook kernel uses your project venv. As a fallback, run this setup cell first:

```python
import sys
from pathlib import Path

project_root = Path.cwd()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print(f'Project root: {project_root}')
print('Python path updated')
```

After this cell runs successfully, subsequent cells with `from src import ...` will work.

Expected output includes train/test shapes and class ratio, confirming Milestone 1 requirements are met.

## Run Step 2 (Baseline Model)
```powershell
python -m src.cli step2
```

This runs logistic regression from scratch, prints training/test classification metrics, and shows loss decrease from first to final epoch.

## Run Step 3 (MLP Model)
```powershell
python -m src.cli step3
```

This trains an MLP from scratch, prints training/test classification metrics, and confirms decreasing training loss.

## Run Step 4 (Final Results + Plots)
```powershell
python -m src.cli step4
```

This runs `get_assignment_results()`, prints assignment-ready metrics for both models, and saves:
- `data/loss_curves.png`
- `data/performance_comparison.png`
- `data/confusion_matrices.png`
- `data/assignment_results.json`

## Loading Results from URL

Instead of using local files, you can load assignment results from a remote URL:

```python
from src.assignment_results import load_assignment_results_from_url

# Load from a hosted JSON file (e.g., GitHub, Google Drive, or any web server)
url = 'https://raw.githubusercontent.com/your-username/your-repo/main/data/assignment_results.json'
results = load_assignment_results_from_url(url)

# Access results the same way as local loading
print(f"Baseline test F1: {results['baseline_model']['test_metrics']['f1']:.4f}")
print(f"MLP test F1: {results['mlp_model']['test_metrics']['f1']:.4f}")
```

### Hosting Options
- **GitHub**: Use `https://raw.githubusercontent.com/username/repo/branch/path/to/file.json`
- **Google Drive**: Share publicly and use the direct download link
- **Any HTTP Server**: Point to your hosted JSON file URL

The function automatically handles JSON parsing and error handling.

### See Demo
```powershell
python run_url_loading_demo.py
python -m src.cli url --results-url https://raw.githubusercontent.com/your-username/your-repo/main/data/assignment_results.json
```

## Quick Validation
Run the smoke test to confirm package imports and data preprocessing contract:

```powershell
python -m unittest tests.test_smoke
```

This shows examples of both local and remote loading patterns.
On this Adult Income split, the baseline logistic regression performs slightly better than the current MLP in generalization and efficiency. Baseline test F1 is about 0.638, while MLP test F1 is about 0.611. The baseline also trains much faster (around a few seconds) compared with the deeper MLP (around ~100 seconds). This suggests that for this feature-engineered tabular setup (standardized numeric + one-hot categorical), a linear decision boundary already captures much of the signal. The MLP reduces training loss steadily and improves over early weak settings, but still underperforms baseline on test metrics, likely due to optimization sensitivity and hyperparameter choices (learning rate, epochs, hidden sizes, threshold). A practical takeaway is that higher-capacity models are not automatically better: they require careful tuning and can cost significantly more compute. In this run, the baseline is the stronger choice for a speed/accuracy trade-off, while the MLP remains valuable for demonstrating full forward/backprop implementation and nonlinear modeling behavior.

## Milestone 1 Acceptance Check
- Dataset has >= 500 samples and >= 10 features.
- Train/test split and preprocessing run without errors.
- Encoded matrix is numeric and model-ready.
- Class distribution is reported for train and test splits.

