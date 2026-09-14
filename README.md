## ModelPilot

ModelPilot is an LLM-guided AutoML system designed to inspect user datasets, generate preprocessing strategies, and assemble training pipelines. It automates the decision-making process for model selection, hyperparameter tuning, and data splitting, providing a transparent and structured workflow for machine learning tasks.

### DEMO

This demo uses a **multi-class classification** dataset. The first view shows the raw dataset distribution **before** training, and the second view shows the dataset **after** the model assigns a predicted class to each point.

https://github.com/user-attachments/assets/2e12b3c4-c7cc-4063-8aeb-cd649347d5a1

<img width="1279" height="478" alt="Screenshot 2026-01-01 at 1 51 44 PM" src="https://github.com/user-attachments/assets/f3901051-0bce-4772-8aff-4281c3831605" />


### Quick start

Create .env Files
```
touch backend/.env frontend/.env
```
- Follow example.env in frontend and backend folder to setup environment variables

Backend (Django + DRF):

```
cd backend

source .venv/bin/activate

pip install -r backend/requirements.txt

python manage.py migrate

python manage.py runserver
```

Frontend (Vite + React):

```
cd frontend

npm install

npm run dev
```

### Features

- LLM-Guided Planning: Utilizes OpenAI's GPT models to inspect dataset metadata (feature names, types, statistics) and user prompts to formulate a complete ML workflow.

- Automated Data Profiling: Performs local feature-level summarization and smart feature selection to optimize token usage before API calls.

- Iterative Refinement: Implements a feedback loop where initial training results are analyzed by the LLM to suggest hyperparameter tuning strategies (Grid Search).

- Target Detection: Automatically infers target columns from user prompts or defaults to unsupervised learning if no target is specified.

### Supported Models

The system currently supports the following algorithms via scikit-learn + PyTorch:

- **Decision Trees**: Supervised classification using tree-based splits; supports tunable depth and split criteria.
- **Naive Bayes**: Gaussian Naive Bayes probabilistic classifier for continuous features.
- **k-Nearest Neighbors (kNN)**: Distance-based classification with configurable *k*, distance metric, and weighting.
- **SVM**: Kernel-based classifier (linear/rbf/poly/sigmoid) with tunable C and gamma.
- **Linear Classifiers (Logistic / SVM / Hinge)**: Linear decision-boundary classifiers; supports binary and multi-class variants (e.g., softmax / multi-class SVM).
- **Linear Regression (L1/L2 Regularization)**: Supervised regression with optional L1 (Lasso) or L2 (Ridge) penalties.
- **Kernel Regression (Non-linear)**: Non-linear regression using kernelized/basis-style methods for modeling curved relationships.
- **K-Means Clustering**: Unsupervised clustering with tunable *k*; includes silhouette score evaluation.
- **DBSCAN**: Density-based clustering; includes silhouette score evaluation where applicable.
- **Hierarchical Clustering**: Agglomerative clustering with linkage options; includes silhouette score evaluation where applicable.
- **PCA (Principal Component Analysis)**: Unsupervised dimensionality reduction for compressing features while preserving variance; reports explained variance and component count.
- **MLP (Neural Network)**: Feedforward multilayer perceptron for tabular data (classification & regression); supports configurable hidden layers, activation, dropout, optimizer, and early stopping.

### Supported Optimizers

- **Gradient Descent**: Set Batch size to n
- **Mini-Batch Gradient Descent**: Set Batch size from 32-128
- **Stochastic Gradient Descent**: Set Batch size to 1
- **Adam (Adaptive Moment Estimation)**: Efficiently adjusts step sizes for each parameter based on past gradients' first (momentum) and second (variance) moments

### Model Ranking

Cheap, single-shot models (`decision_tree`, `naive_bayes`, `knn`, `svm`) are ranked using 3-fold cross-validation over the pooled train/val split rather than one noisy holdout, matching the CV fold count used by the scikit-learn baselines they're benchmarked against. Gradient-trained models (`mlp_classifier`/`mlp_regressor`, `linear_classifier`, `linear_regression`, `kernel_polynomial`) keep a single train/val split, since refitting them per CV fold is too slow to be worth the extra robustness. The held-out test set is never touched by cross-validation, so reported test scores stay leakage-free.

## Benchmarking Against scikit-learn Baselines

`backend/benchmark.py` and `backend/benchmark_modelpilot_llm.py` compare ModelPilot's automated pipeline against untuned and `GridSearchCV`-tuned scikit-learn baselines on the Kaggle datasets in `backend/data/`.

```
cd backend
./.venv/bin/python benchmark_modelpilot_llm.py
```

A round of debugging against this benchmark turned up several real bugs that were quietly capping ModelPilot's accuracy well below what its model zoo is actually capable of:

- The production data split had no feature standardization at all, hurting every distance/gradient-based model (kNN, MLP, linear models) while the sklearn baselines it's compared against got `StandardScaler` for free.
- `decision_tree` silently failed on **every** evaluation due to a fit-signature detection bug that misfired on sklearn's `fit(X, y, sample_weight, check_input)` signature.
- The refinement step always saw `null` for every prior model's validation score (wrong dict key), so the LLM had no signal for which configs had actually worked.
- `mlp_classifier`/`mlp_regressor` crashed whenever the LLM supplied an `activation` value, because the prompt documented it as a float range instead of the actual `"relu"/"leaky_relu"/"tanh"` enum.
- `linear_classifier` crashed on a missing required `loss` argument that the prompt never told the LLM to supply.
- `svm` wasn't a supported model at all, and a contradictory prompt instruction was silently restricting the planner to only ever recommend `decision_tree`/`naive_bayes`/`knn` regardless of the dataset.

Latest benchmark run (3 classification datasets, `gpt-5-nano` planner, ~1 minute per dataset):

| Dataset | Best Untuned | Best Tuned (`GridSearchCV`) | ModelPilot |
|---|---|---|---|
| Titanic | 0.832 | 0.816 | 0.815 (`mlp_classifier`) |
| Heart Failure | 0.891 | 0.891 | 0.849 (`svm`) |
| Wine Quality | 0.673 | 0.681 | 0.570 (`svm`) |

ModelPilot lands within a few points of a manually tuned model on 2 of 3 datasets with no hand-written code or hyperparameter grid. The remaining gap on Wine Quality looks like a ceiling of the small planner model on an imbalanced, multi-class target rather than a pipeline bug — a stronger planning model would be the next lever to pull.

## Roadmap

### Data Preprocessing

- Encoding: One-hot encoding for categorical variables.
- Transformation: Standardization and Discretization.
- Cleaning: Automated removal of duplicate or irrelevant examples.

### Expanded Model Library
- Done :D

### Optimization & Visualization

- Feature Selection: Forward Selection and Score-based search methods.
- Educational Visualization: Automated plotting of objective functions and gradients for 2D datasets using automatic differentiation.
- Leaderboards for who can prompt the best for a given dataset every week/month

## Testing / Example Data

To quickly generate sample datasets (and seed example data for development/testing), run the dataset seeding script from the backend.

```
cd backend

source .venv/bin/activate

python manage.py migrate

python seed_datasets.py
```
What this does:

Generates example datasets you can use to test the pipeline end-to-end.

Helps create consistent, reproducible example data for development (exact outputs depend on the script implementation—check seed_datasets.py for paths and options).

