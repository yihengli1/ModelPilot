## ModelPilot

ModelPilot is an LLM-guided AutoML system designed to inspect user datasets, generate preprocessing strategies, and assemble training pipelines. It automates the decision-making process for model selection, hyperparameter tuning, and data splitting, providing a transparent and structured workflow for machine learning tasks.

### Quick start

Create .env Files

- `touch backend/.env frontend/.env`
- `echo "OPENAI_API_KEY=your_key_here" > backend/.env & echo "OPENAI_MODEL=gpt-4o" >> backend/.env`
- `echo "VITE_API_URL=http://127.0.0.1:8000" > frontend/.env`

Backend (Django + DRF):

- `cd backend`
- `source .venv/bin/activate`
- `pip install -r backend/requirements.txt`
- `python manage.py migrate`
- `python manage.py runserver`

Frontend (Vite + React):

- `cd frontend`
- `npm install`
- `npm run dev`

### Features

- LLM-Guided Planning: Utilizes OpenAI's GPT models to inspect dataset metadata (feature names, types, statistics) and user prompts to formulate a complete ML workflow.

- Automated Data Profiling: Performs local feature-level summarization and smart feature selection to optimize token usage before API calls.

- Iterative Refinement: Implements a feedback loop where initial training results are analyzed by the LLM to suggest hyperparameter tuning strategies (Grid Search).

- Target Detection: Automatically infers target columns from user prompts or defaults to unsupervised learning if no target is specified.

### Supported Models

The system currently supports the following algorithms via scikit-learn:

- Decision Trees: Classification trees with tunable depth and splitting criteria.
- Naive Bayes: Gaussian Naive Bayes for probabilistic classification.
- k-Nearest Neighbors (kNN): Distance-based classification with configurable metrics and weights.
- K-Means Clustering: Unsupervised clustering with silhouette score evaluation.

### Roadmap

# Data Preprocessing

- Encoding: One-hot encoding for categorical variables.
- Transformation: Standardization and Discretization.
- Cleaning: Automated removal of duplicate or irrelevant examples.

# Expanded Model Library

- Clustering: DBSCAN, Hierarchical Clustering, and Ensemble Clustering.
- Regression: Linear Regression (L1/L2 Regularization), Non-linear Kernel Regression.
- Classifiers: Linear Classifiers (SVM, Logistic Regression, Hinge Loss) and Multi-class SVMs/Softmax.
- Ensemble Methods: Random Forests and Gradient Boosted Trees (XGBoost).
- Dimensionality Reduction: PCA, MDS (ISOMAP, Sammon's Map).
- Deep Learning: Neural Networks, CNNs, and Transformers.

# Optimization & Visualization

- Solvers: Gradient Descent, Stochastic Gradient Descent (SGD), and SVD.
- Feature Selection: Forward Selection and Score-based search methods.
- Educational Visualization: Automated plotting of objective functions and gradients for 2D datasets using automatic differentiation.
