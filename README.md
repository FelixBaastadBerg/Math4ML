# Math4ML - Kryptonite-2

This repository contains the code used for the **ImpCML 2025** coursework investigating the *Kryptonite-2* challenge datasets.  
The goal is to test and refute the claim (from the “Kryptonite-2.0” mock paper) that simple procedurally generated binary classification tasks cannot be solved by modern machine learning models.

We performed the following:

- train **single MLPs** with hyperparameter optimisation,
- train **MLP ensembles** for improved performance and uncertainty estimation,
- train and compare against **CNN** and **RBF-kernel SVM** baselines,
- compute **calibration metrics**, such as ECE, reliability diagrams,
- estimate **epistemic uncertainty** using ensemble disagreement and probability variance,
- and derive **generalisation bounds** (Hoeffding vs Chebyshev)

All experiments are run on the Kryptonite-n datasets for  
\(n \in \{10,12,14,16,18,20\}\).

---

## Repository structure

```text
.
├── Convergence_Analysis/        # Scripts / outputs for training curve analysis
├── Datasets/                    # Kryptonite-n data (Train_Data / Test_Data)
├── Evaluation/                  # Saved metrics, plots, and summaries
├── Initial_Exploration/         # Early EDA notebooks / figures
├── MLP_ECE/                     # Single-MLP calibration results & plots
├── MLP_ensemble_optimization/   # Ensemble hyperparameter search results
├── MLP_optimization/            # Single-MLP hyperparameter search results
├── trained_mlp_ensembles/       # Saved ensemble members and manifests
│
├── cnn_genbound.py              # Generalisation bounds for CNN models
├── convergence_analysis.py      # Scripts to generate training/validation curves
├── explore_datasets.ipynb       # Notebook for PCA, correlations, etc.
├── MLP_ECE_eval.py              # Evaluate *single* tuned MLP + ECE + reliability
├── Model_Test_Genbound.py       # Generalisation bounds for selected models
├── requirements.txt             # Python package dependencies
│
├── run_baselines_selectable.py  # Train/evaluate CNN & SVM baselines
├── run_mlp_ensemble_optimisation.py  # Hyperparameter tuning for ensemble MLPs
├── run_mlp_optimisation.py      # Hyperparameter tuning for single MLPs
├── split_kryptonite_test.py     # (Optional) helper for dataset splitting
└── train_mlp_ensemble.py        # Train & evaluate MLP ensembles + uncertainty