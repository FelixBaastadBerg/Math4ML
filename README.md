# Math4ML - Kryptonite-2

This repository contains the code used for the **Kryptonite v2.0** coursework investigating the *Kryptonite-2* challenge datasets.  
The goal is to test and refute the claim from the “Kryptonite-2.0” mock paper that simple generated binary classification tasks cannot be solved by modern machine learning models.

We performed the following:

- train **single MLPs** with hyperparameter optimisation,
- train **MLP ensembles** for improved performance and uncertainty estimation,
- train and compare against **CNN** and **RBF-kernel SVM** baselines,
- compute **calibration metrics**, such as ECE, reliability diagrams,
- estimate **epistemic uncertainty** using ensemble disagreement and probability variance,
- derive **generalisation bounds** (Hoeffding vs Chebyshev),
- train final **single MLP** to generate labels for hidden Kryptonite-n dataset for submission

All experiments are run on the Kryptonite-n datasets for  
n ∈ {10,12,14,16,18,20}.

---

## Repository structure

```text
.
├── Convergence_Analysis/               # Outputs for training curve analysis
├── Datasets/                           # Kryptonite-n data (Train_Data / Test_Data)
├── Hidden_Kryptonite_Submission/       # Submission labels for hidden Kryptonite-n datasets
├── Initial_Exploration/                # Notebooks from initial exploration of dataset along with figures
├── MLP_ECE/                            # Saved Single-MLP calibration results, plots, evaluation metrics, and hyperparameter search results
├── MLP_ensemble_optimization/          # Ensemble hyperparameter search results
├── trained_mlp_ensembles/              # Saved ensemble members, manifests, evaluation metrics, and results
│
├── cnn_genbound.py                     # Generalisation bounds for CNN models 
├── convergence_analysis.py             # Scripts to generate training/validation curves
├── explore_datasets.ipynb              # Notebook for PCA, correlations, etc.
├── MLP_ECE_eval.py                     # Evaluate *single* tuned MLP + ECE + reliability
├── Model_Test_Genbound.py              # Generalisation bounds for selected models
├── requirements.txt                    # Python package dependencies
│
├── run_baselines_selectable.py         # Train/evaluate CNN & SVM baselines
├── run_mlp_ensemble_optimisation.py    # Hyperparameter tuning for ensemble MLPs
├── run_mlp_optimisation.py             # Hyperparameter tuning for single MLPs
├── split_kryptonite_test.py            # (Optional) helper for dataset splitting
├── train_mlp_ensemble.py               # Train & evaluate MLP ensembles + uncertainty
└── train_mlp_hidden_submission.py      # Train MLP on full dataset and generate labels for hidden Kryptonite-n datasets

```

# Models
## CNN
The cnn code is ran from the file "cnn_genbound.py". The current code is optimized for n = 10, which will generate a file where the gen bound decreases to 0.4 with 0.9635 in accuracy. Run "python cnn_genbound.py" to see the accuracy and the genBound update. It is currently using batch norm, learning rate scheduler and very high L2 loss. 
