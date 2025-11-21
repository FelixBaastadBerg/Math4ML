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

# Models

## MLP
Run the best values from Hyperparameter tuning for all $n$ at the same time. 
```
python MLP_ECE.py
```
Run with the creation of the generalization bound. 
```
python Model_Test_Genbound.py
```
Run with the convergence analysis.
```
python convergence_analysis.py
```
Finding hyperparameters:
```
python run_mlp_optimisation.py --search grid
```
Run the optimized model for the hidden submission set to predict labels
```
python train_mlp_hidden_submission.py 
```


## MLP ensemble
For actually training the model with optimal hyperparameters. 
```
python train_mlp_ensemble.py
```
Hyperparameter tuning.
```
python run_mlp_ensemble_optimisation.py --search grid
```

## CNN
The cnn code is ran from the file "cnn_genbound.py". The current code is optimized for n = 10, which will generate a file where the gen bound decreases to 0.4 with 0.9635 in accuracy. Run "python cnn_genbound.py" to see the accuracy and the genBound update. It is currently using batch norm, learning rate scheduler and very high L2 loss. 
```
python cnn_genbound.py
```

## SVM
The svm code is in the "svm.py". The code runs for all $n$ choosing the best hyperparameter for the run.
```
python svm.py --n 10
```


# Exploring models and dataset
For exploring models using 
```
python run_baselines_selectables.py --models poly, logreg, mlp --n-list 10 
```
Splitting the dataset 80/20 
```
python split_kryptonite_test.py 
```



## Repository structure

```text
├── Convergence_Analysis/               # Outputs for training curve analysis
├── Datasets/                           # Kryptonite-n data (Train_Data / Test_Data)
    ├── Splits/                         # Indices of splits
    ├── Test_Data/                      # Testing data
    ├── Train_Data/                     # Training data
├── Initial_Exploration/                # Notebooks from initial exploration of dataset along with figures
├── Kryptonite_Label_Submission/        # Submission labels for hidden Kryptonite-n datasets
├── Misc Artifacts/                     # Additional artifacts
├── MLP_ECE/                            # Saved single MLP hyperparameter search results
    ├── MLP_optimization                # code structure depends on MLP_optimization/
        ├── random                      # code structure depends on random/
├── MLP_ensemble_optimization/          # Ensemble hyperparameter search results
    ├── random                          # code structure depends on random/
├── trained_mlp_ensembles/              # Saved ensemble members, manifests, evaluation metrics, and results
├── cnn_genbound.py                     # Generalisation bounds for CNN models 
├── draw_chart_for_mlp.py               # Chart showing singular MLP accuracies with Hoeffding bounds 
├── Model_Test_Genbound.py              # Generalisation bounds for selected models
├── pytorch_MLP_convergence.py          # Scripts to generate training/validation curves
├── pytorch_MLP_evaluation.py           # Evaluate *single* tuned MLP + ECE + generate hidden labels for submission
├── requirements.txt                    # Python package dependencies
├── run_baselines_selectable.py         # Train/evaluate CNN & SVM baselines
├── run_mlp_ensemble_optimisation.py    # Hyperparameter tuning for ensemble MLPs
├── run_mlp_optimisation.py             # Hyperparameter tuning for single MLPs
├── split_kryptonite_test.py            # (Optional) helper for dataset splitting
├── svm.py                              # Try SVM on dataset
├── train_mlp_ensemble.py               # Train & evaluate MLP ensembles + uncertainty

```

