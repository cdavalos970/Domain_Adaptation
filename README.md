COMP90051 Statistical Machine Learning
Assignment 2 - Transfer Learning

The Project is composed by three files:

1. Baseline
- The same approach as in "Frustratingly Easy Domain Adaptation" where a set of baselines are
implemented to evaluate the performance of the algorithms in three domains
- ALL.py, LININT.py, PRED.py, SRCONLY.py, TGTONLY.py and WEIGHTED.py
- A randomized Cross - Validation is run over certain hyperparameters to select the ones with the best performance
in MSE terms

2. Feda
- The augmented approach created in "Frustratingly Easy Domain Adaptation" is used as alternative approach to
the Transfer Learning topic
- main.py is in charge of running all the implemented algorithms with parameters PER_DEV (Percentage of Development test),
PER_TEST (Percentage of Test set), SAMPLES (Percentage of samples of the target domain),
SAMPLES_DEV (Number of sample of the development set for each fold), FOLDS (Number of folds for CV with size SAMPLES_DEV),
and ITER (Number of iterations for finding hyperparameters)
- Three different Algorithms are used (lasso, NN and SVM) and three different codes are run to split charges
and saving computational cost (main_cv_lasso.py, main_cv_nn.py and main_cv_svm.py)

3. TrAdaBoosting_Regressor
- Based on TrAdaBoost method from "Boosting for Regression Transfer" and the implementation code found in https://github.com/jay15summer/Two-stage-TrAdaboost.R2,
was implemented for two different algorithms.
- main_weight.py is the main code for the algorithm where it calls the method in TwoStageTrAdaBoost.py
- Has the following parameters for the code PER_DEV (Percentage of the development set), PER_TEST (Percentage of the test set),
FOLDS (Number of folds for CV with size SAMPLES_DEV),ITER (Number of iterations for finding hyperparameters),
STEPS (Number of folds for SV on TwoStageTrAdaBoost.py), LEARNING_RATE (Learning rate for SV on TwoStageTrAdaBoost.py),
SAMPLES (Percentage of samples of the target domain) and SAMPLES_DEV (Number of sample of the development set for each fold)

In folder Two-stage-TrAdaboost.R2-master, the implementation of https://github.com/jay15summer/Two-stage-TrAdaboost.R2 can be found


