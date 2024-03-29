# Inspired by GPT4

# Information on type hints
# https://peps.python.org/pep-0585/

# GPT on testing functions, mock functions, testing number of calls, and argument values
# https://chat.openai.com/share/b3fd7739-b691-48f2-bb5e-0d170be4428c


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    ShuffleSplit,
    cross_validate,
    KFold,
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

from typing import Any
from numpy.typing import NDArray

import numpy as np
import utils as u

# Initially empty. Use for reusable functions across
# Sections 1-3 of the homework
import new_utils as nu


# ======================================================================
class Section1:
    def __init__(
        self,
        normalize: bool = True,
        seed: int | None = None,
        frac_train: float = 0.2,
    ):
        """
        Initializes an instance of MyClass.

        Args:
            normalize (bool, optional): Indicates whether to normalize the data. Defaults to True.
            seed (int, optional): The seed value for randomization. If None, each call will be randomized.
                If an integer is provided, calls will be repeatable.

        Returns:
            None

        Notes: notice the argument `seed`. Make sure that any sklearn function that accepts
        `random_state` as an argument is initialized with this seed to allow reproducibility.
        You change the seed ONLY in the section of run_part_1.py, run_part2.py, run_part3.py
        below `if __name__ == "__main__"`
        """
        self.normalize = normalize
        self.frac_train = frac_train
        self.seed = seed

    # ----------------------------------------------------------------------
    """
    A. We will start by ensuring that your python environment is configured correctly and 
       that you have all the required packages installed. For information about setting up 
       Python please consult the following link: https://www.anaconda.com/products/individual. 
       To test that your environment is set up correctly, simply execute `starter_code` in 
       the `utils` module. This is done for you. 
    """

    def partA(self):
        # Return 0 (ran ok) or -1 (did not run ok)
        print("Part A: Starter code\n")
        answer = u.starter_code()
        return answer

    # ----------------------------------------------------------------------
    """
    B. Load and prepare the mnist dataset, i.e., call the prepare_data and filter_out_7_9s 
       functions in utils.py, to obtain a data matrix X consisting of only the digits 7 and 9. Make sure that 
       every element in the data matrix is a floating point number and scaled between 0 and 1 (write
       a function `def scale() in new_utils.py` that returns a bool to achieve this. Checking is not sufficient.) 
       Also check that the labels are integers. Print out the length of the filtered 𝑋 and 𝑦, 
       and the maximum value of 𝑋 for both training and test sets. Use the routines provided in utils.
       When testing your code, I will be using matrices different than the ones you are using to make sure 
       the instructions are followed. 
    """

    def partB(
        self,
    ):
        print("Part B: Load and prepare the mnist dataset\n")
        X, y, Xtest, ytest = u.prepare_data()
        Xtrain, ytrain = u.filter_out_7_9s(X, y)
        Xtest, ytest = u.filter_out_7_9s(Xtest, ytest)

        print(Xtrain.shape, ytrain.shape, Xtest.shape, ytest.shape)
        Xtrain = nu.scale_data(Xtrain)
        Xtest = nu.scale_data(Xtest)
        Xtrain = Xtrain.astype(float)
        Xtest = Xtest.astype(float)
        ytrain = ytrain.astype(int)
        ytest = ytest.astype(int)

        answer = {}

        # Enter your code and fill the `answer` dictionary

        answer["length_Xtrain"] = len(Xtrain)  # Number of samples
        answer["length_Xtest"] = len(Xtest)
        answer["length_ytrain"] = len(ytrain)  # Number of samples
        answer["length_ytest"] = len(ytest)
        answer["max_Xtrain"] = np.max(Xtrain)  # Maximum value of Xtrain
        answer["max_Xtest"] = np.max(Xtest)  # Maximum value of Xtest
        print(f"{answer=}")
        return answer, Xtrain, ytrain, Xtest, ytest

    """
    C. Train your first classifier using k-fold cross validation (see train_simple_classifier_with_cv 
       function). Use 5 splits and a Decision tree classifier. Print the mean and standard deviation 
       for the accuracy scores in each validation set in cross validation. (with k splits, cross_validate
       generates k accuracy scores.)  
       Remember to set the random_state in the classifier and cross-validator.
    """

    # ----------------------------------------------------------------------
    def partC(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
    ):
        print("Part C: Train your first classifier using k-fold cross validation\n")

        # Enter your code and fill the `answer` dictionary
        answer = {}

        answer["clf"] = DecisionTreeClassifier(
            random_state=self.seed
        )  # the estimator (classifier instance)

        answer["cv"] = KFold(
            n_splits=5, random_state=self.seed, shuffle=True
        )  # the cross validator instance
        # the dictionary with the scores  (a dictionary with
        # keys: 'mean_fit_time', 'std_fit_time', 'mean_accuracy', 'std_accuracy'.
        scores = u.train_simple_classifier_with_cv(
            Xtrain=X, ytrain=y, clf=answer["clf"], cv=answer["cv"]
        )
        answer["scores"] = {
            "mean_fit_time": scores["test_score"].mean(),
            "std_fit_time": scores["test_score"].std(),
            "mean_accuracy": scores["test_score"].mean(),
            "std_accuracy": scores["test_score"].std(),
        }
        return answer

    # ---------------------------------------------------------
    """
    D. Repeat Part C with a random permutation (Shuffle-Split) 𝑘-fold cross-validator.
    Explain the pros and cons of using Shuffle-Split versus 𝑘-fold cross-validation.
    """

    def partD(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
    ):
        print(
            "Part D: Repeat Part C with a random permutation (Shuffle-Split) 𝑘-fold cross-validator\n"
        )
        # Enter your code and fill the `answer` dictionary

        # Answer: same structure as partC, except for the key 'explain_kfold_vs_shuffle_split'

        answer = {}
        answer["clf"] = DecisionTreeClassifier(random_state=self.seed)
        answer["cv"] = ShuffleSplit(n_splits=5, random_state=self.seed)

        scores = u.train_simple_classifier_with_cv(
            Xtrain=X, ytrain=y, clf=answer["clf"], cv=answer["cv"]
        )
        answer["scores"] = {
            "mean_fit_time": scores["test_score"].mean(),
            "std_fit_time": scores["test_score"].std(),
            "mean_accuracy": scores["test_score"].mean(),
            "std_accuracy": scores["test_score"].std(),
        }
        answer[
            "explain_kfold_vs_shuffle_split"
        ] = "k-fold segments the whole dataset, using each segment as the \
            test set exactly once. It doesn't scale well with large datasets however, which is where shuffle-split \
            comes in. It randomly subsets the data by set amounts, and is more efficient for large datasets. \
            of course, the cost is that it is less comprehensive than k-fold."
        return answer

    # ----------------------------------------------------------------------
    """
    E. Repeat part D for 𝑘=2,5,8,16, but do not print the training time. 
       Note that this may take a long time (2–5 mins) to run. Do you notice 
       anything about the mean and/or standard deviation of the scores for each k?
    """

    def partE(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
    ):
        print("Part E: Repeat part D for 𝑘=2,5,8,16\n")
        # Answer: built on the structure of partC
        # `answer` is a dictionary with keys set to each split, in this case: 2, 5, 8, 16
        # Therefore, `answer[k]` is a dictionary with keys: 'scores', 'cv', 'clf`

        answer = {}
        answer[2] = {}
        answer[5] = {}
        answer[8] = {}
        answer[16] = {}

        for k in [2, 5, 8, 16]:
            answer[k]["clf"] = DecisionTreeClassifier(random_state=self.seed)
            answer[k]["cv"] = ShuffleSplit(n_splits=k, random_state=self.seed)
            scores = u.train_simple_classifier_with_cv(
                Xtrain=X, ytrain=y, clf=answer[k]["clf"], cv=answer[k]["cv"]
            )
            answer[k]["scores"] = {
                "mean_fit_time": scores["test_score"].mean(),
                "std_fit_time": scores["test_score"].std(),
                "mean_accuracy": scores["test_score"].mean(),
                "std_accuracy": scores["test_score"].std(),
            }
            answer[k]["scores"] = u.train_simple_classifier_with_cv(
                Xtrain=X, ytrain=y, clf=answer[k]["clf"], cv=answer[k]["cv"]
            )
            print(f"Scores for k={k}")
        print(
            "I noticed that the mean and standard deviation of the scores for each k \
                  are inversely related. As k increases, the mean decreases and the standard deviation increases."
        )
        # Enter your code, construct the `answer` dictionary, and return it.
        return answer

    # ----------------------------------------------------------------------
    """
    F. Repeat part D with a Random-Forest classifier with default parameters. 
       Make sure the train test splits are the same for both models when performing 
       cross-validation. (Hint: use the same cross-validator instance for both models.)
       Which model has the highest accuracy on average? 
       Which model has the lowest variance on average? Which model is faster 
       to train? (compare results of part D and part F)

       Make sure your answers are calculated and not copy/pasted. Otherwise, the automatic grading 
       will generate the wrong answers. 
       
       Use a Random Forest classifier (an ensemble of DecisionTrees). 
    """

    def partF(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
    ) -> dict[str, Any]:
        print(
            "Part F: Repeat part D with a Random-Forest classifier with default parameters\n"
        )

        answer = {}

        # Random Forest classifier with default parameters
        clf_RF = RandomForestClassifier(random_state=self.seed)
        # Decision Tree classifier for comparison
        clf_DT = DecisionTreeClassifier(random_state=self.seed)
        # Use ShuffleSplit for cross-validation to ensure identical splits for both classifiers
        cv = ShuffleSplit(n_splits=5, random_state=self.seed)

        # Training and evaluating the Random Forest classifier
        scores_RF = u.train_simple_classifier_with_cv(
            Xtrain=X, ytrain=y, clf=clf_RF, cv=cv
        )
        mean_accuracy_RF = scores_RF["test_score"].mean()
        std_accuracy_RF = scores_RF["test_score"].std()
        mean_fit_time_RF = scores_RF["fit_time"].mean()

        # Training and evaluating the Decision Tree classifier
        scores_DT = u.train_simple_classifier_with_cv(
            Xtrain=X, ytrain=y, clf=clf_DT, cv=cv
        )
        mean_accuracy_DT = scores_DT["test_score"].mean()
        std_accuracy_DT = scores_DT["test_score"].std()
        mean_fit_time_DT = scores_DT["fit_time"].mean()

        # Comparing the performance
        highest_accuracy_model = (
            "random-forest" if mean_accuracy_RF > mean_accuracy_DT else "decision-tree"
        )
        lowest_variance_model = (
            "random-forest"
            if std_accuracy_RF**2 < std_accuracy_DT**2
            else "decision-tree"
        )
        fastest_model = (
            "random-forest" if mean_fit_time_RF < mean_fit_time_DT else "decision-tree"
        )

        # Updating the answer dictionary with the results
        answer["clf_RF"] = clf_RF
        answer["clf_DT"] = clf_DT
        answer["cv"] = cv
        answer["scores_RF"] = {
            "mean_accuracy": mean_accuracy_RF,
            "std_accuracy": std_accuracy_RF,
            "mean_fit_time": mean_fit_time_RF,
        }
        answer["scores_DT"] = {
            "mean_accuracy": mean_accuracy_DT,
            "std_accuracy": std_accuracy_DT,
            "mean_fit_time": mean_fit_time_DT,
        }
        answer["model_highest_accuracy"] = highest_accuracy_model
        answer["model_lowest_variance"] = lowest_variance_model
        answer["model_fastest"] = fastest_model

        return answer

    # ----------------------------------------------------------------------
    """
    G. For the Random Forest classifier trained in part F, manually (or systematically, 
       i.e., using grid search), modify hyperparameters, and see if you can get 
       a higher mean accuracy.  Finally train the classifier on all the training 
       data and get an accuracy score on the test set.  Print out the training 
       and testing accuracy and comment on how it relates to the mean accuracy 
       when performing cross validation. Is it higher, lower or about the same?

       Choose among the following hyperparameters: 
         1) criterion, 
         2) max_depth, 
         3) min_samples_split, 
         4) min_samples_leaf, 
         5) max_features 
    """

    def partG(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> dict[str, Any]:
        print("Part G: Modify hyperparameters and train on all training data\n")

        # Setup the Random Forest classifier with default parameters for comparison
        clf_default = RandomForestClassifier(random_state=self.seed)
        clf_default.fit(X, y)

        # Setup the parameter grid for hyperparameter tuning
        parameters = {
            "criterion": ["gini", "entropy"],
            "max_depth": [2, 10, 20],
            "n_estimators": [20, 50, 100],
        }

        # Initialize GridSearchCV with RandomForestClassifier and the defined parameters
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=self.seed),
            param_grid=parameters,
            cv=ShuffleSplit(n_splits=5, random_state=self.seed),
            scoring="accuracy",
            refit=True,
        )

        # Perform the grid search on the training data
        grid_search.fit(X, y)

        # Extract the best estimator and its performance metrics
        best_estimator = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score_cv = grid_search.best_score_

        # Train the best estimator on the full training data and evaluate on the test set
        best_estimator.fit(X, y)
        accuracy_train_best = best_estimator.score(X, y)
        accuracy_test_best = best_estimator.score(Xtest, ytest)

        # Compare with the default classifier
        accuracy_train_default = clf_default.score(X, y)
        accuracy_test_default = clf_default.score(Xtest, ytest)

        answer = {
            "clf_default": clf_default,
            "best_estimator": best_estimator,
            "grid_search": grid_search,
            "default_parameters": clf_default.get_params(),
            "best_parameters": best_params,
            "mean_accuracy_cv": best_score_cv,
            "accuracy_train_default": accuracy_train_default,
            "accuracy_train_best": accuracy_train_best,
            "accuracy_test_default": accuracy_test_default,
            "accuracy_test_best": accuracy_test_best,
        }

        # Provide insights into the performance
        print(
            f"Default training accuracy: {accuracy_train_default:.4f}, Default test accuracy: {accuracy_test_default:.4f}"
        )
        print(
            f"Optimized training accuracy: {accuracy_train_best:.4f}, Optimized test accuracy: {accuracy_test_best:.4f}"
        )
        # The mean accuracy on the training set with grid search
        # is higher than on the default parameters, suggesting better generalization
        return answer
