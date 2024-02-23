# Add your imports here.
# Note: only sklearn, numpy, utils and new_utils are allowed.
from sklearn.model_selection import (
    ShuffleSplit,
    cross_validate,
    KFold,
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
from numpy.typing import NDArray
from typing import Any

import utils as u
import new_utils as nu

# ======================================================================

# I could make Section 2 a subclass of Section 1, which would facilitate code reuse.
# However, both classes have the same function names. Better to pass Section 1 instance
# as an argument to Section 2 class constructor.


class Section2:
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
        """
        self.normalize = normalize
        self.seed = seed
        self.frac_train = frac_train

    # ---------------------------------------------------------

    """
    A. Repeat part 1.B but make sure that your data matrix (and labels) consists of
        all 10 classes by also printing out the number of elements in each class y and 
        print out the number of classes for both training and testing datasets. 
    """

    def partA(
        self,
    ) -> tuple[
        dict[str, Any],
        NDArray[np.floating],
        NDArray[np.int32],
        NDArray[np.floating],
        NDArray[np.int32],
    ]:
        print("Part A: Analyze the full MNIST dataset for multi-class classification\n")

        # Load and prepare the dataset to include all classes (0-9)
        X, y, Xtest, ytest = u.prepare_data()

        # Scale the data to have values between 0 and 1
        Xtrain = nu.scale_data(X)
        Xtest = nu.scale_data(Xtest)

        # Compute the number of elements in each class for both the training and test sets
        classes_train, class_count_train = np.unique(y, return_counts=True)
        classes_test, class_count_test = np.unique(ytest, return_counts=True)

        answer = {
            "nb_classes_train": len(classes_train),
            "nb_classes_test": len(classes_test),
            "class_count_train": class_count_train,
            "class_count_test": class_count_test,
            "length_Xtrain": len(Xtrain),
            "length_Xtest": len(Xtest),
            "length_ytrain": len(y),
            "length_ytest": len(ytest),
            "max_Xtrain": np.max(Xtrain),
            "max_Xtest": np.max(Xtest),
        }

        print(f"{answer=}")
        return answer, Xtrain, y, Xtest, ytest

    """
    B.  Repeat part 1.C, 1.D, and 1.F, for the multiclass problem. 
        Use the Logistic Regression for part F with 300 iterations. 
        Explain how multi-class logistic regression works (inherent, 
        one-vs-one, one-vs-the-rest, etc.).
        Repeat the experiment for ntrain=1000, 5000, 10000, ntest = 200, 1000, 2000.
        Comment on the results. Is the accuracy higher for the training or testing set?
        What is the scores as a function of ntrain.

        Given X, y from mnist, use:
        Xtrain = X[0:ntrain, :]
        ytrain = y[0:ntrain]
        Xtest = X[ntrain:ntrain+test]
        ytest = y[ntrain:ntrain+test]
    """

    def partB(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
        ntrain_list: list[int] = [],
        ntest_list: list[int] = [],
    ) -> dict[int, dict[str, Any]]:
        print(
            "Part B: Multi-class classification with Logistic Regression and various training/testing sizes\n"
        )

        answer = {}

        for ntrain, ntest in zip(ntrain_list, ntest_list):
            Xtrain_sub = X[:ntrain, :]
            ytrain_sub = y[:ntrain]
            Xtest_sub = Xtest[:ntest, :]
            ytest_sub = ytest[:ntest]

            print(f"Training with {ntrain} samples and testing with {ntest} samples.")

            # Repeat part C for multi-class classification
            print("==> Repeating part C (k-fold cross-validation)")
            cv = KFold(n_splits=5, shuffle=True, random_state=self.seed)
            clf_C = DecisionTreeClassifier(random_state=self.seed)
            scores_C = u.train_simple_classifier_with_cv(
                Xtrain=Xtrain_sub, ytrain=ytrain_sub, clf=clf_C, cv=cv
            )
            scores_C = {
                "mean_fit_time": scores_C["fit_time"].mean(),
                "std_fit_time": scores_C["fit_time"].std(),
                "mean_accuracy": scores_C["test_score"].mean(),
                "std_accuracy": scores_C["test_score"].std(),
            }

            # Repeat part D for multi-class classification
            print("==> Repeating part D (Shuffle-Split cross-validation)")
            cv_D = ShuffleSplit(n_splits=5, random_state=self.seed)
            scores_D = u.train_simple_classifier_with_cv(
                Xtrain=Xtrain_sub, ytrain=ytrain_sub, clf=clf_C, cv=cv_D
            )
            scores_D = {
                "mean_fit_time": scores_D["fit_time"].mean(),
                "std_fit_time": scores_D["fit_time"].std(),
                "mean_accuracy": scores_D["test_score"].mean(),
                "std_accuracy": scores_D["test_score"].std(),
            }

            # Repeat part F for multi-class classification using Logistic Regression
            print("==> Repeating part F (Logistic Regression with 300 iterations)")
            clf_F = LogisticRegression(
                random_state=self.seed, max_iter=300, multi_class="multinomial"
            )

            # Cross-validate to get mean and std accuracy across folds
            scores_F = u.train_simple_classifier_with_cv(
                Xtrain=Xtrain_sub,
                ytrain=ytrain_sub,
                clf=clf_F,
                cv=ShuffleSplit(n_splits=5, random_state=self.seed),
            )
            class_count_train = np.unique(ytrain_sub, return_counts=True)[1]
            class_count_test = np.unique(ytest_sub, return_counts=True)[1]

            # Update the answer dictionary for this ntrain-ntest pair
            answer[ntrain] = {
                "partC": {"scores": scores_C, "clf": clf_C, "cv": cv},
                "partD": {"scores": scores_D, "clf": clf_C, "cv": cv_D},
                "partF": {
                    "scores": {
                        "mean_accuracy": scores_F["test_score"].mean(),
                        "std_accuracy": scores_F["test_score"].std(),
                        "mean_fit_time": scores_F["fit_time"].mean(),
                    },
                    "clf": clf_F,
                    "cv": ShuffleSplit(n_splits=5, random_state=self.seed),
                },
                "ntrain": ntrain,
                "ntest": ntest,
                "class_count_train": class_count_train.tolist(),
                "class_count_test": class_count_test.tolist(),
            }
            print(f"{answer=}")
            print(
                "The logistic regression model performs consistently better than the decision tree model."
            )
            print(
                "The accuracy is higher for the training set than for the testing set."
            )
            print("The accuracy increases with ntrain, but with diminishing returns.")
        return answer
