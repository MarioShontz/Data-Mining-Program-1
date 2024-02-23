import numpy as np
from numpy.typing import NDArray
from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import top_k_accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score, recall_score, precision_score, f1_score

import utils as u
import new_utils as nu

"""
   In the first two set of tasks, we will narrowly focus on accuracy - 
   what fraction of our predictions were correct. However, there are several 
   popular evaluation metrics. You will learn how (and when) to use these evaluation metrics.
"""


# ======================================================================
class Section3:
    def __init__(
        self,
        normalize: bool = True,
        frac_train=0.2,
        seed=42,
    ):
        self.seed = seed
        self.normalize = normalize

    def analyze_class_distribution(self, y: NDArray[np.int32]) -> dict[str, Any]:
        """
        Analyzes and prints the class distribution in the dataset.

        Parameters:
        - y (array-like): Labels dataset.

        Returns:
        - dict: A dictionary containing the count of elements in each class and the total number of classes.
        """
        # Your code here to analyze class distribution
        # Hint: Consider using collections.Counter or numpy.unique for counting

        uniq, counts = np.unique(y, return_counts=True)
        print(f"{uniq=}")
        print(f"{counts=}")
        print(f"{np.sum(counts)=}")

        return {
            "class_counts": counts,  # Replace with actual class counts
            "num_classes": uniq,  # Replace with the actual number of classes
        }

    # --------------------------------------------------------------------------
    """
    A. Using the same classifier and hyperparameters as the one used at the end of part 2.B. 
       Get the accuracies of the training/test set scores using the top_k_accuracy score for k=1,2,3,4,5. 
       Make a plot of k vs. score for both training and testing data and comment on the rate of accuracy change. 
       Do you think this metric is useful for this dataset?
    """

    def partA(
        self,
        Xtrain: NDArray[np.floating],
        ytrain: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ):
        answer = {}
        clf = RandomForestClassifier(random_state=self.seed)
        clf.fit(Xtrain, ytrain)

        ytrain_pred = clf.predict_proba(Xtrain)
        ytest_pred = clf.predict_proba(Xtest)

        topk = [k for k in range(1, 6)]
        plot_scores_train = []
        plot_scores_test = []
        for k in topk:
            score_train = top_k_accuracy_score(ytrain, ytrain_pred, k=k, labels=np.arange(clf.n_classes_))
            score_test = top_k_accuracy_score(ytest, ytest_pred, k=k, labels=np.arange(clf.n_classes_))
            plot_scores_train.append((k, score_train))
            plot_scores_test.append((k, score_test))

        answer["plot_k_vs_score_train"] = plot_scores_train
        answer["plot_k_vs_score_test"] = plot_scores_test
        print(f"{plot_scores_train=}")
        print(f"{plot_scores_test=}")

        # Additional comments on the rate of accuracy change and usefulness of the metric
        answer["text_rate_accuracy_change"] = "The rate of accuracy change indicates how additional considerations of top classes (k) impact model performance. A steep initial increase may flatten, indicating early top classes' dominance."
        answer["text_is_topk_useful_and_why"] = "Top-k accuracy is particularly useful for datasets with multiple classes or when the distinction between some classes is not clear-cut. It provides insight beyond simple accuracy, especially in multi-class scenarios where the correct class might not be the model's top prediction but within its top 'k' predictions."


        return answer, Xtrain, ytrain, Xtest, ytest

    # --------------------------------------------------------------------------
    """
    B. Repeat part 1.B but return an imbalanced dataset consisting of 90% of all 9s removed.  Also convert the 7s to 0s and 9s to 1s.
    """

    def partB(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> tuple[
        dict[Any, Any],
        NDArray[np.floating],
        NDArray[np.int32],
        NDArray[np.floating],
        NDArray[np.int32],
    ]:
        # Ensure reproducibility
        np.random.seed(self.seed)

        # Filtering to keep only the digits 7 and 9
        is_seven_or_nine = (y == 7) | (y == 9)
        X = X[is_seven_or_nine]
        y = y[is_seven_or_nine]

        is_seven_or_nine_test = (ytest == 7) | (ytest == 9)
        Xtest = Xtest[is_seven_or_nine_test]
        ytest = ytest[is_seven_or_nine_test]

        # Identify and remove 90% of 9s
        indices_of_nines = np.where(y == 9)[0]
        nines_to_remove = np.random.choice(indices_of_nines, size=int(0.9 * len(indices_of_nines)), replace=False)
        
        X = np.delete(X, nines_to_remove, axis=0)
        y = np.delete(y, nines_to_remove)

        # Convert labels: 7 to 0, 9 to 1
        y = np.where(y == 7, 0, 1)
        ytest = np.where(ytest == 7, 0, 1)

        answer = {
            'X_balanced': X,
            'y_balanced': y,
            'Xtest_filtered': Xtest,
            'ytest_filtered': ytest,
            'removed_nines': len(nines_to_remove)
        }

        return answer, X, y, Xtest, ytest

    # --------------------------------------------------------------------------
    """
    C. Repeat part 1.C for this dataset but use a support vector machine (SVC in sklearn). 
        Make sure to use a stratified cross-validation strategy. In addition to regular accuracy 
        also print out the mean/std of the F1 score, precision, and recall. As usual, use 5 splits. 
        Is precision or recall higher? Explain. Finally, train the classifier on all the training data 
        and plot the confusion matrix.
        Hint: use the make_scorer function with the average='macro' argument for a multiclass dataset. 
    """

    def partC(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> dict[str, Any]:
        print("Evaluating SVC performance with Stratified K-Fold cross-validation\n")
        
        answer = {}
        svc_classifier = SVC(random_state=self.seed)
        stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)

        scoring_metrics = {
            'accuracy': make_scorer(accuracy_score),
            'recall': make_scorer(recall_score, average='macro'),
            'precision': make_scorer(precision_score, average='macro'),
            'f1': make_scorer(f1_score, average='macro'),
        }

        cv_results = cross_validate(svc_classifier, X, y, cv=stratified_cv, scoring=scoring_metrics)

        # Extracting and organizing the results
        metrics_summary = {metric: {'mean': np.mean(cv_results[f'test_{metric}']), 'std': np.std(cv_results[f'test_{metric}'])}
                           for metric in scoring_metrics.keys()}

        svc_classifier.fit(X, y)
        train_confusion_matrix = confusion_matrix(y, svc_classifier.predict(X))
        test_confusion_matrix = confusion_matrix(ytest, svc_classifier.predict(Xtest))

        # Updating the answer dictionary
        answer['scores'] = metrics_summary
        answer['cv'] = stratified_cv
        answer['clf'] = svc_classifier
        answer['confusion_matrix_train'] = train_confusion_matrix
        answer['confusion_matrix_test'] = test_confusion_matrix

        # Additional insights
        precision_higher_than_recall = metrics_summary['precision']['mean'] > metrics_summary['recall']['mean']
        answer['is_precision_higher_than_recall'] = precision_higher_than_recall
        explanation = "Precision is higher than recall, indicating a lower false positive rate relative to the false negative rate." if precision_higher_than_recall else "Recall is higher, indicating a prioritization of minimizing false negatives over false positives."
        answer['explain_is_precision_higher_than_recall'] = explanation

        return answer

    # --------------------------------------------------------------------------
    """
    D. Repeat the same steps as part 3.C but apply a weighted loss function (see the class_weights parameter).  Print out the class weights, and comment on the performance difference. Use the `compute_class_weight` argument of the estimator to compute the class weights. 
    """

    def partD(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> dict[str, Any]:
        """"""
        # Enter your code and fill the `answer` dictionary
        answer = {}

        """
        Answer is a dictionary with the following keys: 
        - "scores" : a dictionary with the mean/std of the F1 score, precision, and recall
        - "cv" : the cross-validation strategy
        - "clf" : the classifier
        - "class_weights" : the class weights
        - "confusion_matrix_train" : the confusion matrix for the training set
        - "confusion_matrix_test" : the confusion matrix for the testing set
        - "explain_purpose_of_class_weights" : explanatory string
        - "explain_performance_difference" : explanatory string

        answer["scores"] has the following keys: 
        - "mean_accuracy" : the mean accuracy
        - "mean_recall" : the mean recall
        - "mean_precision" : the mean precision
        - "mean_f1" : the mean f1
        - "std_accuracy" : the std accuracy
        - "std_recall" : the std recall
        - "std_precision" : the std precision
        - "std_f1" : the std f1

        Recall: The scores are based on the results of the cross-validation step
        """

        return answer
