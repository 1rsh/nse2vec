from imblearn.over_sampling import BorderlineSMOTE, RandomOverSampler
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    top_k_accuracy_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
import numpy as np

def evaluate_embeddings(
    embedding_matrix: np.ndarray,
    sectors: list,
    classifier: SVC = SVC(kernel="rbf", probability=True, verbose=False),
    smote: bool = True,
    top_k_accuracy: bool = False,
    n_splits: int = 5,
    random_state: int = 42,
    k: int = 3,
) -> None:
    """
    Calculate various scores for the sector classification.

    :param embedding_matrix: Input data for training and testing.
    :param sectors: The target sectors.
    :param classifier: A classifier object, defaults to SVC.
    :param smote: Boolean flag to apply SMOTE, defaults to True.
    :param top_k_accuracy: Boolean flag to calculate top-k accuracy.
    :param n_splits: Number of splits for cross-validation.
    :param random_state: Random state for reproducibility.
    :param k: The 'k' in top-k accuracy.
    """

    accuracy_list = []
    accuracy_list_top_k = []
    f1_list, recall_list, precision_list = [], [], []

    X = embedding_matrix # (num_tickers, EMBEDDING_DIM)
    y = np.array(sectors)#.reshape(-1, 1)
    if smote:
        sm = RandomOverSampler()
        X, y = sm.fit_resample(X, y)


    train_index = np.random.choice(range(len(X)), len(X)//10*7)
    test_index = list(set(range(len(X))) - set(train_index))

    X_train, y_train = X[train_index], y[train_index]
    X_test, y_test = X[test_index], y[test_index]

    classifier.fit(X_train, y_train.ravel())
    y_pred = classifier.predict(X_test)

    accuracy_list.append(classifier.score(X_test, y_test))
    if top_k_accuracy:
        accuracy_list_top_k.append(
            top_k_accuracy_score(y_test, classifier.predict_proba(X_test), k=k)
        )
    f1_list.append(f1_score(y_test, y_pred, average="weighted"))
    recall_list.append(recall_score(y_test, y_pred, average="weighted"))
    precision_list.append(precision_score(y_test, y_pred, average="weighted"))

    print("All labels in y_pred?:", (set(y_test[:]) - set(y_pred[:])) == set())

    print(f"Precision Score: {np.round(np.mean(precision_list), 2)}")
    print(f"Recall Score: {np.round(np.mean(recall_list), 2)}")
    print(f"F1 Score: {np.round(np.mean(f1_list), 2)}")
    print(f"Accuracy Score: {np.round(np.mean(accuracy_list), 2)}")
    if top_k_accuracy:
        print(f"Accuracy Score Top-{k}: {np.round(np.mean(accuracy_list_top_k), 2)}")