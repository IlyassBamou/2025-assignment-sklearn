"""Assignment - making a sklearn estimator and cv splitter.

The goal of this assignment is to implement by yourself:
- a scikit-learn estimator for the KNearestNeighbors for classification
  tasks and check that it is working properly.
- a scikit-learn CV splitter where the splits are based on a Pandas
  DateTimeIndex.
Detailed instructions for question 1:
The nearest neighbor classifier predicts for a point X_i the target y_k of
the training sample X_k which is the closest to X_i. We measure proximity with
the Euclidean distance. The model will be evaluated with the accuracy (average
number of samples corectly classified). You need to implement the `fit`,
`predict` and `score` methods for this class. The code you write should pass
the test we implemented. You can run the tests by calling at the root of the
repo `pytest test_sklearn_questions.py`. Note that to be fully valid, a
scikit-learn estimator needs to check that the input given to `fit` and
`predict` are correct using the `validate_data, check_is_fitted` functions
imported in this file.
You can find more information on how they should be used in the following doc:
https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator.
Make sure to use them to pass `test_nearest_neighbor_check_estimator`.
Detailed instructions for question 2:
The data to split should contain the index or one column in
datatime format. Then the aim is to split the data between train and test
sets when for each pair of successive months, we learn on the first and
predict of the following. For example if you have data distributed from
november 2020 to march 2021, you have have 4 splits. The first split
will allow to learn on november data and predict on december data, the
second split to learn december and predict on january etc.
We also ask you to respect the pep8 convention: https://pep8.org. This will be
enforced with `flake8`. You can check that there is no flake8 errors by
calling `flake8` at the root of the repo.
Finally, you need to write docstrings for the methods you code and for the
class. The docstring will be checked using `pydocstyle` that you can also
call at the root of the repo.
Hints
-----
- You can use the function:
from sklearn.metrics.pairwise import pairwise_distances
to compute distances between 2 sets of samples.
"""
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin

from sklearn.model_selection import BaseCrossValidator

from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import validate_data
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils.multiclass import check_classification_targets


class KNearestNeighbors(ClassifierMixin, BaseEstimator):
    """KNearestNeighbors classifier.

    Predicts the class label of a sample based on the majority class of its
    k nearest neighbors in the training set, using Euclidean distance.

    Parameters
    ----------
    n_neighbors : int, default=1
        Number of neighbors to consider for classification.
    """

    def __init__(self, n_neighbors=1):  # noqa: D107
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """Fitting function.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data to train the model.
        y : ndarray, shape (n_samples,)
            Labels associated with the training data.

        Returns
        -------
        self : instance of KNearestNeighbors
            The current instance of the classifier
        """
        X, y = validate_data(self, X, y=y, reset=True)
        y = np.asarray(y)
        y = np.ravel(y)

        check_classification_targets(y)

        if y.ndim != 1:
            raise ValueError("y should be a 1D array-like.")

        if not isinstance(self.n_neighbors, int) or self.n_neighbors <= 0:
            raise ValueError("n_neighbors must be a strictly positive int.")

        if self.n_neighbors > X.shape[0]:
            raise ValueError(
                f"n_neighbors cannot be greater than n_samples "
                f"(n_samples = {X.shape[0]})."
            )

        # Store training data
        self.X_ = X

        # Encode classes to allow fast voting via bincount
        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        self.y_encoded_ = y_encoded.astype(int)
        return self

        return self

    def predict(self, X):
        """Predict function.

        Parameters
        ----------
        X : ndarray, shape (n_test_samples, n_features)
            Data to predict on.

        Returns
        -------
        y : ndarray, shape (n_test_samples,)
            Predicted class labels for each test data sample.
        """
        # FIX 2: Check if fit has been called
        check_is_fitted(self, attributes=["X_", "y_encoded_", "classes_"])
        X = validate_data(self, X, reset=False)

        distances = pairwise_distances(X, self.X_, metric="euclidean")
        nn_idx = np.argsort(distances, axis=1)[:, : self.n_neighbors]

        n_test = X.shape[0]
        y_pred_encoded = np.empty(n_test, dtype=int)

        n_classes = self.classes_.shape[0]
        for i in range(n_test):
            neigh_labels = self.y_encoded_[nn_idx[i]]
            counts = np.bincount(neigh_labels, minlength=n_classes)
            y_pred_encoded[i] = int(np.argmax(counts))

        return self.classes_[y_pred_encoded]

    def score(self, X, y):
        """Calculate the score of the prediction.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data to score on.
        y : ndarray, shape (n_samples,)
            target values.

        Returns
        ----------
        score : float
            Accuracy of the model computed for the (X, y) pairs.
        """
        check_is_fitted(self, attributes=["X_", "y_encoded_", "classes_"])
        y = np.asarray(y)
        y_pred = self.predict(X)
        return float(np.mean(y_pred == y))


class MonthlySplit(BaseCrossValidator):
    """CrossValidator based on monthly split.

    Split data based on the given `time_col` (or default to index). Each split
    corresponds to one month of data for the training and the next month of
    data for the test.

    Parameters
    ----------
    time_col : str, defaults to 'index'
        Column of the input DataFrame that will be used to split the data. This
        column should be of type datetime. If split is called with a DataFrame
        for which this column is not a datetime, it will raise a ValueError.
        To use the index as column just set `time_col` to `'index'`.
    """

    def __init__(self, time_col='index'):  # noqa: D107
        self.time_col = time_col

    def __repr__(self):
        """Representation for the cross-validator."""
        return f"MonthlySplit(time_col='{self.time_col}')"

    def _extract_times(self, X):
        """Validate input and return a DataFrame and datetime index."""
        if isinstance(X, pd.Series):
            X_df = X.to_frame()
        elif isinstance(X, pd.DataFrame):
            X_df = X
        else:
            raise ValueError("X must be a pandas DataFrame or Series.")

        if self.time_col == "index":
            times = X_df.index
        else:
            if self.time_col not in X_df.columns:
                raise ValueError("time_col must be a column of X or 'index'.")
            times = X_df[self.time_col]

        if not pd.api.types.is_datetime64_any_dtype(times):
            raise ValueError("The time index/column must be datetime-like.")

        return X_df, pd.DatetimeIndex(times)

    def _get_time_data(self, X):
        """Extract and validate time series data."""
        if not isinstance(X, (pd.DataFrame, pd.Series)):
            # Create a DataFrame to ensure we can access index or columns
            X = pd.DataFrame(X)

        if self.time_col == 'index':
            time_series = X.index
        else:
            if self.time_col not in X.columns:
                raise ValueError(
                    f"Time column '{self.time_col}' not found in X."
                )
            time_series = X[self.time_col]
        # Ensure the series is in datetime format
        try:
            time_series = pd.to_datetime(time_series)
        except ValueError:
            raise ValueError(
                "The specified time column or index is not in datetime format."
            )
        return X, time_series.reset_index(drop=True)  # Consistent 0..N-1 index

    def get_n_splits(self, X, y=None, groups=None):
        """Return the number of splitting iterations in the cross-validator.

        ... (docstring omitted for brevity) ...
        """
        _, times = self._extract_times(X)
        months = times.to_period("M")
        n_months = months.unique().shape[0]
        return int(max(n_months - 1, 0))

    def split(self, X, y, groups=None):
        """Generate indices to split data into training and test set.

        ... (docstring omitted for brevity) ...

        Yields
        ------
        idx_train : ndarray
            The training set indices for that split.
        idx_test : ndarray
            The testing set indices for that split.
        """
        X_df, times = self._extract_times(X)

        if y is not None and len(y) != len(X_df):
            raise ValueError("X and y must be the same length.")

        months = times.to_period("M")
        uniq_months = np.array(sorted(months.unique()))

        for current_month, next_month in zip(
            uniq_months[:-1], uniq_months[1:]
        ):
            idx_train = np.where(months == current_month)[0]
            idx_test = np.where(months == next_month)[0]

            idx_train = idx_train[np.argsort(times[idx_train].values)]
            idx_test = idx_test[np.argsort(times[idx_test].values)]
            yield idx_train, idx_test
