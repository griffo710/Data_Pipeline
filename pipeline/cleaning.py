from sklearn.base import BaseEstimator, TransformerMixin


class MissingValueImputer(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible transformer that fills missing values
    using statistics learned from training data.

    This class follows the sklearn estimator contract:
    - Configuration is defined in __init__
    - Data-driven learning happens in fit()
    - Learned values are applied in transform()

    Why this matters:
    -----------------
    - Prevents data leakage
    - Ensures reproducibility
    - Allows safe use in sklearn Pipelines and GridSearch
    """

    def __init__(
        self,
        numerical_strategy="mean",
        categorical_strategy="mode",
        date_strategy="mode",
    ):  # For rules.
        """
        Initialize the imputer with a chosen imputation strategy.

        Parameters
        ----------
        strategy : str, default="mean"
            Strategy used to compute replacement values for missing data.
            Supported values:
            - "mean": replace missing values with the column mean
            - "median": replace missing values with the column median

        Notes
        -----
        No data processing must happen here.
        This method should only store configuration parameters.
        """
        self.numerical_strategy = numerical_strategy
        self.categorical_strategy = categorical_strategy
        self.date_strategy = date_strategy
        self.numeric_fill_values_ = {}
        self.categorical_fill_values_ = {}
        self.date_fill_values_ = {}

    def fit(self, X, y=None):  # For learning.
        """
        Learn imputation values from the training dataset.

        This method inspects the training data and computes the
        replacement value for each column according to the chosen
        strategy.

        Parameters
        ----------
        X : pandas.DataFrame
            Training feature data.
        y : ignored
            Present for sklearn compatibility.

        Returns
        -------
        self : MissingValueImputer
            The fitted transformer with learned statistics stored.

        Important
        ---------
        - This method must NOT modify X.
        - Statistics learned here will be reused during transform().
        """
        numeric_cols = X.select_dtypes(include="number").columns
        categorical_cols = X.select_dtypes(include=["object", "string"]).columns
        date_columns = X.select_dtypes(include=["datetime"]).columns

        for col in numeric_cols:
            if self.numerical_strategy == "mean":
                self.numeric_fill_values_[col] = round(X[col].mean(), 2)
            elif self.numerical_strategy == "median":
                self.numeric_fill_values_[col] = round(X[col].median(), 2)
            else:
                raise ValueError("Invalid imputation method for numerical columns.")

        for col in categorical_cols:
            valid_values = X[col].dropna()

            if valid_values.empty:
                raise ValueError(f"No valid categorical values to learn from in {col}")

            if self.categorical_strategy == "mode":
                self.categorical_fill_values_[col] = valid_values.mode().iloc[0]

            else:
                raise ValueError("Imputation method used is not allowed.")

        for col in date_columns:
            valid_dates = X[col].dropna()

            if valid_dates.empty:
                raise ValueError(f"No valid dates to learn from {col}.")

            else:
                if self.date_strategy == "mode":
                    self.date_fill_values_[col] = valid_dates.mode(dropna=True)[0]

                elif self.date_strategy == "min":
                    self.date_fill_values_[col] = valid_dates.min()

                elif self.date_strategy == "max":
                    self.date_fill_values_[col] = valid_dates.max()

                else:
                    raise ValueError("Invalid imputation method for date columns.")
        return self

    def transform(self, X, logger=None):  # For applying.
        """
        Apply learned imputation values to new data.

        This method fills missing values using statistics computed
        during fit(). No new statistics are calculated here.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature data to transform (validation, test, or production).

        Returns
        -------
        pandas.DataFrame
            A new DataFrame with missing values filled.

        Important
        ---------
        - Does NOT recompute statistics
        - Does NOT modify input data in-place
        - Guarantees deterministic behavior
        """
        X = X.copy()
        missing_before = X.isna().sum()
        for col, value in self.numeric_fill_values_.items():
            X[col] = X[col].fillna(value)

        for col, value in self.categorical_fill_values_.items():
            X[col] = X[col].fillna(value)

        for col, value in self.date_fill_values_.items():
            X[col] = X[col].fillna(value)

        missing_after = X.isna().sum()
        imputed_counts = missing_before - missing_after
        imputed_counts = imputed_counts[imputed_counts > 0]

        if logger:
            if not imputed_counts.empty:
                logger.info(
                    "Missing values imputed per column:\n%s", imputed_counts.to_string()
                )
            else:
                logger.info("No missing values imputation required.")

            logger.info("Numeric imputation values used: %s", self.numeric_fill_values_)
            logger.info(
                "Categorical imputation values used: %s", self.categorical_fill_values_
            )
            logger.info("Date imputation values used: %s", self.date_fill_values_)

        return X


class DuplicateRemover(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible transformer that removes duplicate rows
    from a dataset based on a fixed, user-defined rule.

    Duplicate removal is rule-based and does not learn from data,
    making it safe to apply during both training and inference.
    """

    def __init__(self, subset=None, keep="first"):
        """
        Initialize the duplicate removal strategy.

        Parameters
        ----------
        subset : list of str or None, default=None
            Columns to consider when identifying duplicates.
            If None, all columns are used.

        keep : {'first', 'last', False}, default='first'
            Determines which duplicate rows to keep:
            - 'first': keep the first occurrence
            - 'last': keep the last occurrence
            - False: drop all duplicates
        """
        self.subset = subset
        self.keep = keep

    def fit(self, X, y=None):
        """
        Fit the transformer.

        Duplicate removal does not require learning from data,
        so this method performs no action and exists only for
        sklearn compatibility.

        Returns
        -------
        self : DuplicateRemover
            The transformer itself.
        """
        return self

    def transform(self, X, logger=None):
        """
        Remove duplicate rows according to the configured rule.

        Parameters
        ----------
        X : pandas.DataFrame
            Input feature data.

        Returns
        -------
        pandas.DataFrame
            A new DataFrame with duplicate rows removed.

        Notes
        -----
        - Does not modify input data in-place
        - Behavior is deterministic and reproducible
        - Stores metadata about removed duplicates
        """
        X = X.copy()

        duplicate_data = X.duplicated(subset=self.subset, keep=self.keep)
        print("Total duplicate rows:", duplicate_data.sum())

        self.removed_indices_ = X.index[duplicate_data].tolist()
        self.removed_rows_ = X.loc[duplicate_data].copy()
        self.removed_count_ = int(duplicate_data.sum())

        X_clean = X.loc[~duplicate_data]
        if logger:
            if self.removed_count > 0:
                logger.info(
                    "Duplicate removal complete. Rows removed: %s", self.removed_count
                )
                logger.info("Removed duplicate indices: %s", self.removed_indices)
            else:
                logger.info("No duplicate rows detected.")

        return X_clean
