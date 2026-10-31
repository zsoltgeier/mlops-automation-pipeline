"""Shared scikit-learn preprocessing fitted exclusively on training records."""

from typing import Self

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from garments_mlops.data import CALENDAR_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, RAW_FEATURES, clean_features


class FeatureCleaner(TransformerMixin, BaseEstimator):
    """Adapt stateless record normalization to the sklearn transformer interface."""

    def fit(self, X: pd.DataFrame, y: object = None) -> Self:
        """Validate inputs while ignoring labels and population statistics."""
        clean_features(X)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the same normalization to training and prediction inputs."""
        return clean_features(X)

    def get_feature_names_out(self, input_features: object = None) -> NDArray[np.object_]:
        """Expose the cleaned schema to downstream sklearn components."""
        return np.asarray([*RAW_FEATURES, *CALENDAR_FEATURES], dtype=object)


def build_preprocessor() -> Pipeline:
    """Return an unfitted, cloneable preprocessing pipeline."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    columns = ColumnTransformer([
        ("numeric", numeric, [*NUMERIC_FEATURES, *CALENDAR_FEATURES]),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ], remainder="drop")
    return Pipeline([("clean", FeatureCleaner()), ("columns", columns)])
