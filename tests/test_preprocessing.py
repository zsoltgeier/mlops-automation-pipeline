"""Prove train-only statistics and consistent serialized inference."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from sklearn.base import clone

from garments_mlops.data import RAW_FEATURES, TARGET, clean_dataset
from garments_mlops.preprocessing import FeatureCleaner, build_preprocessor
from garments_mlops.splitting import split_by_date


def test_statistics_are_learned_only_from_train(records: pd.DataFrame) -> None:
    split = split_by_date(clean_dataset(records))
    train = split.train[RAW_FEATURES].copy()
    current = split.validation[RAW_FEATURES].copy()
    train.loc[0, "wip"] = 10.0
    current["wip"] = 1e9
    current["department"] = "unseen_department"
    pipeline = build_preprocessor().fit(train)
    numeric = pipeline.named_steps["columns"].named_transformers_["numeric"]
    assert numeric.named_steps["impute"].statistics_[2] == train.wip.median()
    assert numeric.named_steps["scale"].mean_[1] == train.smv.mean()
    before = numeric.named_steps["impute"].statistics_.copy()
    transformed = pipeline.transform(current)
    assert np.isfinite(transformed).all()
    assert transformed.shape[1] == pipeline.transform(train).shape[1]
    assert_allclose(numeric.named_steps["impute"].statistics_, before)
    names = pipeline.get_feature_names_out()
    assert len(names) == transformed.shape[1]
    assert not any(TARGET in name or "unseen_department" in name for name in names)


def test_missing_training_columns_and_new_missing_values_are_supported(records: pd.DataFrame) -> None:
    train = records[RAW_FEATURES].copy()
    train["wip"] = np.nan
    train["department"] = None
    pipeline = build_preprocessor().fit(train)
    current = records[RAW_FEATURES].copy()
    current["smv"] = np.nan
    assert np.isfinite(pipeline.transform(train)).all()
    assert np.isfinite(pipeline.transform(current)).all()


def test_serialized_clone_has_identical_batch_and_single_predictions(records: pd.DataFrame, tmp_path: Path) -> None:
    features = records[RAW_FEATURES]
    pipeline = clone(build_preprocessor()).fit(features)
    path = tmp_path / "preprocessor.joblib"
    joblib.dump(pipeline, path)
    restored = joblib.load(path)
    assert_allclose(restored.transform(features), pipeline.transform(features))
    assert_allclose(restored.transform(features.iloc[:1]), restored.transform(features)[:1])
    cleaner = FeatureCleaner().fit(features, np.zeros(len(features)))
    assert cleaner.n_features_in_ == len(RAW_FEATURES)
    assert len(cleaner.get_feature_names_out(RAW_FEATURES)) == len(RAW_FEATURES) + 2


def test_raw_and_cleaned_records_share_one_transformation(records: pd.DataFrame) -> None:
    features = records[RAW_FEATURES]
    pipeline = build_preprocessor().fit(features)
    cleaned = clean_dataset(records)[RAW_FEATURES]
    assert_allclose(pipeline.transform(features), pipeline.transform(cleaned))
    with pytest.raises(ValueError, match="Target must not"):
        pipeline.transform(records)
