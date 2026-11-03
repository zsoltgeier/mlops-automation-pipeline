"""Check JSON contracts, missing values and unwanted input rejection."""

import json

import pandas as pd
from pydantic import ValidationError
import pytest

from garments_mlops.api_models import GarmentRecord, PredictRequest, TrainRequest


@pytest.fixture
def record(records: pd.DataFrame) -> dict[str, object]:
    """Convert an original observation into the public ISO-date contract."""
    value = records.iloc[0].drop(labels="actual_productivity").to_dict()
    value["date"] = "2015-01-01"
    return value


def test_record_normalization_preserves_fractional_workers_and_null_wip(record: dict[str, object]) -> None:
    record["department"] = " finishing "
    record["wip"] = None
    parsed = GarmentRecord.model_validate(record)
    assert parsed.department == "finishing"
    assert parsed.wip is None
    assert parsed.no_of_workers == 30.5
    assert parsed.model_dump(mode="json")["date"] == "2015-01-01"
    assert GarmentRecord.model_validate({**record, "department": "sweing"}).department == "sewing"
    assert GarmentRecord.model_validate({key: value for key, value in record.items() if key != "wip"}).wip is None


@pytest.mark.parametrize("field,value", [
    ("date", "01/01/2015"), ("date", 123456), ("date", "2015-02-30"),
    ("department", 42), ("department", "unknown"), ("quarter", "Quarter6"), ("day", "NotADay"),
    ("team", 0), ("team", True), ("team", 1.5), ("smv", -1), ("smv", "2.0"),
    ("wip", float("inf")), ("over_time", float("nan")), ("no_of_style_change", 0.5),
    ("no_of_workers", None), ("actual_productivity", 0.8), ("password", "must-not-be-logged"),
])
def test_invalid_record_fields_are_rejected(record: dict[str, object], field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        GarmentRecord.model_validate({**record, field: value})


def test_batch_contract_and_limits(record: dict[str, object]) -> None:
    assert len(PredictRequest.model_validate({"records": [record, record]}).records) == 2
    assert PredictRequest.model_validate({"records": [record]}).source == "live"
    with pytest.raises(ValidationError):
        PredictRequest.model_validate({"records": [record], "source": "unknown"})
    for body in [{"records": []}, {"records": [record] * 1001}, {"records": [record], "secret": "ignored"}]:
        with pytest.raises(ValidationError):
            PredictRequest.model_validate(body)
    assert json.loads(TrainRequest().model_dump_json()) == {"model": "random_forest"}
    with pytest.raises(ValidationError):
        TrainRequest.model_validate({"model": "unknown"})
    with pytest.raises(ValidationError):
        TrainRequest.model_validate({"minimum_improvement": 0})
