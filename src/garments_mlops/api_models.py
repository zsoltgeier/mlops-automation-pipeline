"""Validated, bounded request and response schemas for the productivity API."""

from datetime import date as CalendarDate
import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from garments_mlops.modeling import ModelKind, PromotionDecision

NonnegativeNumber = Annotated[float, Field(ge=0, allow_inf_nan=False, strict=True)]
PositiveInteger = Annotated[int, Field(ge=1, strict=True)]


class GarmentRecord(BaseModel):
    """Raw model inputs only; unknown fields and targets are rejected."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    date: CalendarDate
    quarter: Literal["Quarter1", "Quarter2", "Quarter3", "Quarter4", "Quarter5"]
    department: Literal["sewing", "finishing"]
    day: Literal["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    team: PositiveInteger
    targeted_productivity: NonnegativeNumber
    smv: NonnegativeNumber
    wip: NonnegativeNumber | None = None
    over_time: NonnegativeNumber
    incentive: NonnegativeNumber
    idle_time: NonnegativeNumber
    idle_men: NonnegativeNumber
    no_of_style_change: Annotated[int, Field(ge=0, strict=True)]
    no_of_workers: NonnegativeNumber

    @field_validator("date", mode="before")
    @classmethod
    def iso_date(cls, value: object) -> CalendarDate:
        """Require an ISO calendar date rather than accepting epoch timestamps."""
        if not isinstance(value, str) or re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is None:
            raise ValueError("date must be an ISO date string")
        return CalendarDate.fromisoformat(value)

    @field_validator("department", mode="before")
    @classmethod
    def normalize_department(cls, value: object) -> object:
        """Accept the original dataset spelling and surrounding whitespace."""
        if isinstance(value, str):
            return {"sweing": "sewing"}.get(value.strip(), value.strip())
        return value


class PredictRequest(BaseModel):
    """A bounded batch suitable for synchronous inference and auditing."""

    model_config = ConfigDict(extra="forbid")
    records: Annotated[list[GarmentRecord], Field(min_length=1, max_length=1000)]
    source: Literal["live", "demo"] = "live"


class PredictResponse(BaseModel):
    """Predictions tied to an exact Registry version and audit run."""

    request_id: str
    model_version: str
    run_id: str
    predictions: list[float]


class TrainRequest(BaseModel):
    """Train on configured server data without exposing paths or policy overrides."""

    model_config = ConfigDict(extra="forbid")
    model: ModelKind = "random_forest"


class MonitorRequest(BaseModel):
    """Choose an audit source without exposing storage paths or model policies."""

    model_config = ConfigDict(extra="forbid")
    source: Literal["live", "demo"] = "live"


class TrainResponse(BaseModel):
    """Completed training result with correlation, tracking and Registry identifiers."""

    request_id: str
    status: Literal["completed"] = "completed"
    run_id: str
    model_version: str
    model_uri: str
    champion_version: str
    metrics: dict[str, float]
    decision: PromotionDecision


class HealthResponse(BaseModel):
    """Readiness requires a reachable Registry and an active champion."""

    status: Literal["ready", "unavailable"]
    model_version: str | None = None
