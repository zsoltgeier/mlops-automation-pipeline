"""Dependency-free feature names shared by data processing and workflow clients."""

TARGET = "actual_productivity"
NUMERIC_FEATURES = [
    "targeted_productivity", "smv", "wip", "over_time", "incentive",
    "idle_time", "idle_men", "no_of_style_change", "no_of_workers",
]
CATEGORICAL_FEATURES = ["quarter", "department", "day", "team"]
RAW_FEATURES = ["date", *CATEGORICAL_FEATURES, *NUMERIC_FEATURES]
CALENDAR_FEATURES = ["month", "day_of_month"]
