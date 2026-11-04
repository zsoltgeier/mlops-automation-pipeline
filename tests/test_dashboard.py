"""Run the actual Streamlit script, including interactive controls and failure states."""

from unittest.mock import Mock

import pytest
from streamlit.testing.v1 import AppTest

from garments_mlops import monitor_cli, monitoring, tracking
from garments_mlops.monitoring import MonitoringSummary
from garments_mlops.monitoring_data import NoCurrentData
from garments_mlops.training_config import TrainingConfig


@pytest.fixture
def dashboard_mocks(monitoring_summary: MonitoringSummary, monkeypatch: pytest.MonkeyPatch) -> dict[str, Mock]:
    values = {"generate": Mock(return_value=monitoring_summary), "initialize": Mock(return_value=monitoring_summary),
              "demo": Mock(), "champion": Mock(return_value=Mock(version="2"))}
    monkeypatch.setattr(monitor_cli, "service_config", Mock(return_value=TrainingConfig()))
    monkeypatch.setattr(monitor_cli, "initialize_monitoring", values["initialize"])
    monkeypatch.setattr(monitor_cli, "submit_demo", values["demo"])
    monkeypatch.setattr(monitoring, "generate_monitoring", values["generate"])
    monkeypatch.setattr(monitoring, "training_overview", Mock(return_value=[{"validation_mae": .11}]))
    monkeypatch.setattr(monitoring, "read_artifact", Mock(return_value="<html><body>Evidently report</body></html>"))
    monkeypatch.setattr(tracking, "tracking_client", Mock())
    monkeypatch.setattr(tracking, "champion_version", values["champion"])
    return values


def test_dashboard_displays_report_and_source_controls(dashboard_mocks: dict[str, Mock]) -> None:
    app = AppTest.from_file("../src/garments_mlops/dashboard.py").run()
    assert not app.exception
    assert app.metric[0].value == "2"
    assert app.metric[1].value == "60.0%"
    assert len(app.dataframe) == 2
    assert not app.warning
    app.radio[0].set_value("demo").run()
    assert dashboard_mocks["generate"].call_args.args[1] == "demo"
    dashboard_mocks["generate"].return_value = dashboard_mocks["generate"].return_value.model_copy(
        update={"source": "demo", "current_rows": 3})
    app.button[0].click().run()
    assert len(app.warning) == 2
    assert "DEMÓ" in app.warning[0].value
    app.button[1].click().run()
    dashboard_mocks["demo"].assert_called_once()
    assert app.success
    app.radio[0].set_value("live").run()
    assert dashboard_mocks["generate"].call_args.args[1] == "live"
    assert not app.exception


def test_dashboard_handles_missing_source_and_redacts_backend_errors(dashboard_mocks: dict[str, Mock]) -> None:
    dashboard_mocks["champion"].return_value = None
    dashboard_mocks["generate"].side_effect = NoCurrentData()
    app = AppTest.from_file("../src/garments_mlops/dashboard.py").run()
    app.radio[0].set_value("live").run()
    assert app.metric[0].value == "Nincs"
    assert app.info
    dashboard_mocks["generate"].side_effect = OSError("private-token")
    app.button[0].click().run()
    assert app.error
    assert "private-token" not in app.error[0].value
    assert not app.exception
