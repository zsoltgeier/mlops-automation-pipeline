"""Thin Streamlit presentation for persisted feature-drift reports and model metadata."""

import os

import streamlit as st

from garments_mlops.monitor_cli import initialize_monitoring, service_config, submit_demo
from garments_mlops.monitoring import generate_monitoring, read_artifact, training_overview
from garments_mlops.monitoring_data import NoCurrentData
from garments_mlops.tracking import champion_version, tracking_client


def render_dashboard() -> None:
    """Render explicit data origins, native Evidently HTML, and independent training results."""
    st.set_page_config(page_title="Produktivitás monitoring", layout="wide")
    st.title("Produktivitás – adatváltozás és modellek")
    st.caption("A drift a bemenetek eloszlását vizsgálja. "
               "Célértékek nélkül nem mér modellpontosságot, és nem vált modellt.")
    source = st.radio("Predikciós adatok", ["auto", "live", "demo"],
                      format_func=lambda value: {"auto": "Automatikus – éles, ennek hiányában demó",
                                                 "live": "Éles kérések", "demo": "Demó – szintetikus kérések"}[value])
    st.button("Riport frissítése")
    try:
        config = service_config()
        api_url = os.getenv("MLOPS_API_URL", "http://127.0.0.1:8000")
        if st.button("Demókérések küldése", help="200 determinisztikus, módosított rekordot küld a predict API-nak."):
            submit_demo(config, api_url)
            st.success("A megjelölt demókérések elkészültek. Válaszd a demóadatokat a riporthoz.")
        client = tracking_client(config)
        active = champion_version(client, config.model_name)
        st.metric("Aktuális champion", str(active.version) if active is not None else "Nincs")
        st.subheader("Tanítási eredmények")
        st.dataframe(training_overview(client, config), hide_index=True)
        try:
            summary = (initialize_monitoring(config, api_url) if source == "auto"
                       else generate_monitoring(config, "demo" if source == "demo" else "live"))
        except NoCurrentData:
            st.info("Ehhez a forráshoz még nincs sikeres predikció. Küldj API-kéréseket, vagy válaszd a demót.")
            return
        if summary.source == "demo":
            st.warning("DEMÓ: szintetikusan módosított bemenetek, valódi API-predikciókkal. Nem üzemi megfigyelések.")
        if summary.current_rows < 100:
            st.warning("Kis minta: az eloszláskülönbség bizonytalan, az eredmény tájékoztató jellegű.")
        st.metric("Eltérő eloszlású oszlopok", f"{summary.drift_share:.1%}")
        st.write({"Referenciasorok": summary.reference_rows, "Aktuális sorok": summary.current_rows,
                  "Predikciós modellverziók": summary.prediction_model_versions,
                  "Riport ideje (UTC)": summary.created_at_utc, "MLflow run": summary.run_id})
        st.dataframe([item.model_dump() for item in summary.columns], hide_index=True)
        html = read_artifact(client, summary.run_id, "monitoring/report.html")
        st.download_button("Evidently HTML letöltése", html, "drift-report.html", "text/html")
        st.subheader("Evidently drift report")
        st.iframe(html, height=900, alt="Evidently adatdrift report")
    except Exception as error:
        st.error(f"A monitoring most nem érhető el ({type(error).__name__}). Ellenőrizd a szolgáltatások állapotát.")


if __name__ == "__main__":
    render_dashboard()
