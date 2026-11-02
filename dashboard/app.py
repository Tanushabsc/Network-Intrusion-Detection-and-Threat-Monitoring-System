import os
import json

import pandas as pd
import requests
import streamlit as st

# ---------------- CONFIGURATION ----------------

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
DATA_PATH = "data/processed/test.csv"

st.set_page_config(
    page_title="NetGuard AI",
    page_icon="🛡️",
    layout="wide"
)

# ---------------- CUSTOM DESIGN ----------------

st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
.metric-card {
    background-color: #1b2430;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #303b4b;
}
.title-text {
    font-size: 32px;
    font-weight: bold;
    color: #ffffff;
}
.subtitle {
    color: #aab6c5;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------

st.markdown(
    '<div class="title-text">🛡️ NetGuard AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-Powered Network Intrusion Detection and Threat Monitoring</div>',
    unsafe_allow_html=True
)

st.divider()

# ---------------- API STATUS ----------------

try:
    health_response = requests.get(
        f"{API_URL}/health",
        timeout=5
    )
    api_online = health_response.status_code == 200
except requests.RequestException:
    api_online = False

col1, col2 = st.columns([1, 3])

with col1:
    if api_online:
        st.success("API ONLINE")
    else:
        st.error("API OFFLINE")

with col2:
    st.caption("Model: Random Forest | Dataset: UNSW-NB15")

if not api_online:
    st.warning(
        "Start the FastAPI server before running predictions."
    )
    st.stop()

# ---------------- LOAD DATA ----------------

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

try:
    data = load_data()
except Exception as error:
    st.error(f"Could not load dataset: {error}")
    st.stop()

# ---------------- SIDEBAR ----------------

st.sidebar.title("Monitoring Controls")

st.sidebar.info(
    "This dashboard simulates network-flow monitoring "
    "using records from the UNSW-NB15 test dataset."
)

record_limit = st.sidebar.slider(
    "Number of records to analyse",
    min_value=10,
    max_value=200,
    value=50,
    step=10
)

st.sidebar.caption(
    "Each record is submitted to the FastAPI prediction endpoint."
)

# ---------------- PREDICTION FUNCTION ----------------

def predict_record(row):
    features = json.loads(
        row.drop(labels=["label"]).to_json()
    )

    response = requests.post(
        f"{API_URL}/predict",
        json={"features": features},
        timeout=30
    )

    response.raise_for_status()
    return response.json()


# ---------------- MONITORING ----------------

st.subheader("Network Threat Monitoring")

st.write(
    "Analyse network-flow records and identify suspicious activity."
)

if st.button("▶ Start Network Analysis", type="primary"):

    results = []
    progress = st.progress(0)
    status = st.empty()

    selected_data = data.head(record_limit)

    for index, (_, row) in enumerate(selected_data.iterrows()):

        try:
            prediction = predict_record(row)

            results.append({
                "Record": index + 1,
                "Prediction": prediction["prediction"],
                "Risk Level": prediction["risk_level"],
                "Attack Probability": prediction["attack_probability"],
                "Actual Class": (
                    "Attack" if int(row["label"]) == 1
                    else "Benign"
                )
            })

        except Exception as error:
            st.error(
                f"Error processing record {index + 1}: {error}"
            )
            break

        progress.progress((index + 1) / record_limit)
        status.write(
            f"Analysing network record {index + 1} "
            f"of {record_limit}..."
        )

    st.session_state["results"] = results

    status.success(
        f"Analysis completed: {len(results)} records processed."
    )

# ---------------- DISPLAY RESULTS ----------------

if "results" in st.session_state:

    results_df = pd.DataFrame(
        st.session_state["results"]
    )

    if not results_df.empty:

        total = len(results_df)

        attacks = (
            results_df["Prediction"] == "Attack"
        ).sum()

        benign = (
            results_df["Prediction"] == "Benign"
        ).sum()

        high_risk = (
            results_df["Risk Level"] == "HIGH"
        ).sum()

        st.divider()
        st.subheader("Security Overview")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Records Analysed", total)
        c2.metric("Attacks Detected", int(attacks))
        c3.metric("Benign Records", int(benign))
        c4.metric("High-Risk Records", int(high_risk))

        # ---------------- CHARTS ----------------

        st.divider()
        st.subheader("Threat Distribution")

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.write("Prediction Distribution")
            distribution = results_df[
                "Prediction"
            ].value_counts()

            st.bar_chart(distribution)

        with chart_col2:
            st.write("Attack Probability")
            st.line_chart(
                results_df.set_index("Record")[
                    "Attack Probability"
                ]
            )

        # ---------------- ALERTS ----------------

        st.divider()
        st.subheader("Threat Alerts")

        attack_records = results_df[
            results_df["Prediction"] == "Attack"
        ]

        if not attack_records.empty:
            for _, alert in attack_records.head(10).iterrows():

                st.error(
                    f"Record {alert['Record']} | "
                    f"ATTACK DETECTED | "
                    f"Attack probability: "
                    f"{alert['Attack Probability']:.2%}"
                )
        else:
            st.success("No attacks detected in these records.")

        # ---------------- RESULTS TABLE ----------------

        st.divider()
        st.subheader("Network Analysis Results")

        st.dataframe(
            results_df,
            use_container_width=True,
            hide_index=True
        )

        # ---------------- DOWNLOAD ----------------

        csv_data = results_df.to_csv(index=False)

        st.download_button(
            label="Download Analysis Report",
            data=csv_data,
            file_name="netguard_threat_report.csv",
            mime="text/csv"
        )

st.divider()

st.caption(
    "NetGuard AI | Network Intrusion Detection and Threat Monitoring | "
    "Powered by Machine Learning"
)
