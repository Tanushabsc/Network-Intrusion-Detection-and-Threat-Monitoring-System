
import os
import requests
import joblib
import pandas as pd
import streamlit as st

# ---------------- CONFIGURATION ----------------

MODEL_URL = (
    "https://github.com/Tanushabsc/"
    "Network-Intrusion-Detection-and-Threat-Monitoring-System/"
    "releases/download/v1.0/intrusion_model.joblib"
)

MODEL_PATH = "models/intrusion_model.joblib"
DATA_PATH = "data/processed/test.csv"

st.set_page_config(
    page_title="NetGuard AI",
    page_icon="🛡️",
    layout="wide"
)

# ---------------- DESIGN ----------------

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

# ---------------- DOWNLOAD AND LOAD MODEL ----------------

@st.cache_resource
def load_model():

    os.makedirs("models", exist_ok=True)

    if not os.path.exists(MODEL_PATH):

        response = requests.get(
            MODEL_URL,
            stream=True,
            timeout=300
        )

        response.raise_for_status()

        with open(MODEL_PATH, "wb") as model_file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    model_file.write(chunk)

    return joblib.load(MODEL_PATH)


with st.spinner("Loading the trained Random Forest model..."):
    try:
        model = load_model()
        model_online = True

    except Exception as error:
        model_online = False
        st.error(f"Could not load the model: {error}")
        st.stop()

# ---------------- MODEL STATUS ----------------

col1, col2 = st.columns([1, 3])

with col1:
    st.success("MODEL ONLINE")

with col2:
    st.caption("Model: Random Forest | Dataset: UNSW-NB15")

st.caption("Predictions are performed directly using the trained machine learning model.")

# ---------------- LOAD DATA ----------------

st.subheader("Dataset")

uploaded_file = st.file_uploader(
    "Upload UNSW-NB15 test dataset (CSV)",
    type=["csv"]
)

@st.cache_data
def load_local_data():
    return pd.read_csv(DATA_PATH)

data = None

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.success("Uploaded dataset loaded successfully.")

    except Exception as error:
        st.error(f"Could not read uploaded CSV: {error}")
        st.stop()

elif os.path.exists(DATA_PATH):
    try:
        data = load_local_data()
        st.success("Local test dataset loaded successfully.")

    except Exception as error:
        st.error(f"Could not load local dataset: {error}")
        st.stop()

else:
    st.info(
        "Upload your UNSW-NB15 testing-set CSV to start network analysis. "
        "The online dashboard does not automatically access your computer's files."
    )
    st.stop()

if "label" not in data.columns:
    st.error(
        "The uploaded CSV must contain a 'label' column "
        "with 0 for Benign and 1 for Attack."
    )
    st.stop()

if data.empty:
    st.error("The uploaded dataset is empty.")
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
    max_value=min(200, len(data)),
    value=min(50, len(data)),
    step=10
)

st.sidebar.caption(
    "Each selected network record is analysed by the trained model."
)

# ---------------- PREDICTION FUNCTION ----------------

def predict_record(row):

    # Remove target and unnecessary columns
    features = row.drop(
        labels=["label", "id", "attack_cat"],
        errors="ignore"
    )

    # Convert to a one-row DataFrame
    features = pd.DataFrame([features])

    prediction_value = int(model.predict(features)[0])

    probabilities = model.predict_proba(features)[0]

    # Find probability belonging to attack class (1)
    classes = list(model.classes_)

    if 1 in classes:
        attack_probability = float(probabilities[classes.index(1)])
    else:
        attack_probability = 0.0

    prediction = "Attack" if prediction_value == 1 else "Benign"

    # Risk classification based on attack probability
    if attack_probability >= 0.70:
        risk_level = "HIGH"
    elif attack_probability >= 0.40:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "prediction": prediction,
        "risk_level": risk_level,
        "attack_probability": attack_probability
    }

# ---------------- MONITORING ----------------

st.divider()

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

        # ---------------- OVERVIEW ----------------

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

        # ---------------- DOWNLOAD REPORT ----------------

        csv_data = results_df.to_csv(index=False)

        st.download_button(
            label="Download Analysis Report",
            data=csv_data,
            file_name="netguard_threat_report.csv",
            mime="text/csv"
        )

# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "NetGuard AI | Network Intrusion Detection and Threat Monitoring | "
    "Powered by Machine Learning"
)
