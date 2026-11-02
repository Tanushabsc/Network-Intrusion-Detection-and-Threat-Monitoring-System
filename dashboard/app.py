
import os
import requests
import joblib
import pandas as pd
import streamlit as st

# =====================================================
# CONFIGURATION
# =====================================================

MODEL_URL = (
    "https://github.com/Tanushabsc/"
    "Network-Intrusion-Detection-and-Threat-Monitoring-System/"
    "releases/download/v1.0/intrusion_model.joblib"
)

MODEL_PATH = "models/intrusion_model.joblib"
DATA_PATH = "data/processed/test.csv"

# Previously measured model performance on the test set
MODEL_METRICS = {
    "Accuracy": 0.8921,
    "Precision": 0.8510,
    "Recall": 0.9746,
    "F1-score": 0.9086
}

st.set_page_config(
    page_title="NetGuard AI",
    page_icon="🛡️",
    layout="wide"
)

# =====================================================
# CUSTOM DESIGN
# =====================================================

st.markdown("""
<style>
.main {
    background-color: #0e1117;
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
.section-note {
    color: #aab6c5;
    font-size: 13px;
}
.alert-heading {
    font-size: 17px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================

st.markdown(
    '<div class="title-text">🛡️ NetGuard AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-Powered Network Intrusion Detection and Threat Monitoring</div>',
    unsafe_allow_html=True
)

st.divider()

# =====================================================
# LOAD TRAINED MODEL
# =====================================================

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
            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):
                if chunk:
                    model_file.write(chunk)

    return joblib.load(MODEL_PATH)


with st.spinner("Loading trained Random Forest model..."):

    try:
        model = load_model()

    except Exception as error:
        st.error(f"Could not load the model: {error}")
        st.stop()

# =====================================================
# MODEL STATUS
# =====================================================

status_col1, status_col2 = st.columns([1, 3])

with status_col1:
    st.success("MODEL ONLINE")

with status_col2:
    st.caption("Model: Random Forest | Dataset: UNSW-NB15")

st.caption(
    "Predictions are performed directly using the trained machine learning model."
)

# =====================================================
# MODEL PERFORMANCE
# =====================================================

st.divider()

st.subheader("Model Performance")

st.caption(
    "Previously measured evaluation results on the UNSW-NB15 test set."
)

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Accuracy",
    f"{MODEL_METRICS['Accuracy']:.2%}"
)

m2.metric(
    "Precision",
    f"{MODEL_METRICS['Precision']:.2%}"
)

m3.metric(
    "Recall",
    f"{MODEL_METRICS['Recall']:.2%}"
)

m4.metric(
    "F1-score",
    f"{MODEL_METRICS['F1-score']:.2%}"
)

with st.expander("What do these performance metrics mean?"):

    st.write(
        "**Accuracy:** Overall percentage of correctly classified records."
    )

    st.write(
        "**Precision:** How often the records predicted as attacks were actually attacks."
    )

    st.write(
        "**Recall:** How many actual attacks were successfully detected."
    )

    st.write(
        "**F1-score:** Balance between precision and recall."
    )

# =====================================================
# DATASET UPLOAD
# =====================================================

st.divider()
st.subheader("Dataset")

uploaded_file = st.file_uploader(
    "Upload UNSW-NB15 testing dataset (CSV)",
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
        "The online dashboard cannot automatically access files on your computer."
    )
    st.stop()


if "label" not in data.columns:
    st.error("Dataset must contain a 'label' column.")
    st.stop()

if data.empty:
    st.error("The uploaded dataset is empty.")
    st.stop()

if not data["label"].isin([0, 1]).all():
    st.error("The label column must contain only 0 (Benign) and 1 (Attack).")
    st.stop()

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("Monitoring Controls")

st.sidebar.info(
    "This dashboard simulates network-flow monitoring "
    "using records from the UNSW-NB15 test dataset."
)

if len(data) >= 10:

    record_limit = st.sidebar.slider(
        "Number of records to analyse",
        min_value=10,
        max_value=min(200, len(data)),
        value=min(50, len(data)),
        step=10
    )

else:

    record_limit = st.sidebar.slider(
        "Number of records to analyse",
        min_value=1,
        max_value=len(data),
        value=len(data)
    )

st.sidebar.caption(
    "Each selected network record is analysed by the trained model."
)

# =====================================================
# PREDICTION FUNCTION
# =====================================================

def predict_record(row):

    features = row.drop(
        labels=["label", "id", "attack_cat"],
        errors="ignore"
    )

    features = pd.DataFrame([features])

    prediction_value = int(model.predict(features)[0])

    probabilities = model.predict_proba(features)[0]

    classes = list(model.classes_)

    if 1 in classes:
        attack_probability = float(
            probabilities[classes.index(1)]
        )
    else:
        attack_probability = 0.0

    prediction = (
        "Attack" if prediction_value == 1 else "Benign"
    )

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


# =====================================================
# NETWORK MONITORING
# =====================================================

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

            actual_class = (
                "Attack" if int(row["label"]) == 1 else "Benign"
            )

            predicted_class = prediction["prediction"]

            # Determine detection outcome
            if predicted_class == "Attack" and actual_class == "Attack":
                outcome = "True Positive"
            elif predicted_class == "Attack" and actual_class == "Benign":
                outcome = "False Positive"
            elif predicted_class == "Benign" and actual_class == "Benign":
                outcome = "True Negative"
            else:
                outcome = "False Negative"

            results.append({
                "Record": index + 1,
                "Prediction": predicted_class,
                "Risk Level": prediction["risk_level"],
                "Attack Probability": prediction["attack_probability"],
                "Actual Class": actual_class,
                "Detection Outcome": outcome
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

# =====================================================
# DISPLAY RESULTS
# =====================================================

if "results" in st.session_state:

    results_df = pd.DataFrame(
        st.session_state["results"]
    )

    if not results_df.empty:

        # -------------------------------------------------
        # SECURITY OVERVIEW
        # -------------------------------------------------

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

        false_positives = (
            results_df["Detection Outcome"] == "False Positive"
        ).sum()

        false_negatives = (
            results_df["Detection Outcome"] == "False Negative"
        ).sum()

        st.divider()
        st.subheader("Security Overview")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Records Analysed", total)
        c2.metric("Attacks Detected", int(attacks))
        c3.metric("Benign Records", int(benign))
        c4.metric("High-Risk Records", int(high_risk))

        # -------------------------------------------------
        # DETECTION QUALITY
        # -------------------------------------------------

        st.subheader("Detection Quality")

        q1, q2 = st.columns(2)

        q1.metric(
            "False Positives",
            int(false_positives),
            help="Benign records incorrectly classified as attacks."
        )

        q2.metric(
            "False Negatives",
            int(false_negatives),
            help="Actual attacks incorrectly classified as benign."
        )

        if false_positives > 0:
            st.warning(
                f"{int(false_positives)} benign record(s) were incorrectly "
                "flagged as attacks. These are false positives."
            )

        if false_negatives > 0:
            st.warning(
                f"{int(false_negatives)} attack record(s) were missed. "
                "These are false negatives."
            )

        # -------------------------------------------------
        # CHARTS
        # -------------------------------------------------

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

        # -------------------------------------------------
        # IMPROVED THREAT ALERTS
        # -------------------------------------------------

        st.divider()
        st.subheader("Threat Alerts")

        st.caption(
            "Records classified as attacks, with risk level and prediction confidence."
        )

        attack_records = results_df[
            results_df["Prediction"] == "Attack"
        ]

        if not attack_records.empty:

            st.write(
                f"**{len(attack_records)} suspicious record(s) detected.**"
            )

            for _, alert in attack_records.head(10).iterrows():

                with st.container(border=True):

                    alert_col1, alert_col2, alert_col3 = st.columns(
                        [1.2, 1, 1.5]
                    )

                    with alert_col1:
                        st.markdown(
                            f"**Record {int(alert['Record'])}**"
                        )
                        st.caption("Threat detected")

                    with alert_col2:

                        if alert["Risk Level"] == "HIGH":
                            st.error("HIGH RISK")
                        elif alert["Risk Level"] == "MEDIUM":
                            st.warning("MEDIUM RISK")
                        else:
                            st.info("LOW RISK")

                    with alert_col3:
                        st.metric(
                            "Attack Probability",
                            f"{alert['Attack Probability']:.2%}"
                        )

                    if alert["Detection Outcome"] == "False Positive":
                        st.warning(
                            "False Positive: Actual class is Benign, "
                            "but the model predicted Attack."
                        )

                    else:
                        st.caption(
                            f"Actual class: {alert['Actual Class']} "
                            "| Outcome: {alert['Detection Outcome']}"
                        )

            if len(attack_records) > 10:
                st.caption(
                    f"Showing 10 of {len(attack_records)} attack alerts. "
                    "See the results table for all records."
                )

        else:

            st.success("No attacks detected in these records.")

        # -------------------------------------------------
        # IMPROVED RESULTS TABLE
        # -------------------------------------------------

        st.divider()
        st.subheader("Network Analysis Results")

        st.caption(
            "Compare model predictions with actual dataset labels. "
            "False positives and false negatives are identified separately."
        )

        display_df = results_df.copy()

        display_df["Attack Probability"] = (
            display_df["Attack Probability"].map(
                lambda value: f"{value:.2%}"
            )
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        # -------------------------------------------------
        # DOWNLOAD REPORT
        # -------------------------------------------------

        st.divider()
        st.subheader("Download Report")

        csv_data = results_df.to_csv(index=False)

        st.download_button(
            label="⬇ Download Analysis Report",
            data=csv_data,
            file_name="netguard_threat_report.csv",
            mime="text/csv"
        )

# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "NetGuard AI | Network Intrusion Detection and Threat Monitoring | "
    "Powered by Machine Learning"
)
