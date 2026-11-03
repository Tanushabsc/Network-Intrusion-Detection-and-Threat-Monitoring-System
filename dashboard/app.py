
import os
import requests
import joblib
import pandas as pd
import streamlit as st

# =========================================================
# CONFIGURATION
# =========================================================

MODEL_URL = (
    "https://github.com/Tanushabsc/"
    "Network-Intrusion-Detection-and-Threat-Monitoring-System/"
    "releases/download/v1.0/intrusion_model.joblib"
)

MODEL_PATH = "models/intrusion_model.joblib"
DATA_PATH = "data/processed/test.csv"

MODEL_METRICS = {
    "Accuracy": 0.8921,
    "Precision": 0.8510,
    "Recall": 0.9746,
    "F1-score": 0.9086,
}

st.set_page_config(
    page_title="NetGuard AI | Security Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# CUSTOM UI DESIGN
# =========================================================

st.markdown("""
<style>
.stApp {
    background: radial-gradient(
        ellipse at 8% 0%,
        rgba(14,165,233,.13),
        transparent 34%
    ),
    linear-gradient(
        180deg,
        #08111f 0%,
        #0b1422 58%,
        #08111f 100%
    );
    color: #edf5ff;
}

[data-testid="stSidebar"] {
    background: #0b1728;
    border-right: 1px solid rgba(148,163,184,.18);
}

[data-testid="stSidebar"] * {
    color: #e5eefb;
}

.block-container {
    max-width: 1480px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: #f4f8ff !important;
    letter-spacing: -.02em;
}

p, li, label {
    color: #d3deec;
}

.ng-eyebrow {
    color: #67d5ff;
    font-size: .75rem;
    font-weight: 800;
    letter-spacing: .16em;
}

.ng-title {
    color: #f8fbff;
    font-size: 2.55rem;
    font-weight: 850;
    line-height: 1.12;
}

.ng-subtitle {
    color: #a9bad0;
    font-size: 1rem;
    line-height: 1.65;
    max-width: 900px;
}

.ng-panel {
    background: linear-gradient(
        145deg,
        rgba(20,36,58,.97),
        rgba(13,27,45,.97)
    );
    border: 1px solid rgba(148,163,184,.18);
    border-radius: 16px;
    padding: 18px 20px;
    min-height: 112px;
}

.ng-kpi-label {
    color: #9fb1c8;
    font-size: .82rem;
    font-weight: 650;
}

.ng-kpi-value {
    color: #f5f9ff;
    font-size: 1.9rem;
    font-weight: 850;
    margin-top: 5px;
}

.ng-kpi-note {
    color: #8fa4bd;
    font-size: .76rem;
    margin-top: 2px;
}

.ng-info {
    border: 1px solid rgba(56,189,248,.28);
    border-left: 4px solid #38bdf8;
    border-radius: 10px;
    padding: 13px 16px;
    background: rgba(14,165,233,.08);
    color: #d9efff;
    margin: 10px 0 18px;
}

.ng-good {
    color: #6ee7b7;
    font-weight: 750;
}

.ng-muted {
    color: #9aabc1;
    font-size: .83rem;
}

div[data-testid="stMetric"] {
    background: rgba(20,36,58,.82);
    border: 1px solid rgba(148,163,184,.18);
    padding: 14px 16px;
    border-radius: 14px;
}

div[data-testid="stMetricLabel"] {
    color: #a9bad0;
}

div[data-testid="stMetricValue"] {
    color: #f5f9ff;
}

.stButton>button,
.stDownloadButton>button {
    border-radius: 10px;
    font-weight: 750;
    min-height: 2.7rem;
}

.stButton>button[kind="primary"] {
    background: linear-gradient(90deg,#0284c7,#0ea5e9);
    border: 0;
    color: white;
}

hr {
    border-color: rgba(148,163,184,.16);
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

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


with st.spinner("Loading the trained Random Forest model..."):

    try:
        model = load_model()

    except Exception as error:
        st.error(f"Could not load the trained model: {error}")
        st.stop()


# =========================================================
# HEADER
# =========================================================

header_left, header_right = st.columns([5, 1.4])

with header_left:

    st.markdown(
        '<div class="ng-eyebrow">'
        ''
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="ng-title">🛡️ NetGuard AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="ng-subtitle">'
        'Network Intrusion Detection & Threat Monitoring. '
        'Analyse network-flow records, identify suspicious activity '
        'and understand the results through clear security insights.'
        '</div>',
        unsafe_allow_html=True
    )

with header_right:

    st.markdown(
        '<div class="ng-panel" '
        'style="text-align:center;min-height:0;margin-top:8px;">'
        '<div class="ng-muted">MODEL STATUS</div>'
        '<div class="ng-good" '
        'style="font-size:1.15rem;margin-top:7px;">'
        '● ONLINE</div>'
        '<div class="ng-muted">Random Forest</div>'
        '</div>',
        unsafe_allow_html=True
    )


st.markdown(
    '<div class="ng-info">'
    '<b>What does NetGuard AI do?</b><br>'
    'NetGuard AI uses a trained Machine Learning model to examine '
    'network-flow records and classify them as '
    '<b>Benign</b> (normal activity) or '
    '<b>Attack</b> (potentially malicious activity). '
    'It then displays attack probability, risk level and a report '
    'to help users review the results.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR AND DATASET UPLOAD
# =========================================================

st.sidebar.markdown("## 🛡️ NetGuard AI")
st.sidebar.caption("Network security analysis workspace")
st.sidebar.divider()

st.sidebar.markdown("### 1 · Select your dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload network traffic CSV",
    type=["csv"],
    help="Use the UNSW-NB15 testing-set CSV. "
         "It must include a binary label column."
)


@st.cache_data
def load_local_data():
    return pd.read_csv(DATA_PATH)


data = None

if uploaded_file is not None:

    try:
        data = pd.read_csv(uploaded_file)
        st.sidebar.success("Uploaded CSV loaded.")

    except Exception as error:
        st.error(f"Could not read the uploaded CSV: {error}")
        st.stop()

elif os.path.exists(DATA_PATH):

    try:
        data = load_local_data()
        st.sidebar.success("Local test dataset loaded.")

    except Exception as error:
        st.error(f"Could not load the local dataset: {error}")
        st.stop()

else:

    st.info(
        "To begin, upload the UNSW-NB15 testing-set CSV "
        "using the left sidebar. The online dashboard cannot "
        "automatically access files on your computer."
    )

    st.stop()


if data.empty:
    st.error("The uploaded dataset is empty.")
    st.stop()

if "label" not in data.columns:
    st.error("Dataset must contain a 'label' column.")
    st.stop()

if not data["label"].isin([0, 1]).all():
    st.error(
        "The label column must contain only "
        "0 (Benign) and 1 (Attack)."
    )
    st.stop()


# =========================================================
# SIDEBAR ANALYSIS SETTINGS
# =========================================================

st.sidebar.divider()
st.sidebar.markdown("### 2 · Analysis settings")

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
    "The first selected records will be analysed."
)

st.sidebar.divider()
st.sidebar.markdown("### Dataset summary")

st.sidebar.metric(
    "Available records",
    f"{len(data):,}"
)

st.sidebar.metric(
    "Number of columns",
    len(data.columns)
)

st.sidebar.caption("Dataset: UNSW-NB15")


# =========================================================
# MODEL PERFORMANCE
# =========================================================

st.markdown("## Model performance")

st.caption(
    "Previously measured evaluation results on the "
    "complete UNSW-NB15 test set."
)

metric_notes = {
    "Accuracy": "Overall correct predictions",
    "Precision": "Correctness of attack predictions",
    "Recall": "Actual attacks successfully detected",
    "F1-score": "Balance between precision and recall",
}

metric_cols = st.columns(4)

for col, (metric_name, metric_value) in zip(
    metric_cols,
    MODEL_METRICS.items()
):

    with col:

        st.metric(
            metric_name,
            f"{metric_value:.2%}"
        )

        st.caption(metric_notes[metric_name])


with st.expander("What do these metrics mean?"):

    st.markdown("""
    - **Accuracy:** percentage of all records classified correctly.
    - **Precision:** among records predicted as attacks, how many were actually attacks.
    - **Recall:** among all actual attacks, how many the model successfully identified.
    - **F1-score:** a combined measure of precision and recall.

    These values are the model's previously measured results on the full test set.
    They are not recalculated from the smaller sample selected for this dashboard run.
    """)


st.divider()


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_record(row):

    features = row.drop(
        labels=["label", "id", "attack_cat"],
        errors="ignore"
    )

    features = pd.DataFrame([features])

    prediction_value = int(
        model.predict(features)[0]
    )

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
        "attack_probability": attack_probability,
    }


# =========================================================
# START NETWORK ANALYSIS
# =========================================================

st.markdown("## Network traffic analysis")

st.write(
    "Select the number of records in the sidebar and run "
    "the analysis. The model will examine each selected "
    "network-flow record."
)

st.markdown(
    '<div class="ng-muted">'
    'Analysis flow: CSV record → Random Forest model → '
    'prediction → risk level → security report'
    '</div>',
    unsafe_allow_html=True
)

start_col, note_col = st.columns([1.5, 4])

with start_col:

    run_analysis = st.button(
        "▶  Start network analysis",
        type="primary",
        use_container_width=True
    )

with note_col:

    st.caption(
        f"Ready to analyse {record_limit:,} record(s)."
    )


if run_analysis:

    results = []

    selected_data = data.head(record_limit)

    progress = st.progress(
        0,
        text="Preparing records..."
    )

    failed = False

    for index, (_, row) in enumerate(
        selected_data.iterrows()
    ):

        try:

            prediction = predict_record(row)

            actual_class = (
                "Attack" if int(row["label"]) == 1
                else "Benign"
            )

            predicted_class = prediction["prediction"]

            if (
                predicted_class == "Attack"
                and actual_class == "Attack"
            ):

                outcome = "True Positive"

            elif (
                predicted_class == "Attack"
                and actual_class == "Benign"
            ):

                outcome = "False Positive"

            elif (
                predicted_class == "Benign"
                and actual_class == "Benign"
            ):

                outcome = "True Negative"

            else:

                outcome = "False Negative"


            results.append({

                "Record": index + 1,

                "Prediction": predicted_class,

                "Risk Level": prediction["risk_level"],

                "Attack Probability": prediction[
                    "attack_probability"
                ],

                "Actual Class": actual_class,

                "Detection Outcome": outcome

            })


            progress.progress(
                (index + 1) / len(selected_data),
                text=(
                    f"Analysing record {index + 1} "
                    f"of {len(selected_data)}..."
                )
            )


        except Exception as error:

            st.error(
                f"Error processing record {index + 1}: {error}. "
                "Check that the uploaded CSV contains the same "
                "feature columns used when the model was trained."
            )

            failed = True
            break


    if results:

        st.session_state["results"] = results

        st.session_state["results_source"] = (
            uploaded_file.name
            if uploaded_file
            else "Local test dataset"
        )

        st.success(
            f"Analysis completed: {len(results)} record(s) processed."
        )

    else:

        st.session_state.pop("results", None)
        st.session_state.pop("results_source", None)

    if failed:

        st.warning(
            "The report contains only records processed "
            "before the error."
        )


# =========================================================
# DISPLAY ANALYSIS RESULTS
# =========================================================

if (
    "results" in st.session_state
    and st.session_state["results"]
):

    results_df = pd.DataFrame(
        st.session_state["results"]
    )

    total = len(results_df)

    attacks = int(
        (results_df["Prediction"] == "Attack").sum()
    )

    benign = int(
        (results_df["Prediction"] == "Benign").sum()
    )

    high_risk = int(
        (results_df["Risk Level"] == "HIGH").sum()
    )

    false_positives = int(
        (
            results_df["Detection Outcome"]
            == "False Positive"
        ).sum()
    )

    false_negatives = int(
        (
            results_df["Detection Outcome"]
            == "False Negative"
        ).sum()
    )

    correct = int(
        (
            results_df["Prediction"]
            == results_df["Actual Class"]
        ).sum()
    )


    # -----------------------------------------------------
    # SECURITY OVERVIEW
    # -----------------------------------------------------

    st.divider()

    st.markdown("## Security overview")

    st.caption(
        f"Results for: "
        f"{st.session_state.get('results_source', 'selected dataset')}"
    )

    kpi_cols = st.columns(4)

    kpis = [

        (
            "Records analysed",
            f"{total:,}",
            "Records processed"
        ),

        (
            "Attack predictions",
            f"{attacks:,}",
            "Classified as potential attacks"
        ),

        (
            "Benign predictions",
            f"{benign:,}",
            "Classified as normal activity"
        ),

        (
            "High-risk records",
            f"{high_risk:,}",
            "Attack probability ≥ 70%"
        )

    ]

    for col, (label, value, note) in zip(
        kpi_cols,
        kpis
    ):

        with col:

            st.markdown(
                f'<div class="ng-panel">'
                f'<div class="ng-kpi-label">{label}</div>'
                f'<div class="ng-kpi-value">{value}</div>'
                f'<div class="ng-kpi-note">{note}</div>'
                f'</div>',
                unsafe_allow_html=True
            )


    # -----------------------------------------------------
    # DETECTION QUALITY
    # -----------------------------------------------------

    st.markdown("### Detection quality")

    q1, q2, q3 = st.columns(3)

    q1.metric(
        "False positives",
        false_positives,
        help="Benign records incorrectly predicted as attacks."
    )

    q2.metric(
        "False negatives",
        false_negatives,
        help="Actual attacks incorrectly predicted as benign."
    )

    q3.metric(
        "Correct predictions",
        correct
    )


    if false_positives:

        st.warning(
            f"{false_positives} benign record(s) were "
            "incorrectly flagged as attacks. These are "
            "false positives (potential false alarms)."
        )


    if false_negatives:

        st.warning(
            f"{false_negatives} actual attack record(s) "
            "were classified as benign. These are "
            "false negatives (missed attacks)."
        )


    # -----------------------------------------------------
    # RESULTS TABS
    # -----------------------------------------------------

    overview_tab, alerts_tab, records_tab, guide_tab = st.tabs([

        "📊 Overview",

        "🚨 Threat alerts",

        "📋 Record details",

        "ℹ️ Help & definitions"

    ])


    # =====================================================
    # OVERVIEW TAB
    # =====================================================

    with overview_tab:

        chart_left, chart_right = st.columns(2)

        with chart_left:

            st.markdown("#### Prediction distribution")

            distribution = (
                results_df["Prediction"]
                .value_counts()
                .reindex(
                    ["Attack", "Benign"],
                    fill_value=0
                )
            )

            st.bar_chart(distribution)

            st.caption(
                "How many records were classified "
                "as Attack or Benign."
            )


        with chart_right:

            st.markdown("#### Attack probability by record")

            st.line_chart(
                results_df.set_index("Record")[
                    "Attack Probability"
                ]
            )

            st.caption(
                "Higher values indicate a higher "
                "model-estimated probability of an attack."
            )


        st.markdown("#### Summary")

        st.write(
            f"The model classified **{attacks} of {total}** "
            f"records as attacks and **{benign}** as benign. "
            f"**{high_risk}** record(s) reached the dashboard's "
            "high-risk probability threshold."
        )


    # =====================================================
    # THREAT ALERTS TAB
    # =====================================================

    with alerts_tab:

        st.markdown("#### Records predicted as attacks")

        st.caption(
            "Up to 20 alerts are shown here. "
            "Open Record details for the complete table."
        )

        attack_records = results_df[
            results_df["Prediction"] == "Attack"
        ]


        if attack_records.empty:

            st.success(
                "No records were predicted as attacks in this analysis."
            )

        else:

            st.write(
                f"**{len(attack_records)} attack prediction(s) found.**"
            )

            for _, alert in attack_records.head(20).iterrows():

                with st.container(border=True):

                    a, b, c = st.columns([1.2, 1.2, 1.5])

                    with a:

                        st.markdown(
                            f"**Record {int(alert['Record'])}**"
                        )

                        st.caption(
                            f"Actual class: {alert['Actual Class']}"
                        )


                    with b:

                        if alert["Risk Level"] == "HIGH":

                            st.error("HIGH RISK")

                        elif alert["Risk Level"] == "MEDIUM":

                            st.warning("MEDIUM RISK")

                        else:

                            st.info("LOW RISK")


                    with c:

                        st.metric(
                            "Attack probability",
                            f"{alert['Attack Probability']:.2%}"
                        )


                    if (
                        alert["Detection Outcome"]
                        == "False Positive"
                    ):

                        st.warning(
                            "False positive: the dataset's "
                            "actual label is Benign."
                        )

                    else:

                        st.caption(
                            f"Detection outcome: "
                            f"{alert['Detection Outcome']}"
                        )


            if len(attack_records) > 20:

                st.info(
                    f"{len(attack_records) - 20} more attack "
                    "prediction(s) are listed in the Record details tab."
                )


    # =====================================================
    # RECORD DETAILS TAB
    # =====================================================

    with records_tab:

        st.markdown("#### Complete analysis results")

        st.caption(
            "Prediction is the model output. Actual class comes "
            "from the dataset label. Detection outcome compares the two."
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


        csv_data = results_df.to_csv(
            index=False
        ).encode("utf-8")


        st.download_button(
            "⬇ Download analysis report (CSV)",
            data=csv_data,
            file_name="netguard_threat_report.csv",
            mime="text/csv"
        )


    # =====================================================
    # HELP AND DEFINITIONS TAB
    # =====================================================

    with guide_tab:

        st.markdown("#### Understand the results")

        st.markdown("""
        - **Benign:** the model classified the record as normal network activity.
        - **Attack:** the model classified the record as potentially malicious.
        - **Attack probability:** the model-estimated probability that the record belongs to the Attack class.
        - **Risk level:** a dashboard category based on the attack probability.
        - **True Positive:** an actual attack correctly predicted as Attack.
        - **False Positive:** a benign record incorrectly predicted as Attack.
        - **True Negative:** a benign record correctly predicted as Benign.
        - **False Negative:** an actual attack incorrectly predicted as Benign.
        """)

        st.markdown("#### Risk level guide")

        st.write(
            "**High:** 70% or higher attack probability."
        )

        st.write(
            "**Medium:** 40% to below 70%."
        )

        st.write(
            "**Low:** below 40%."
        )

        st.caption(
            "These probability bands are dashboard-defined "
            "thresholds, not official cybersecurity severity ratings."
        )


# =========================================================
# ABOUT THE PROJECT
# =========================================================

st.divider()

st.markdown("## About this project")

about_left, about_right = st.columns(2)


with about_left:

    st.markdown("### How NetGuard AI works")

    st.write(
        "1. A user provides network-flow records in a CSV file. "
        "2. The Random Forest model analyses the selected records. "
        "3. Each record receives a Benign or Attack prediction "
        "and an attack probability. "
        "4. The dashboard organises the results into summaries, "
        "alerts and a downloadable report."
    )


with about_right:

    st.markdown("### Technologies used")

    st.write(
        "**Machine Learning:** Random Forest  \n"
        "**Dataset:** UNSW-NB15  \n"
        "**API:** FastAPI  \n"
        "**Dashboard:** Streamlit  \n"
        "**MLOps tools:** DVC, MLflow, Docker, Kubernetes "
        "and GitHub Actions"
    )


st.markdown(
    '<div class="ng-info">'
    '<b>Important project scope</b><br>'
    'This version demonstrates monitoring by analysing uploaded '
    'or locally available UNSW-NB15 network-flow records. '
    'It does <b>not</b> capture live packets from a network or '
    'provide continuous live-network protection. The risk bands '
    'are dashboard-defined thresholds, and model predictions '
    'should be reviewed in context.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "NetGuard AI • Machine Learning-based Network "
    "Intrusion Detection and Threat Monitoring"
)
