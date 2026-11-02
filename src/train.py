
import os
import json
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# 1. PROJECT PATHS
# ==========================================

TRAIN_FILE = "data/processed/train.csv"
TEST_FILE = "data/processed/test.csv"

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ==========================================
# 2. LOAD DATASET
# ==========================================

print("\nLoading UNSW-NB15 dataset...")

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print("Training records:", len(train_df))
print("Testing records:", len(test_df))

# ==========================================
# 3. SEPARATE FEATURES AND TARGET
# ==========================================

X_train = train_df.drop(columns=["label"])
y_train = train_df["label"]

X_test = test_df.drop(columns=["label"])
y_test = test_df["label"]

# ==========================================
# 4. IDENTIFY COLUMN TYPES
# ==========================================

categorical_columns = X_train.select_dtypes(
    include=["object", "category", "str"]
).columns.tolist()

numerical_columns = X_train.select_dtypes(
    include=["number", "bool"]
).columns.tolist()

print("\nNumerical features:", len(numerical_columns))
print("Categorical features:", len(categorical_columns))

# ==========================================
# 5. NUMERICAL PREPROCESSING
# ==========================================

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# ==========================================
# 6. CATEGORICAL PREPROCESSING
# ==========================================

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# ==========================================
# 7. COMBINE PREPROCESSING
# ==========================================

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_columns),
    ("cat", categorical_pipeline, categorical_columns)
])

# ==========================================
# 8. BUILD MACHINE LEARNING MODEL
# ==========================================

model = Pipeline([
    ("preprocessor", preprocessor),

    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    ))
])

# ==========================================
# 9. CONFIGURE MLFLOW
# ==========================================

# Use a local tracking directory.
# This avoids the Windows username path issue.

# Configure MLflow using SQLite database
mlflow.set_tracking_uri("sqlite:///mlflow.db")

mlflow.set_experiment(
    "NetGuard-AI-Intrusion-Detection"
)

# ==========================================
# 10. TRAIN AND TRACK EXPERIMENT
# ==========================================

with mlflow.start_run() as run:

    print("\n================================")
    print("TRAINING NETGUARD AI MODEL")
    print("================================")

    model.fit(X_train, y_train)

    print("\nModel training completed!")

    # ======================================
    # 11. MODEL PREDICTIONS
    # ======================================

    predictions = model.predict(X_test)

    # ======================================
    # 12. MODEL EVALUATION
    # ======================================

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    print("\n================================")
    print("MODEL EVALUATION RESULTS")
    print("================================")

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))

    # ======================================
    # 13. CLASSIFICATION REPORT
    # ======================================

    print("\nCLASSIFICATION REPORT")

    print(classification_report(
        y_test,
        predictions,
        target_names=["Benign", "Attack"],
        zero_division=0
    ))

    # ======================================
    # 14. CONFUSION MATRIX
    # ======================================

    cm = confusion_matrix(y_test, predictions)

    print("\nCONFUSION MATRIX")
    print(cm)

    # ======================================
    # 15. SAVE MODEL
    # ======================================

    model_path = os.path.join(
        MODEL_DIR,
        "intrusion_model.joblib"
    )

    joblib.dump(model, model_path)

    print("\nModel saved successfully!")

    # ======================================
    # 16. SAVE EVALUATION METRICS
    # ======================================

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist()
    }

    with open("metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    # ======================================
    # 17. MLFLOW EXPERIMENT TRACKING
    # ======================================

    mlflow.log_param(
        "algorithm",
        "RandomForestClassifier"
    )

    mlflow.log_param(
        "n_estimators",
        100
    )

    mlflow.log_param(
        "random_state",
        42
    )

    mlflow.log_param(
        "training_records",
        len(X_train)
    )

    mlflow.log_param(
        "testing_records",
        len(X_test)
    )

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    
    mlflow.sklearn.log_model(
        model,
        name="intrusion_detection_model",
        skops_trusted_types=[
            "numpy.dtype",
            "sklearn.tree._tree.Tree"
        ]
    )


    print("\n================================")
    print("MLFLOW TRACKING COMPLETED")
    print("================================")

    print("Experiment: NetGuard-AI-Intrusion-Detection")
    print("Run ID:", run.info.run_id)

    print("\nModel location:", model_path)

    print("\nALL OPERATIONS COMPLETED SUCCESSFULLY!")
