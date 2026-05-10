
import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_model_and_metadata():
    models_dir = "models"
    model_files = [f for f in os.listdir(models_dir) if f.startswith("best_model") and f.endswith(".pkl")]
    if not model_files:
        raise FileNotFoundError("No trained model found in 'models' directory")
    model_path = os.path.join(models_dir, model_files[0])
    model = joblib.load(model_path)

    metadata_path = os.path.join(models_dir, "model_metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    scaler_path = os.path.join(models_dir, "scaler.pkl")
    scaler = joblib.load(scaler_path)

    print("Model, scaler, metadata loaded successfully")
    print("Model file:", model_path)
    print("Scaler expected features:", list(scaler.feature_names_in_))
    print("Selected features:", metadata.get("selected_features"))

    return model, metadata, scaler

def load_csv_with_labels(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    required_cols = ["Time", "Height", "Resultant_acceleration", "Resultant_velocity", "AGC"]
    for col in required_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "classlabel" in df.columns:
        df["classlabel"] = pd.to_numeric(df["classlabel"], errors="coerce").astype("Int64")
        return df, df["classlabel"].values

    return df, None

def extract_window_features(window_df):
    feature_dict = {}
    columns = ["Height", "Resultant_acceleration", "Resultant_velocity", "AGC"]

    for col in columns:
        values = pd.to_numeric(window_df[col], errors="coerce").astype(float).values
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

        feature_dict[f"{col}_mean"] = float(np.mean(values))
        feature_dict[f"{col}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        feature_dict[f"{col}_min"] = float(np.min(values))
        feature_dict[f"{col}_max"] = float(np.max(values))
        feature_dict[f"{col}_median"] = float(np.median(values))
        feature_dict[f"{col}_range"] = float(np.max(values) - np.min(values))

        if len(values) > 1:
            rate = np.diff(values)
            feature_dict[f"{col}_rate_mean"] = float(np.mean(rate))
            feature_dict[f"{col}_rate_std"] = float(np.std(rate, ddof=1)) if len(rate) > 1 else 0.0
        else:
            feature_dict[f"{col}_rate_mean"] = 0.0
            feature_dict[f"{col}_rate_std"] = 0.0

    return pd.DataFrame([feature_dict])

def preprocess_data(features_df, scaler, metadata):
    try:
        scaler_features = list(scaler.feature_names_in_)

        for feat in scaler_features:
            if feat not in features_df.columns:
                features_df[feat] = 0.0

        X = features_df[scaler_features].copy()
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        X_scaled = scaler.transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=scaler_features)

        selected_features = metadata.get("selected_features", scaler_features)

        for feat in selected_features:
            if feat not in X_scaled_df.columns:
                X_scaled_df[feat] = 0.0

        X_final = X_scaled_df[selected_features]
        return X_final

    except Exception as e:
        print(f"Preprocessing error: {e}")
        print("Available extracted features:", list(features_df.columns))
        try:
            print("Scaler expected features:", list(scaler.feature_names_in_))
            print("Metadata selected features:", metadata.get("selected_features"))
        except Exception:
            pass
        return None

def predict_window(window_df, model, scaler, metadata, actual_class=None):
    try:
        features_df = extract_window_features(window_df)
        X_processed = preprocess_data(features_df, scaler, metadata)

        if X_processed is None:
            return None, None, None

        prediction = int(model.predict(X_processed)[0])

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_processed)[0]
            class_names = model.classes_
            top_3_indices = np.argsort(probabilities)[-3:][::-1]
            top_3_predictions = [
                {"class": int(class_names[idx]), "probability": float(probabilities[idx])}
                for idx in top_3_indices
            ]
        else:
            top_3_predictions = [{"class": prediction, "probability": 1.0}]

        accuracy = None
        if actual_class is not None and actual_class != "N/A":
            accuracy = 100.0 if prediction == int(actual_class) else 0.0

        return prediction, top_3_predictions, accuracy

    except Exception as e:
        print(f"Prediction error: {e}")
        return None, None, None

def compute_metrics(actual_labels, predicted_labels):
    accuracy = accuracy_score(actual_labels, predicted_labels)
    precision = precision_score(actual_labels, predicted_labels, average="weighted", zero_division=0)
    recall = recall_score(actual_labels, predicted_labels, average="weighted", zero_division=0)
    f1 = f1_score(actual_labels, predicted_labels, average="weighted", zero_division=0)
    return {
        "Accuracy": accuracy * 100,
        "Precision": precision * 100,
        "Recall": recall * 100,
        "F1-Score": f1 * 100
    }

def generate_confusion_matrix(actual_labels, predicted_labels, classes):
    return confusion_matrix(actual_labels, predicted_labels, labels=classes)

def plot_metrics_bar_graph(metrics):
    fig, ax = plt.subplots(figsize=(10, 6))
    names = list(metrics.keys())
    values = list(metrics.values())
    bars = ax.bar(names, values)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Model Performance Metrics")
    ax.set_ylim(0, 105)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.2f}%", ha="center")
    plt.tight_layout()
    return fig

def plot_confusion_matrix(cm, classes):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    return fig
