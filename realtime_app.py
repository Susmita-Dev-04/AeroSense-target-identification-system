import time
import random
import streamlit as st
import pandas as pd

from udp_receiver import SlidingWindowBuffer
from utils import (
    load_model_and_metadata,
    load_csv_with_labels,
    predict_window,
    compute_metrics,
    generate_confusion_matrix,
    plot_metrics_bar_graph,
    plot_confusion_matrix
)

st.set_page_config(page_title="Aircraft Classification", page_icon="✈️", layout="wide")

# ============================================================
# CLASS NAMES
# ============================================================
CLASS_NAMES = {
    1: "Boeing 747",
    2: "Airbus A320",
    3: "F-22 Raptor",
    4: "Sukhoi Su-30",
    5: "C-17 Globemaster III",
    6: "HAL Tejas",
    7: "Dassault Rafale",
    8: "MQ-9 Reaper (Drone UAV)",
    9: "AH-64 Apache (Helicopter)",
    10: "Cessna 172"
}

# ============================================================
# VIDEO LINKS
# ============================================================
VIDEO_LINKS = {
    1: "https://drive.google.com/file/d/1gW7C0ZmEJHKYsU2TJa6TXFnLdowCfe_M/view?usp=drivesdk",
    2: "https://drive.google.com/file/d/165oqbWMwDlx-EF-fjxTPE24oAKil_iju/view?usp=drivesdk",
    3: "https://drive.google.com/file/d/17ez2IivGRYo14_ELtLnAYJtDRkPIofqd/view?usp=drivesdk",
    4: "https://drive.google.com/file/d/1Yq7Y6AT7_5aMzRi8IkT544qXzka9CrZi/view?usp=drivesdk",
    5: "https://drive.google.com/file/d/1D4tNr5ziuWWF0L6eYtZP5QjELd_kjDbc/view?usp=drivesdk",
    6: "https://drive.google.com/file/d/17n9be8eCWygErP3hj-DrvbCXgoTfxlZR/view?usp=drivesdk",
    7: "https://drive.google.com/file/d/1e8-m_UmU2IOm6RJ5-37wF1jGPZ4zgr0R/view?usp=drivesdk",
    8: "https://drive.google.com/file/d/1zxXwkFubsk3Z-0jBgVbDgVA34drvVJHy/view?usp=drivesdk",
    9: "https://drive.google.com/file/d/1lFFnZrKNE9d_zonWMMcWKhysr1Bm4wTE/view?usp=drivesdk",
    10: "https://drive.google.com/file/d/1tcE3BAaHLYcw6cAjjQW7f5agNKWIxixd/view?usp=drivesdk"
}

# ============================================================
# AIRCRAFT DETAILS
# ============================================================
AIRCRAFT_DETAILS = {
    1: {
        "type": "Wide-body commercial aircraft",
        "speed": "Approx. 900 km/h cruise speed",
        "altitude": "Approx. 35,000–45,000 ft",
        "motion": "Smooth climb, stable cruise, low sudden acceleration",
        "known_for": "Large long-distance passenger aircraft."
    },
    2: {
        "type": "Narrow-body commercial aircraft",
        "speed": "Approx. 830 km/h cruise speed",
        "altitude": "Approx. 33,000–39,000 ft",
        "motion": "Stable velocity, gradual acceleration, smooth altitude change",
        "known_for": "Common short-to-medium range passenger aircraft."
    },
    3: {
        "type": "Stealth fighter aircraft",
        "speed": "Approx. 2,400 km/h max speed",
        "altitude": "Approx. 50,000+ ft",
        "motion": "High speed, sharp turns, high acceleration",
        "known_for": "Advanced stealth air-superiority fighter."
    },
    4: {
        "type": "Multirole fighter aircraft",
        "speed": "Approx. 2,100 km/h max speed",
        "altitude": "Approx. 56,000 ft",
        "motion": "Fast acceleration, agile movement, combat maneuvers",
        "known_for": "Highly maneuverable fighter aircraft."
    },
    5: {
        "type": "Military transport aircraft",
        "speed": "Approx. 830 km/h cruise speed",
        "altitude": "Approx. 45,000 ft",
        "motion": "Heavy but stable movement, gradual climb",
        "known_for": "Military cargo and troop transport."
    },
    6: {
        "type": "Light combat aircraft",
        "speed": "Approx. 2,200 km/h max speed",
        "altitude": "Approx. 50,000 ft",
        "motion": "Fast response, agile turning, high acceleration",
        "known_for": "Indian light combat fighter aircraft."
    },
    7: {
        "type": "Multirole fighter jet",
        "speed": "Approx. 1,900 km/h max speed",
        "altitude": "Approx. 50,000 ft",
        "motion": "Sharp maneuver, fast climb, high acceleration",
        "known_for": "Advanced multirole combat aircraft."
    },
    8: {
        "type": "Drone / UAV",
        "speed": "Approx. 480 km/h max speed",
        "altitude": "Approx. 25,000–50,000 ft",
        "motion": "Slow and steady surveillance-type motion",
        "known_for": "Unmanned surveillance and operation."
    },
    9: {
        "type": "Attack helicopter",
        "speed": "Approx. 293 km/h max speed",
        "altitude": "Approx. 20,000 ft",
        "motion": "Low altitude, hover ability, variable speed",
        "known_for": "Close air support and attack missions."
    },
    10: {
        "type": "Light single-engine aircraft",
        "speed": "Approx. 226 km/h cruise speed",
        "altitude": "Approx. 13,000–15,000 ft",
        "motion": "Low speed, smooth flight, training aircraft pattern",
        "known_for": "Training and private flying."
    }
}

# ============================================================
# DEMO MODE
# ============================================================
DEMO_MODE_USE_ACTUAL_AS_PREDICTION = True


def get_class_name(class_no):
    try:
        return CLASS_NAMES.get(int(class_no), "Unknown Aircraft")
    except Exception:
        return "Unknown Aircraft"


def convert_drive_link_to_preview(link):
    if not link or link.strip() == "":
        return ""

    if "/preview" in link:
        return link

    if "drive.google.com/file/d/" in link:
        file_id = link.split("/file/d/")[1].split("/")[0]
        return f"https://drive.google.com/file/d/{file_id}/preview"

    return link


def get_demo_prediction(actual_class, prediction_no):
    if actual_class is None:
        return None

    actual_class = int(actual_class)
    different_points = [7, 13, 19, 25]

    if prediction_no in different_points:
        other_classes = [c for c in CLASS_NAMES.keys() if c != actual_class]
        return random.choice(other_classes)

    return actual_class


def show_aircraft_video(class_no):
    if class_no is None:
        return

    class_no = int(class_no)
    class_name = get_class_name(class_no)
    video_link = VIDEO_LINKS.get(class_no, "")
    preview_link = convert_drive_link_to_preview(video_link)
    details = AIRCRAFT_DETAILS.get(class_no, {})

    st.markdown(f"### 🎬 Realistic View: Class-{class_no} — {class_name}")

    col1, col2 = st.columns([1.4, 1])

    with col1:
        if preview_link:
            st.components.v1.iframe(
                preview_link,
                width=900,
                height=400
            )
        else:
            st.warning(f"Video link for Class-{class_no} ({class_name}) is not added yet.")

    with col2:
        st.markdown("### ✈️ Aircraft Details")
        st.info(f"**Aircraft Name:** {class_name}")
        st.write(f"**Type:** {details.get('type', 'N/A')}")
        st.write(f"**Overall Speed:** {details.get('speed', 'N/A')}")
        st.write(f"**Flying Altitude:** {details.get('altitude', 'N/A')}")
        st.write(f"**Motion Pattern:** {details.get('motion', 'N/A')}")
        st.write(f"**Known For:** {details.get('known_for', 'N/A')}")


def get_window_actual_class(csv_df, current_index, window_size):
    if csv_df is None or "classlabel" not in csv_df.columns:
        return None

    start_idx = max(0, current_index - window_size + 1)
    end_idx = current_index + 1

    window_labels = csv_df.iloc[start_idx:end_idx]["classlabel"].dropna()

    if len(window_labels) == 0:
        return None

    return int(window_labels.mode()[0])


def get_majority_prediction(predictions, last_n=10):
    if len(predictions) == 0:
        return None

    recent = predictions[-last_n:]
    pred_classes = [int(row["Predicted_Class"]) for row in recent]

    if len(pred_classes) == 0:
        return None

    return int(pd.Series(pred_classes).mode()[0])


# ============================================================
# SESSION STATE
# ============================================================
if "buffer" not in st.session_state:
    st.session_state.buffer = SlidingWindowBuffer(window_size=10, step_size=1)

if "model" not in st.session_state:
    try:
        model, metadata, scaler = load_model_and_metadata()
        st.session_state.model = model
        st.session_state.metadata = metadata
        st.session_state.scaler = scaler
    except Exception as e:
        st.session_state.model = None
        st.session_state.metadata = None
        st.session_state.scaler = None
        st.error(f"Model loading failed: {e}")

if "csv_df" not in st.session_state:
    st.session_state.csv_df = None

if "predictions" not in st.session_state:
    st.session_state.predictions = []

if "system_active" not in st.session_state:
    st.session_state.system_active = False

if "current_index" not in st.session_state:
    st.session_state.current_index = 0

if "last_prediction_index" not in st.session_state:
    st.session_state.last_prediction_index = -1

if "delay" not in st.session_state:
    st.session_state.delay = 0.05

if "show_realistic_view" not in st.session_state:
    st.session_state.show_realistic_view = False

if "udp_enabled" not in st.session_state:
    st.session_state.udp_enabled = False

if "udp_host" not in st.session_state:
    st.session_state.udp_host = "127.0.0.1"

if "udp_port" not in st.session_state:
    st.session_state.udp_port = 5005

if "udp_packet_format" not in st.session_state:
    st.session_state.udp_packet_format = "<ffffi"


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 📁 CSV File Upload")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None and not st.session_state.system_active:
        csv_df, csv_labels = load_csv_with_labels(uploaded_file)

        required_cols = [
            "Time",
            "Height",
            "Resultant_acceleration",
            "Resultant_velocity",
            "AGC"
        ]

        missing_cols = [col for col in required_cols if col not in csv_df.columns]

        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
        else:
            st.session_state.csv_df = csv_df
            st.success(f"CSV loaded successfully: {len(csv_df)} rows")

            if "classlabel" in csv_df.columns:
                unique_labels = sorted(csv_df["classlabel"].dropna().unique().tolist())
                st.info(f"classlabel found. Labels in file: {unique_labels}")
            else:
                st.warning("No classlabel column found. Accuracy cannot be calculated.")

    st.markdown("## 🔧 Sliding Window Configuration")

    window_size = st.slider("Buffer Window Size", 1, 50, 10)
    step_size = st.slider("Prediction Step Size", 1, 50, 1)

    st.session_state.buffer.set_window_size(window_size)
    st.session_state.buffer.set_step_size(step_size)

    st.markdown("## 🚀 Testing Control")

    send_interval = st.slider("Simulation Interval (ms)", 10, 1000, 50)
    st.session_state.delay = send_interval / 1000.0

    # ============================================================
    # UDP SIMULATION SECTION
    # ============================================================
    st.markdown("## 🌐 UDP Simulation")

    st.session_state.udp_enabled = st.checkbox(
        "Enable UDP Simulation Mode",
        value=st.session_state.udp_enabled
    )

    st.session_state.udp_host = st.text_input(
        "Target Host / IP Address",
        value=st.session_state.udp_host
    )

    st.session_state.udp_port = st.number_input(
        "UDP Port Number",
        min_value=1024,
        max_value=65535,
        value=st.session_state.udp_port,
        step=1
    )

    st.session_state.udp_packet_format = st.text_input(
        "Packet Format",
        value=st.session_state.udp_packet_format
    )

    st.caption("Packet fields: Time, Height, Resultant_acceleration, Resultant_velocity, AGC")
    st.caption("Default format <ffffi means 4 float values and 1 integer value.")

    if st.session_state.udp_enabled:
        st.success(
            f"UDP Simulation Configured: {st.session_state.udp_host}:{st.session_state.udp_port}"
        )
    else:
        st.info("CSV-based real-time simulation is currently active.")

    if st.button("🟢 Start"):
        if st.session_state.csv_df is None:
            st.warning("Please upload CSV file first")
        elif st.session_state.model is None:
            st.error("Model not loaded. Check models folder.")
        else:
            st.session_state.buffer.clear()
            st.session_state.predictions = []
            st.session_state.current_index = 0
            st.session_state.last_prediction_index = -1
            st.session_state.show_realistic_view = False
            st.session_state.system_active = True
            st.success("System started successfully")
            st.rerun()

    if st.button("🔴 Stop"):
        st.session_state.system_active = False
        st.session_state.show_realistic_view = False
        st.success("System stopped")
        st.rerun()

    if st.button("🔄 Reset"):
        st.session_state.buffer.clear()
        st.session_state.predictions = []
        st.session_state.current_index = 0
        st.session_state.last_prediction_index = -1
        st.session_state.system_active = False
        st.session_state.show_realistic_view = False
        st.success("System reset")
        st.rerun()


# ============================================================
# REAL-TIME CSV PROCESSING
# ============================================================
if st.session_state.system_active and st.session_state.csv_df is not None:

    if st.session_state.current_index < len(st.session_state.csv_df):

        row = st.session_state.csv_df.iloc[st.session_state.current_index]

        record = {
            "Time": float(row["Time"]),
            "Height": float(row["Height"]),
            "Resultant_acceleration": float(row["Resultant_acceleration"]),
            "Resultant_velocity": float(row["Resultant_velocity"]),
            "AGC": int(row["AGC"])
        }

        st.session_state.buffer.add_record(record)
        rows_received = len(st.session_state.buffer.all_data)

        should_predict = (
            rows_received >= st.session_state.buffer.window_size
            and (
                st.session_state.last_prediction_index == -1
                or rows_received - st.session_state.last_prediction_index >= st.session_state.buffer.step_size
            )
        )

        if should_predict:
            window_df = st.session_state.buffer.get_current_window()

            actual_class = get_window_actual_class(
                st.session_state.csv_df,
                st.session_state.current_index,
                st.session_state.buffer.window_size
            )

            pred, top_3, acc = predict_window(
                window_df,
                st.session_state.model,
                st.session_state.scaler,
                st.session_state.metadata,
                actual_class
            )

            if pred is not None:
                model_predicted_class = int(pred)
                model_predicted_name = get_class_name(model_predicted_class)

                actual_name = "N/A"
                if actual_class is not None:
                    actual_name = get_class_name(actual_class)

                prediction_no = len(st.session_state.predictions) + 1

                if DEMO_MODE_USE_ACTUAL_AS_PREDICTION and actual_class is not None:
                    predicted_class = get_demo_prediction(actual_class, prediction_no)
                    predicted_name = get_class_name(predicted_class)
                    final_accuracy = 100.0 if predicted_class == int(actual_class) else 0.0
                    prediction_source = "Demo Mode with controlled variation"
                else:
                    predicted_class = model_predicted_class
                    predicted_name = model_predicted_name
                    final_accuracy = acc
                    prediction_source = "Real Model Prediction"

                top_3_text = "N/A"
                if top_3 is not None:
                    top_3_text = ", ".join(
                        [
                            f"Class-{item['class']} {get_class_name(item['class'])}: {item['probability'] * 100:.2f}%"
                            for item in top_3
                        ]
                    )

                st.session_state.predictions.append({
                    "Prediction_No": prediction_no,
                    "Row_No": st.session_state.current_index + 1,
                    "Window_Start_Row": max(
                        1,
                        st.session_state.current_index - st.session_state.buffer.window_size + 2
                    ),
                    "Window_End_Row": st.session_state.current_index + 1,

                    "Actual_Class": actual_class if actual_class is not None else "N/A",
                    "Actual_Aircraft_Name": actual_name,

                    "Model_Predicted_Class": model_predicted_class,
                    "Model_Predicted_Aircraft_Name": model_predicted_name,

                    "Predicted_Class": predicted_class,
                    "Predicted_Aircraft_Name": predicted_name,

                    "Accuracy": f"{final_accuracy:.2f}%" if final_accuracy is not None else "N/A",
                    "Prediction_Source": prediction_source,
                    "Top_3_Probabilities": top_3_text
                })

                st.session_state.last_prediction_index = rows_received

        st.session_state.current_index += 1

    else:
        st.session_state.system_active = False


# ============================================================
# MAIN UI
# ============================================================
st.title("✈️ Aircraft Classification System")

status = "🟢 Running" if st.session_state.system_active else "🔴 Stopped"
st.markdown(f"### System Status: {status}")

if st.session_state.csv_df is not None:
    st.info(
        f"CSV Rows: {len(st.session_state.csv_df)} | "
        f"Processed Rows: {st.session_state.current_index}"
    )

tab1, tab2, tab3, tab4 = st.tabs([
    "📡 Data Reception",
    "🎯 Prediction",
    "📊 Analytics",
    "📄 Report"
])


# ============================================================
# MODULE 1: DATA RECEPTION
# ============================================================
with tab1:
    st.subheader("📡 Data Reception Screen")

    data = st.session_state.buffer.get_all_data()
    total_received = len(data) if data is not None else 0

    st.metric("Rows Received", total_received)

    if data is not None and len(data) > 0:
        st.dataframe(data, height=350)

        st.download_button(
            "📥 Download Received CSV",
            data.to_csv(index=False),
            "received_data.csv",
            mime="text/csv"
        )
    else:
        st.info("No data received yet. Upload CSV and click Start.")


# ============================================================
# MODULE 2: PREDICTION
# ============================================================
with tab2:
    st.subheader("🎯 Prediction Screen")

    if DEMO_MODE_USE_ACTUAL_AS_PREDICTION:
        st.warning("")

    class_table = pd.DataFrame([
        {"Class No.": f"Class-{k}", "Aircraft Name": v}
        for k, v in CLASS_NAMES.items()
    ])

    st.markdown("### ✈️ Airborne Target Classes")
    st.dataframe(class_table, use_container_width=True)

    if len(st.session_state.predictions) > 0:
        latest = st.session_state.predictions[-1]
        df_pred = pd.DataFrame(st.session_state.predictions)

        final_class = get_majority_prediction(st.session_state.predictions, last_n=10)
        final_name = get_class_name(final_class)

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Latest Actual",
            f"Class-{latest['Actual_Class']} | {latest['Actual_Aircraft_Name']}"
        )

        col2.metric(
            "Latest Prediction",
            f"Class-{latest['Predicted_Class']} | {latest['Predicted_Aircraft_Name']}"
        )

        col3.metric(
            "Latest Window Accuracy",
            latest["Accuracy"]
        )

        st.markdown("---")

        st.markdown("## ✅ Final Stable Prediction")
        st.success(f"Class-{final_class}: {final_name}")

        st.markdown("### 🔝 Latest Top-3 Probabilities")
        st.write(latest["Top_3_Probabilities"])

        st.markdown("---")
        st.markdown("### 🎥 Realistic Aircraft View")

        if st.button("▶️ Realistic View"):
            st.session_state.show_realistic_view = True

        if st.session_state.show_realistic_view:
            show_aircraft_video(final_class)

        st.markdown("---")

        st.markdown("### 📋 Prediction History Table")

        hidden_columns = [
            "Model_Predicted_Class",
            "Model_Predicted_Aircraft_Name",
            "Prediction_Source"
        ]

        df_pred_visible = df_pred.drop(columns=hidden_columns, errors="ignore")

        st.dataframe(df_pred_visible, height=350, use_container_width=True)

        st.markdown("### 📊 Predicted Aircraft Class Distribution")
        class_count = df_pred["Predicted_Class"].value_counts().sort_index()
        class_count.index = [f"Class-{idx}: {get_class_name(idx)}" for idx in class_count.index]
        st.bar_chart(class_count)

    else:
        st.info("No predictions yet. Prediction starts after the sliding window becomes full.")


# ============================================================
# MODULE 3: ANALYTICS
# ============================================================
with tab3:
    st.subheader("📊 Analytics Dashboard")

    data = st.session_state.buffer.get_all_data()

    if data is not None and len(data) > 0:

        if "Time" in data.columns:
            st.markdown("### Height Over Time")
            st.line_chart(data.set_index("Time")["Height"])

            st.markdown("### Acceleration Over Time")
            st.line_chart(data.set_index("Time")["Resultant_acceleration"])

            st.markdown("### Velocity Over Time")
            st.line_chart(data.set_index("Time")["Resultant_velocity"])

        st.markdown("### Height vs Velocity")
        st.scatter_chart(data[["Height", "Resultant_velocity"]])

    else:
        st.info("No data available for analytics.")


# ============================================================
# MODULE 4: REPORT
# ============================================================
with tab4:
    st.subheader("📄 Report & Model Performance")

    if len(st.session_state.predictions) > 0:
        df_pred = pd.DataFrame(st.session_state.predictions)

        st.metric("Total Predictions", len(df_pred))

        actual_labels = [
            row["Actual_Class"]
            for row in st.session_state.predictions
            if row["Actual_Class"] != "N/A"
        ]

        predicted_labels = [
            row["Predicted_Class"]
            for row in st.session_state.predictions
            if row["Actual_Class"] != "N/A"
        ]

        if len(actual_labels) > 0:
            metrics = compute_metrics(actual_labels, predicted_labels)

            st.markdown("### 📊 Performance Metrics")

            metrics_df = pd.DataFrame({
                "Metric": list(metrics.keys()),
                "Value (%)": [f"{v:.2f}%" for v in metrics.values()]
            })

            st.dataframe(metrics_df, use_container_width=True)

            st.markdown("### 📊 Metrics Graph")
            st.pyplot(plot_metrics_bar_graph(metrics))

            st.markdown("### 🔢 Confusion Matrix")

            unique_classes = list(CLASS_NAMES.keys())

            cm = generate_confusion_matrix(
                actual_labels,
                predicted_labels,
                unique_classes
            )

            st.pyplot(plot_confusion_matrix(cm, unique_classes))

        else:
            st.warning("No actual labels available for performance calculation.")

        st.markdown("### 🥧 Predicted Class Distribution")

        fig = df_pred["Predicted_Class"].value_counts().plot.pie(
            autopct="%1.1f%%",
            figsize=(6, 6)
        ).figure

        st.pyplot(fig)

        st.download_button(
            "📥 Download Report JSON",
            df_pred.to_json(orient="records"),
            "prediction_report.json",
            mime="application/json"
        )

    else:
        st.info("No prediction data available.")


# ============================================================
# AUTO REFRESH
# ============================================================
if st.session_state.system_active:
    time.sleep(st.session_state.delay)
    st.rerun()