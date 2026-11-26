
import os
import tempfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Grab-and-Go Autonomous Checkout",
    page_icon="🛒",
    layout="wide"
)

MODEL_PATH = os.path.join("models", "action_model_v5_compatible.keras")

LABELS = ["TAKE", "PUT", "TOUCH"]

FRAMES_PER_WINDOW = 32
IMG_SIZE = 112

WINDOW_SECONDS = 2.0
CONFIDENCE_THRESHOLD = 0.60


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return tf.keras.models.load_model(MODEL_PATH, compile=False)


# --------------------------------------------------
# READ VIDEO
# --------------------------------------------------

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Could not open the uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0:
        cap.release()
        raise ValueError("Could not determine video frame rate.")

    frames = []

    while True:
        success, frame = cap.read()

        if not success:
            break

        frame = cv2.resize(
            frame,
            (IMG_SIZE, IMG_SIZE)
        )

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        frames.append(frame)

    cap.release()

    if not frames:
        raise ValueError("No readable frames found in the video.")

    return np.asarray(frames, dtype=np.uint8), float(fps)


# --------------------------------------------------
# PREPARE MODEL INPUT
# --------------------------------------------------

def prepare_window(frames):
    if len(frames) == 0:
        raise ValueError("Cannot prepare an empty video window.")

    indices = np.linspace(
        0,
        len(frames) - 1,
        FRAMES_PER_WINDOW
    ).astype(int)

    sampled_frames = frames[indices]

    # Match the V5 training input shape:
    # (batch, 32, 112, 112, 3)
    sampled_frames = sampled_frames.astype(np.float32) / 255.0

    return np.expand_dims(sampled_frames, axis=0)


# --------------------------------------------------
# PREDICT ACTIONS IN VIDEO WINDOWS
# --------------------------------------------------

def analyze_video(model, frames, fps):
    total_frames = len(frames)

    window_size = max(
        1,
        int(round(WINDOW_SECONDS * fps))
    )

    # Half-window stride gives overlapping windows.
    step = max(1, window_size // 2)

    results = []

    for start in range(0, total_frames, step):
        end = min(
            start + window_size,
            total_frames
        )

        segment_length = end - start

        # Skip a very short final fragment.
        if segment_length < max(2, window_size // 4):
            continue

        clip = frames[start:end]

        model_input = prepare_window(clip)

        probabilities = model.predict(
            model_input,
            verbose=0
        )[0]

        predicted_index = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_index])

        if confidence >= CONFIDENCE_THRESHOLD:
            action = LABELS[predicted_index]
        else:
            action = "UNCERTAIN"

        results.append({
            "Start (s)": round(start / fps, 2),
            "End (s)": round(end / fps, 2),
            "Predicted action": action,
            "Confidence (%)": round(confidence * 100, 1),
            "TAKE (%)": round(float(probabilities[0]) * 100, 1),
            "PUT (%)": round(float(probabilities[1]) * 100, 1),
            "TOUCH (%)": round(float(probabilities[2]) * 100, 1)
        })

        if end == total_frames:
            break

    return results


# --------------------------------------------------
# GROUP CONSECUTIVE PREDICTIONS
# --------------------------------------------------

def group_events(results):
    if not results:
        return []

    events = []

    for row in results:
        action = row["Predicted action"]

        if events and events[-1]["Action"] == action:
            events[-1]["End (s)"] = row["End (s)"]
            events[-1]["Window count"] += 1
            events[-1]["Confidence values"].append(
                row["Confidence (%)"]
            )
        else:
            events.append({
                "Start (s)": row["Start (s)"],
                "End (s)": row["End (s)"],
                "Action": action,
                "Window count": 1,
                "Confidence values": [
                    row["Confidence (%)"]
                ]
            })

    grouped = []

    for event in events:
        confidence_values = event["Confidence values"]

        grouped.append({
            "Start (s)": event["Start (s)"],
            "End (s)": event["End (s)"],
            "Candidate action": event["Action"],
            "Windows combined": event["Window count"],
            "Average confidence (%)": round(
                sum(confidence_values) / len(confidence_values),
                1
            ),
            "Status": "Review required"
        })

    return grouped


# --------------------------------------------------
# STREAMLIT USER INTERFACE
# --------------------------------------------------

st.title("🛒 Grab-and-Go Autonomous Checkout")

st.write(
    "Analyze prerecorded shopping videos using an AI action-recognition model."
)

st.caption(
    "Current model: V5 | Actions: TAKE, PUT, TOUCH"
)

st.divider()

uploaded_video = st.file_uploader(
    "Upload a prerecorded shopping video",
    type=["mp4", "avi", "mov", "mkv"]
)

if uploaded_video is not None:

    st.subheader("Video Preview")
    st.video(uploaded_video)

    st.write("Selected file:", uploaded_video.name)

    if st.button("Analyze Full Video"):

        temporary_path = None

        try:
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=os.path.splitext(uploaded_video.name)[1]
            ) as temporary_file:

                temporary_file.write(
                    uploaded_video.getbuffer()
                )

                temporary_path = temporary_file.name

            with st.spinner(
                "Loading model and analyzing video..."
            ):
                model = load_model()

                frames, fps = read_video(
                    temporary_path
                )

                results = analyze_video(
                    model,
                    frames,
                    fps
                )

            if not results:
                st.warning(
                    "No suitable video windows were found."
                )

            else:
                video_duration = len(frames) / fps

                st.success(
                    "Video analysis completed successfully."
                )

                st.subheader("Video Information")

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Duration",
                    f"{video_duration:.2f} sec"
                )

                col2.metric(
                    "Frame rate",
                    f"{fps:.2f} FPS"
                )

                col3.metric(
                    "Windows analyzed",
                    len(results)
                )

                # ----------------------------------
                # RAW PREDICTIONS
                # ----------------------------------

                st.subheader("Action Timeline")

                raw_df = pd.DataFrame(results)

                st.dataframe(
                    raw_df,
                    use_container_width=True
                )

                # ----------------------------------
                # GROUPED EVENTS
                # ----------------------------------

                st.subheader("Grouped Candidate Events")

                grouped_events = group_events(
                    results
                )

                grouped_df = pd.DataFrame(
                    grouped_events
                )

                st.dataframe(
                    grouped_df,
                    use_container_width=True
                )

                # ----------------------------------
                # SUMMARY
                # ----------------------------------

                st.subheader("Prediction Summary")

                summary = {}

                for label in LABELS:
                    summary[label] = sum(
                        row["Predicted action"] == label
                        for row in results
                    )

                summary["UNCERTAIN"] = sum(
                    row["Predicted action"] == "UNCERTAIN"
                    for row in results
                )

                c1, c2, c3, c4 = st.columns(4)

                c1.metric("TAKE windows", summary["TAKE"])
                c2.metric("PUT windows", summary["PUT"])
                c3.metric("TOUCH windows", summary["TOUCH"])
                c4.metric("Uncertain windows", summary["UNCERTAIN"])

                # ----------------------------------
                # IMPORTANT LIMITATIONS
                # ----------------------------------

                st.warning(
                    "These are window-level predictions, not verified "
                    "product transactions. Overlapping windows may repeat "
                    "the same action. Grouping combines adjacent predictions "
                    "but does not establish the exact action time or identify "
                    "the product. Do not update the shopping cart automatically "
                    "from these predictions."
                )

        except Exception as error:
            st.error(
                f"Video analysis failed: {error}"
            )

        finally:
            if (
                temporary_path is not None
                and os.path.exists(temporary_path)
            ):
                os.remove(temporary_path)

else:
    st.info(
        "Upload a prerecorded shopping video to begin."
    )

st.divider()

st.caption(
    "Grab-and-Go | AI-Based Autonomous Checkout | Local Demo"
)
