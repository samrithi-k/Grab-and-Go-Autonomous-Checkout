# Grab-and-Go – AI-Based Retail Action Recognition

## Project Overview

Grab-and-Go is a computer vision-based retail monitoring prototype that identifies customer-product interaction actions from uploaded video recordings.

The current prototype uses a CNN-LSTM deep learning model to classify actions into three categories:

- TAKE
- PUT
- TOUCH

## Features

- Upload prerecorded video for analysis.
- Extract and preprocess video frames using OpenCV.
- Predict customer interaction actions using a trained TensorFlow/Keras model.
- Display prediction confidence scores.
- Visualize predicted actions through an event timeline.
- Flag uncertain predictions for manual review.

## Technology Stack

- Python
- TensorFlow and Keras
- CNN-LSTM
- OpenCV
- NumPy
- Pandas
- Streamlit

## Project Structure

    Grab-and-Go-Autonomous-Checkout/
    ├── app.py
    ├── requirements.txt
    ├── README.md
    └── models/
        └── action_model_v5.keras

## Run Locally

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install dependencies:

       pip install -r requirements.txt

4. Start the application:

       streamlit run app.py

## Current Status

This is an ongoing academic prototype. The current implementation supports prerecorded video-based action recognition. Product identification, multi-camera tracking, virtual cart management, payment processing and automated billing are not yet implemented as integrated features.

Predictions are experimental and should be reviewed before being used for any retail transaction.
