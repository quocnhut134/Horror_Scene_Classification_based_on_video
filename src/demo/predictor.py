import pandas as pd
import numpy as np
import os
import sys

current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.abspath(os.path.join(current_file_dir, '..', '..'))
sys.path.append(project_root_dir)

from src.train_model.serialization_utils import load_model_artifacts

final_classifier_model = None
scaler_for_inference = None
label_encoder_for_inference = None

def load_inference_models(project_root_dir, model_name=""):

    global final_classifier_model, scaler_for_inference, label_encoder_for_inference

    if final_classifier_model is None or scaler_for_inference is None or label_encoder_for_inference is None:
        final_classifier_model, scaler_for_inference, label_encoder_for_inference = \
            load_model_artifacts(project_root_dir, model_name)
        
        if final_classifier_model and scaler_for_inference and label_encoder_for_inference:
            print("Classifier model, scaler, and label encoder loaded successfully.")
        else:
            print("Failed to load one or more inference artifacts. Please ensure models are trained and saved.")
            final_classifier_model = None
            scaler_for_inference = None
            label_encoder_for_inference = None
    else:
        print("Inference models already loaded.")
            
    return final_classifier_model, scaler_for_inference, label_encoder_for_inference

def predict_segment_horror_category(features_dict, model, scaler, label_encoder):

    if model is None or scaler is None or label_encoder is None:
        print("Model, scaler, or label encoder not loaded. Cannot predict segment.")
        return "Unknown", 0.0

    feature_columns = [
        'violence_score', 'brightness_std', 'scream_score', 'silence_score',
        'noise_score', 'music_score', 'hum_score', 'speech_score', 'tension_score'
    ]
    
    segment_df = pd.DataFrame([features_dict], columns=feature_columns)
    
    scaled_features = scaler.transform(segment_df)
    
    # Predict the label
    prediction_encoded = model.predict(scaled_features)
    predicted_label = label_encoder.inverse_transform(prediction_encoded)[0]

    # Get probabilities
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(scaled_features)[0]
        predicted_proba_idx = np.where(label_encoder.classes_ == predicted_label)[0][0]
        predicted_probability = probabilities[predicted_proba_idx]
    else:
        predicted_probability = 1.0 # Or some default if model doesn't support proba

    return predicted_label, predicted_probability


def classify_long_video_by_thresholding(segment_predictions, final_thresholds):

    if not segment_predictions:
        return "Calm Neutral"

    total_segments = len(segment_predictions)
    # Count frequency of appearance of each label
    label_counts = {label: segment_predictions.count(label) for label in set(segment_predictions)}

    # Calculate the appearance ratio of each label
    label_ratios = {label: count / total_segments for label, count in label_counts.items()}
    
    # Determine final label by thresholds (Disturbing > Jump Scare > Psychological)
    if "Disturbing Visuals" in label_ratios and label_ratios["Disturbing Visuals"] >= final_thresholds.get("Disturbing Visuals", 0):
        return "Disturbing Visuals"
    
    if "Jump scare" in label_ratios and label_ratios["Jump scare"] >= final_thresholds.get("Jump Scare", 0):
        return "Jump scare"
        
    if "Psychological Tension" in label_ratios and label_ratios["Psychological Tension"] >= final_thresholds.get("Psychological Tension", 0):
        return "Psychological Tension"

    return "Calm Neutral"