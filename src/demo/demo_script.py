import os
import sys
import time
import gc
import tensorflow as tf
import numpy as np

current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.abspath(os.path.join(current_file_dir, '..', '..'))
sys.path.append(project_root_dir)

from src.demo.video_processor import get_video_clip, extract_video_segment, get_frames_from_segment, get_audio_from_segment
from src.demo.feature_extractor import load_models_for_feature_extraction, extract_features_from_segment
from src.demo.predictor import load_inference_models, predict_segment_horror_category, classify_long_video_by_thresholding

def setup_gpu_environment():

    print("\n--- Checking GPU ---")
    gpus_tf = tf.config.experimental.list_physical_devices('GPU')
    if gpus_tf:
        print(f"TensorFlow found GPU: {gpus_tf}")
        try:
            for gpu in gpus_tf:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"Error with GPU settings for TensorFlow: {e}")
    else:
        print("TensorFlow could not find GPU, trying to use CPU instead.")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"PyTorch found GPU: {torch.cuda.get_device_name(0)} (Total: {torch.cuda.device_count()})")
            try:
                test_tensor = torch.randn(1).to('cuda')
                del test_tensor
            except Exception as e:
                print(f"PyTorch found GPU but take errors when using it: {e}")
                print("There are some problems with CUDA/CuDNN or conflicting between versions.")
        else:
            print("PyTorch could not find GPU, trying to use CPU instead.")
    except ImportError:
        print("PyTorch is not downloaded")


def process_video_for_horror_classification(video_path, segment_duration=10, model_name="Logistic Regression"):
    
    print(f"\n--- Start classification process for input video: {video_path} ---")
    print(f"Project Root Directory: {project_root_dir}")

    # Yolo, YAMNet, Classifier
    yolo_model, yamnet_model = load_models_for_feature_extraction(project_root_dir)
    classifier_model, scaler, label_encoder = load_inference_models(project_root_dir, model_name)

    if not all([yolo_model, yamnet_model, classifier_model, scaler, label_encoder]):
        print("Error: Cannot download all of need models")
        return "Toang"

    # Video loading and processing
    video_clip = get_video_clip(video_path)
    if video_clip is None:
        print("Error: Cannot download input video")
        return "Toang"

    total_duration = video_clip.duration
    segment_predictions = []

    print(f"Length of video: {total_duration:.2f} seconds. Length of segment: {segment_duration} seconds.")

    for start_time in np.arange(0, total_duration, segment_duration):
        end_time = min(start_time + segment_duration, total_duration)
        if end_time - start_time < 1.0: # Skip short segments
            continue

        print(f"\nProcessing segment: {start_time:.2f}s - {end_time:.2f}s")
        segment_clip = None
        segment_frames = []
        segment_audio_data = np.array([])

        try:
            # Segments extracting from video, audio
            segment_clip = extract_video_segment(video_clip, start_time, end_time)
            segment_frames = get_frames_from_segment(segment_clip, frames_per_second=1)
            segment_audio_data, _ = get_audio_from_segment(segment_clip)
        except Exception as e:
            print(f"Error when extracting segments ({start_time:.2f}-{end_time:.2f}s): {e}")
            continue 

        # Feature extracting from segments
        if segment_frames or segment_audio_data.size > 0:
            features = extract_features_from_segment(segment_frames, segment_audio_data, yolo_model, yamnet_model)
            
            # Predict for segments
            predicted_label, predicted_proba = predict_segment_horror_category(features, classifier_model, scaler, label_encoder)
            segment_predictions.append(predicted_label)
            print(f"Predict for segment: '{predicted_label}' (Probability: {predicted_proba:.4f})")
        else:
            print(f"There is no image or audio data in segment {start_time:.2f}-{end_time:.2f}s. Skipping...")
            segment_predictions.append("Calm Neutral") # Neutral label for segment if cannot extract anything.
        
        del segment_clip, segment_frames, segment_audio_data, features
        gc.collect()
        tf.keras.backend.clear_session()
        time.sleep(0.5) 

    # Thresholds
    final_thresholds = {
        "Disturbing Visuals": 0.001, 
        "Jump scare": 0.10,         
        "Psychological Tension": 0.30 
    }
    final_overall_label = classify_long_video_by_thresholding(segment_predictions, final_thresholds)

    print(f"\n--- Complete classifying total video ---")
    print(f"All segment predictions: {segment_predictions}")
    
    return final_overall_label

if __name__ == "__main__":
    
    # P/s: Replace this link if you have FFMPEG in another position or it existed in system path
    os.environ["FFMPEG_BINARY"] = r"C:\ffmpeg\bin\ffmpeg.exe" 

    setup_gpu_environment()

    video_to_classify_path = r"E:\Working\Learning\Subjects\Horror_scene_CLS\test_video\1.mp4"

    trained_model_name = "random_forest" 

    if os.path.exists(video_to_classify_path):
        final_category = process_video_for_horror_classification(
            video_to_classify_path, 
            segment_duration=10, # Length of segment(s)
            model_name=trained_model_name
        )
        print(f"\n>>> FINAL CLASSIFICATION RESULT FOR VIDEO '{os.path.basename(video_to_classify_path)}': {final_category}")
    else:
        print(f"Error: cannot find video at: {video_to_classify_path}")