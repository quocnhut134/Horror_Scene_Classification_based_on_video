import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import joblib

current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.abspath(os.path.join(current_file_dir, '..', '..'))
sys.path.append(project_root_dir)

from src.utils.constants import YAMNET_CLASS_NAMES

model_yolo_cls = None
model_yamnet = None

TRAINED_YOLO_CLS_MODEL_PATH = os.path.join(
    project_root_dir, 'src', 'YoloV8', 'yolov8n_violence_classification_model', 'weights', 'best.pt'
)

def load_models_for_feature_extraction(project_root_dir):
    global model_yolo_cls, model_yamnet

    if model_yolo_cls is None:
        try:
            if os.path.exists(TRAINED_YOLO_CLS_MODEL_PATH):
                model_yolo_cls = YOLO(TRAINED_YOLO_CLS_MODEL_PATH)
            else:
                print(f"Error: YOLO model file not found at {TRAINED_YOLO_CLS_MODEL_PATH}")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            model_yolo_cls = None
    else:
        print("YOLO model already loaded.")

    # YAMNet Model
    if model_yamnet is None:
        try:
            model_yamnet = hub.load('https://tfhub.dev/google/yamnet/1')
        except Exception as e:
            print(f"Error loading YAMNet model from TensorFlow Hub: {e}")
            model_yamnet = None
    else:
        print("YAMNet model already loaded.")

    return model_yolo_cls, model_yamnet

def calculate_violence_score(frames, yolo_model):

    if not yolo_model:
        return 0.0

    violence_scores = []
    for frame in frames:
        results = yolo_model(frame, verbose=False) 
        
        violent_act_detected = False
        for r in results:
            if r.boxes: 
                for cls_id in r.boxes.cls:
                    if int(cls_id) == 0: 
                        violent_act_detected = True
                        break
            if violent_act_detected:
                break
        
        violence_scores.append(1 if violent_act_detected else 0)
    
    return np.mean(violence_scores) if violence_scores else 0.0

def calculate_brightness_std(frames):

    brightness_values = []
    for frame in frames:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness_values.append(np.mean(gray_frame))
    return np.std(brightness_values) if brightness_values else 0.0

def calculate_audio_features_yamnet(audio_data, yamnet_model):

    if yamnet_model is None or audio_data.size == 0:
        return {
            'scream_score': 0.0, 'silence_score': 0.0, 'noise_score': 0.0,
            'music_score': 0.0, 'hum_score': 0.0, 'speech_score': 0.0,
            'tension_score': 0.0
        }

    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)
    if np.max(np.abs(audio_data)) > 1.0:
        audio_data = audio_data / np.max(np.abs(audio_data))

    scores, embeddings, spectrogram = yamnet_model(audio_data)
    
    class_names = YAMNET_CLASS_NAMES

    scream_score = 0.0
    for idx, name in enumerate(class_names):
        if "Screaming" in name or "Shriek" in name or "Scream" in name:
            scream_score += np.mean(scores.numpy()[:, idx])
    
    silence_score = np.mean(scores.numpy()[:, class_names.index('Silence')]) if 'Silence' in class_names else 0.0
    noise_score = np.mean(scores.numpy()[:, class_names.index('Noise')]) if 'Noise' in class_names else 0.0
    music_score = np.mean(scores.numpy()[:, class_names.index('Music')]) if 'Music' in class_names else 0.0
    hum_score = np.mean(scores.numpy()[:, class_names.index('Hum')]) if 'Hum' in class_names else 0.0
    speech_score = np.mean(scores.numpy()[:, class_names.index('Speech')]) if 'Speech' in class_names else 0.0

    tension_score = 0.0
    tension_keywords = ["Screaming", "Shriek", "Howl", "Gurgle", "Moan", "Groan", "Alarm", "Siren", "Explosion", "Noise", "Sniff"]
    for idx, name in enumerate(class_names):
        if any(keyword in name for keyword in tension_keywords):
            tension_score += np.mean(scores.numpy()[:, idx])

    return {
        'violence_score': 0.0, 
        'brightness_std': 0.0, 
        'scream_score': scream_score,
        'silence_score': silence_score,
        'noise_score': noise_score,
        'music_score': music_score,
        'hum_score': hum_score,
        'speech_score': speech_score,
        'tension_score': tension_score
    }

def extract_features_from_segment(frames, audio_data, yolo_model, yamnet_model):

    violence_score = calculate_violence_score(frames, yolo_model)
    brightness_std = calculate_brightness_std(frames)
    audio_features = calculate_audio_features_yamnet(audio_data, yamnet_model)

    features = {
        'violence_score': violence_score,
        'brightness_std': brightness_std,
        'scream_score': audio_features.get('scream_score', 0.0),
        'silence_score': audio_features.get('silence_score', 0.0),
        'noise_score': audio_features.get('noise_score', 0.0),
        'music_score': audio_features.get('music_score', 0.0),
        'hum_score': audio_features.get('hum_score', 0.0),
        'speech_score': audio_features.get('speech_score', 0.0),
        'tension_score': audio_features.get('tension_score', 0.0)
    }
    return features