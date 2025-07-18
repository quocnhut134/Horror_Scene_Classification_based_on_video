import os
import cv2
import numpy as np
import librosa
from moviepy import VideoFileClip
import tensorflow as tf
import gc

from src.models.loader import get_yolo_model, get_yamnet_model
from src.utils.constants import YAMNET_DISPLAY_NAMES
from src.utils.video_audio import calculate_brightness_std, delete_temp_file_with_retries

def get_violence_score_from_trained_model(video_path, violence_class_name='violence', confidence_threshold=0.5, inference_skip_frames=5):
    model_yolo_cls = get_yolo_model()
    if model_yolo_cls is None: 
        return 0.0

    cap = None
    try:
        cap = cv2.VideoCapture(video_path) 
        if not cap.isOpened():
            print(f"Error: cannnot open file for calculating violence_score: {os.path.basename(video_path)}") 
            return 0.0
        
        total_frames_processed = 0 
        violence_classified_frames = 0 
        
        class_names_map = model_yolo_cls.names 
        violence_cls_id = None
        for cls_id, name in class_names_map.items(): 
            if name.lower() == violence_class_name.lower():
                violence_cls_id = cls_id 
                break
        
        if violence_cls_id is None: 
            print(f"Error: Cannot find class '{violence_class_name}' in cls md. Available class: {class_names_map}") 
            return 0.0

        frame_read_count = 0 
        while True:
            ret, frame = cap.read() 
            if not ret:
                break
            
            frame_read_count += 1 
            if frame_read_count % inference_skip_frames != 0: 
                continue

            total_frames_processed += 1 

            if frame is None or frame.size == 0: 
                continue
            
            results = model_yolo_cls(frame, verbose=False, device='0' if tf.config.list_physical_devices('GPU') else 'cpu') 

            if results and results[0].probs: 
                predicted_class_id = results[0].probs.top1 
                confidence = results[0].probs.top1conf.item() 

                if predicted_class_id == violence_cls_id and confidence >= confidence_threshold: 
                    violence_classified_frames += 1 
        
        violence_score = violence_classified_frames / total_frames_processed if total_frames_processed > 0 else 0.0 
        return violence_score

    except Exception as e:
        print(f"Error when calculating violence_score for {os.path.basename(video_path)}: {e}") 
        return 0.0
    finally:
        if cap is not None and cap.isOpened(): 
            cap.release()
            del cap
        gc.collect()

def analyze_audio_yamnet(video_path, target_class_name):
    model_yamnet = get_yamnet_model()
    if model_yamnet is None: 
        return 0.0

    audio_temp_path = video_path.replace(".mp4", "_yamnet_audio_temp.wav")
    
    audio_data = None
    sr_data = None
    video_clip = None

    try:
        if not os.path.exists(audio_temp_path): 
            video_clip = VideoFileClip(video_path) 
            if video_clip.audio is None:
                return 0.0 # sound track is not exist

            video_clip.audio.write_audiofile(audio_temp_path, codec='pcm_s16le', logger=None) 
            video_clip.audio.close() 
            video_clip.close()

        audio_data, sr_data = librosa.load(audio_temp_path, sr=16000, mono=True) 
        scores, embeddings, spectrogram = model_yamnet(tf.constant(audio_data, dtype=tf.float32)) 
        
        class_names = YAMNET_DISPLAY_NAMES
        target_idx = np.where(class_names == target_class_name)[0] 
        
        if target_idx.size > 0:
            score_mean = tf.reduce_mean(scores[:, target_idx[0]]) 
            return float(score_mean)
        else:
            return 0.0 

    except Exception as e:
        print(f"Error with audio processing ('{target_class_name}') at {os.path.basename(video_path)}: {e}") 
        return 0.0
    finally:
        if video_clip is not None:
            video_clip.close()
            del video_clip
        
        delete_temp_file_with_retries(audio_temp_path)

        if audio_data is not None: del audio_data 
        if sr_data is not None: del sr_data 
        if 'scores' in locals() and scores is not None: del scores 
        if 'embeddings' in locals() and embeddings is not None: del embeddings 
        if 'spectrogram' in locals() and spectrogram is not None: del spectrogram 
        gc.collect() 


def extract_horror_features(video_path):
    if not os.path.exists(video_path): 
        print(f"Error: video is not exist {os.path.basename(video_path)}") 
        return {
            'violence_score': 0.0,
            'brightness_std': 0.0,
            'scream_score': 0.0,
            'silence_score': 0.0,
            'noise_score': 0.0,
            'music_score': 0.0,
            'hum_score': 0.0,
            'speech_score': 0.0,
            'tension_score': 0.0,
            'has_audio_track': False 
        }

    has_audio_track = False
    try:
        clip_check_audio = VideoFileClip(video_path) 
        has_audio_track = clip_check_audio.audio is not None 
        clip_check_audio.close() 
        del clip_check_audio
    except Exception as e:
        print(f"Error check sound track of {os.path.basename(video_path)}: {e}") 
        has_audio_track = False
    
    violence_score = get_violence_score_from_trained_model(
        video_path,
        violence_class_name='violence',
        confidence_threshold=0.5, 
        inference_skip_frames=5
    ) 
    
    brightness_std = calculate_brightness_std(video_path, frame_skip=5) 

    scream_score = 0.0
    silence_score = 0.0
    noise_score = 0.0
    music_score = 0.0
    hum_score = 0.0
    speech_score = 0.0

    if has_audio_track:
        scream_score = analyze_audio_yamnet(video_path, 'Scream') 
        silence_score = analyze_audio_yamnet(video_path, 'Silence') 
        noise_score = analyze_audio_yamnet(video_path, 'Noise') 
        music_score = analyze_audio_yamnet(video_path, 'Music') 
        hum_score = analyze_audio_yamnet(video_path, 'Hum') 
        speech_score = analyze_audio_yamnet(video_path, 'Speech') 
    
    scream_score = max(0.0, min(1.0, scream_score)) 
    silence_score = max(0.0, min(1.0, silence_score)) 
    noise_score = max(0.0, min(1.0, noise_score)) 
    music_score = max(0.0, min(1.0, music_score)) 
    hum_score = max(0.0, min(1.0, hum_score)) 
    speech_score = max(0.0, min(1.0, speech_score)) 


    tension_score_combined = (scream_score * 0.4 + 
                              (1 - speech_score) * 0.3 + 
                              noise_score * 0.2 + 
                              silence_score * 0.1 +  
                              music_score * 0.1) 
    
    total_weights = 0.4 + 0.3 + 0.2 + 0.1 + 0.1 
    tension_score = tension_score_combined / total_weights if total_weights > 0 else 0.0 

    return {
        'violence_score': violence_score,
        'brightness_std': brightness_std,
        'scream_score': scream_score,
        'silence_score': silence_score,
        'noise_score': noise_score,
        'music_score': music_score,
        'hum_score': hum_score, 
        'speech_score': speech_score, 
        'tension_score': tension_score, 
        'has_audio_track': has_audio_track 
    }