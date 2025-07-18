import os
import datetime
import json
import pandas as pd
import gc

from src.features.extractor import extract_horror_features
from src.utils.video_audio import trim_video, delete_temp_file_with_retries
from src.utils.constants import TH_VIOLENCE, TH_SCREAM, TH_BRIGHTNESS_STD, TH_TENSION

def assign_final_horror_label(features):
    violence_score = features['violence_score'] 
    brightness_std = features['brightness_std'] 
    scream_score = features['scream_score'] 
    tension_score = features['tension_score'] 
    
    # 1. Disturbing Visuals -> Highest priority
    if violence_score > TH_VIOLENCE: 
        return 'Disturbing Visuals', 3
    # 2. Jump Scare 
    elif (scream_score > TH_SCREAM) or (brightness_std > TH_BRIGHTNESS_STD): 
        return 'Jump scare', 2 
    # 3. Psychological Tension
    elif tension_score > TH_TENSION: 
        return 'Psychological Tension', 1 
    # 4. Calm / Neutral 
    else:
        return 'Calm Neutral', 0 

def process_vidharm_dataset(json_path, video_base_path, dataset_name, project_root_dir, output_csv_filename_prefix='vidharm_horror_features_and_labels'):

    if not os.path.exists(json_path): 
        print(f"Error: File JSON is not found at {json_path}") 
        return []
    if not os.path.exists(video_base_path): 
        print(f"Error: base video dir is not found at {video_base_path}") 
        return []

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f) 

    results_for_csv = [] 
    total_clips = len(data) 
    
    processed_clips_dir = os.path.join(project_root_dir, 'processed_clips') 
    os.makedirs(processed_clips_dir, exist_ok=True) 

    print(f"\n--- Start processing dataset: {dataset_name} ({total_clips} clips) ---") 

    for i, clip in enumerate(data):
        current_time = datetime.datetime.now().strftime("%H:%M:%S") 
        print(f"[{current_time}] processing video {i+1}/{total_clips} of {dataset_name}: {clip['filename']}") 
        
        video_id = clip['filename'].split('.')[0] 
        video_original_path = os.path.join(video_base_path, video_id, f"{video_id}.mp4") 
        trimmed_video_path = os.path.join(processed_clips_dir, f"trimmed_{video_id}.mp4") 

        row_data = {
            'video_id': video_id,
            'original_vidharm_label': clip.get('label', 'N/A'),  
            'dataset_split': dataset_name,  
            'violence_score': 0.0,
            'brightness_std': 0.0,
            'scream_score': 0.0, 
            'silence_score': 0.0, 
            'noise_score': 0.0, 
            'music_score': 0.0, 
            'hum_score': 0.0, 
            'speech_score': 0.0, 
            'tension_score': 0.0, 
            'has_audio_track': False, 
            'horror_label_threshold_based': 'Error_Processing',  
            'horror_level_threshold_based': -1  
        }

        try:
            trim_video(video_original_path, trimmed_video_path, duration_seconds=10) 
            if not os.path.exists(trimmed_video_path):
                raise FileNotFoundError(f"File trimmed is not created: {trimmed_video_path}") 

            features = extract_horror_features(trimmed_video_path) 
            row_data.update(features) 

            final_label, horror_level = assign_final_horror_label(features) 
            row_data['horror_label_threshold_based'] = final_label 
            row_data['horror_level_threshold_based'] = horror_level 
            

        except FileNotFoundError as fnfe:
            print(f"Skip {video_id} of {dataset_name} because of File Not Found: {fnfe}") 
            row_data['horror_label_threshold_based'] = 'Error_File_Missing' 
        except Exception as e:
            print(f"Skip {video_id} của {dataset_name}: {e}") 
            row_data['horror_label_threshold_based'] = 'Error_Processing' 
        finally:
            results_for_csv.append(row_data) 
            delete_temp_file_with_retries(trimmed_video_path)
            gc.collect() 

    output_csv_path = os.path.join(project_root_dir, 'relabeled_data', f"{output_csv_filename_prefix}_{dataset_name}.csv") 
    df = pd.DataFrame(results_for_csv) 
    df.to_csv(output_csv_path, index=False, encoding='utf-8') 
    
    print(f"\nProcessed and saved features with labels for {dataset_name} at: {output_csv_path}") 

    return results_for_csv