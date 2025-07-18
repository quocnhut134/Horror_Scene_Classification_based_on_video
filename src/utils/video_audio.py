import os
import cv2
import numpy as np
from moviepy import VideoFileClip
import gc
import time

def trim_video(input_path, output_path, duration_seconds=10):
    clip = None
    trimmed_clip = None
    try:
        if os.path.exists(output_path):
            return

        if not os.path.exists(input_path): 
            raise FileNotFoundError(f"Root video is not found: {input_path}")

        clip = VideoFileClip(input_path) 
        
        actual_duration = min(duration_seconds, clip.duration) 
        
        if actual_duration <= 0:
            print(f"Warning: video is too short: {os.path.basename(input_path)}") 
            return 

        trimmed_clip = clip.subclipped(0, actual_duration) 
        trimmed_clip.write_videofile(output_path, codec='libx264', audio_codec='aac', fps=24, logger=None) 
        
    except Exception as e:
        print(f"Error when cutting video {os.path.basename(input_path)}: {e}") 
        if os.path.exists(output_path): # Remove error files
            try:
                os.remove(output_path) 
            except Exception as e_del:
                print(f"Error when removing error trimmed {os.path.basename(output_path)}: {e_del}") 
        raise 
    finally:
        if trimmed_clip is not None:
            trimmed_clip.close() 
            del trimmed_clip
        if clip is not None:
            clip.close() 
            del clip
        gc.collect()

def calculate_brightness_std(video_path, frame_skip=5):
    cap = None
    brightness_values = []
    try:
        cap = cv2.VideoCapture(video_path) 
        if not cap.isOpened():
            print(f"Cannot open file to calculate brightness: {os.path.basename(video_path)}") 
            return 0.0
        
        frame_idx = 0
        while True:
            ret, frame = cap.read() 
            if not ret:
                break
            
            if frame_idx % frame_skip == 0: 
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
                brightness = np.mean(gray_frame) 
                brightness_values.append(brightness) 
            frame_idx += 1
        
        return np.std(brightness_values) if len(brightness_values) > 1 else 0.0 
    except Exception as e:
        print(f"Error when calculating std of brightness for {os.path.basename(video_path)}: {e}") 
        return 0.0
    finally:
        if cap is not None and cap.isOpened():
            cap.release() 
            del cap
        gc.collect()

def delete_temp_file_with_retries(file_path, max_retries=3, delay_sec=1):
    for attempt in range(max_retries): 
        try:
            if os.path.exists(file_path):
                os.remove(file_path) 
            break
        except PermissionError as pe:
            print(f"Error PermissionError when deleting {os.path.basename(file_path)} (lần {attempt+1}/{max_retries}): {pe}") 
            if attempt < max_retries - 1:
                time.sleep(delay_sec) 
            else:
                print(f"Cannot delete {os.path.basename(file_path)} after {max_retries} times to try.") 
        except Exception as e:
            print(f"Unexpected error when deleting {os.path.basename(file_path)}: {e}") 
            break