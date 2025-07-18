import os
from moviepy import VideoFileClip
import numpy as np
import cv2
import librosa

def get_video_clip(video_path):

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    try:
        clip = VideoFileClip(video_path)
        print(f"Video loaded: {video_path}")
        print(f"Duration: {clip.duration:.2f} seconds, FPS: {clip.fps:.2f}")
        return clip
    except Exception as e:
        print(f"Error loading video clip {video_path}: {e}")
        return None

def extract_video_segment(clip, start_time, end_time):

    return clip.subclipped(start_time, end_time)

def get_frames_from_segment(segment_clip, frames_per_second=1):

    frames = []
    for t in np.arange(0, segment_clip.duration, 1/frames_per_second):
        frame = segment_clip.get_frame(t)
        frames.append(frame)
    return frames

def get_audio_from_segment(segment_clip, sr=16000):

    try:
        audio_array = segment_clip.audio.to_soundarray(fps=sr)
        if audio_array.ndim > 1:
            audio_array = audio_array[:, 0]
        return audio_array, sr
    except Exception as e:
        print(f"Error extracting audio from segment: {e}")
        return np.array([]), sr