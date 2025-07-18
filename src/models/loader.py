import os
from ultralytics import YOLO
import tensorflow_hub as hub
import tensorflow as tf

model_yolo_cls = None
model_yamnet = None

def load_yolo_model(project_root_dir, model_name='yolov8n_violence_classification_model'):
    global model_yolo_cls
    TRAINED_YOLO_CLS_MODEL_PATH = os.path.join(
        project_root_dir, 'src', 'YoloV8', model_name, 'weights', 'best.pt'
    )
    print(f"Best parameters model YoloV8: {TRAINED_YOLO_CLS_MODEL_PATH}") 

    try:
        if os.path.exists(TRAINED_YOLO_CLS_MODEL_PATH): 
            model_yolo_cls = YOLO(TRAINED_YOLO_CLS_MODEL_PATH) 
            print(f"Trained YOLO Cls: {TRAINED_YOLO_CLS_MODEL_PATH}") 
            print(f"Classes for YOLO Cls: {model_yolo_cls.names}") 
            if not any(name.lower() == 'violence' for name in model_yolo_cls.names.values()): 
                print("Waring: Class violence is not found in YOLO Cls") 
        else:
            print(f"Error: YOLO Cls not found at {TRAINED_YOLO_CLS_MODEL_PATH}") 
    except Exception as e:
        print(f"Error when downloading YOLO Cls: {e}") 
    return model_yolo_cls

def load_yamnet_model():
    global model_yamnet
    try:
        model_yamnet = hub.load('https://tfhub.dev/google/yamnet/1') 
        print("Download YAMNet successfully") 
    except Exception as e:
        print(f"Error when downloading YAMNet: {e}") 
        model_yamnet = None
    return model_yamnet

def get_yolo_model():
    return model_yolo_cls

def get_yamnet_model():
    return model_yamnet