import os
import tensorflow as tf
import torch
import gc

from src.models.loader import load_yolo_model, load_yamnet_model
from src.processing.pipeline import process_vidharm_dataset

def main():
    project_root_dir = os.getcwd() 
    print(f"root_dir: {project_root_dir}")

    print("\n--- Checking GPU ---")
    gpus_tf = tf.config.list_physical_devices('GPU') 
    if gpus_tf:
        print(f"TensorFlow found GPU: {gpus_tf}") 
        try:
            for gpu in gpus_tf:
                tf.config.experimental.set_memory_growth(gpu, True) 
            print("Completed GPU settings for TensorFlow") 
        except RuntimeError as e:
            print(f"Error GPU settings for TensorFlow: {e}") 
    else:
        print("TensorFlow could not find GPU, trying to use CPU") 
    
    if torch.cuda.is_available(): 
        print(f"PyTorch found GPU: {torch.cuda.get_device_name(0)} (Total GPU: {torch.cuda.device_count()})") 
        try:
            test_tensor = torch.randn(1).to('cuda') 
            print("PyTorch can use GPU.") 
            del test_tensor
        except Exception as e:
            print(f"PyTorch found GPU but cannot use beacause of some errors: {e}") 
            print("There are some problems with settings, downloading CUDA/cuDNN or conflicting between versions") 
    else:
        print("PyTorch could not find GPU, turning to use CPU.") 
    print("--- Completed checking GPU ---\n")

    # Load models
    load_yolo_model(project_root_dir)
    load_yamnet_model()
    
    vidharm_video_base_path = os.path.join(project_root_dir, 'vidharm', 'clips')
    
    datasets_to_process = {
        'train': os.path.join(project_root_dir, 'vidharm', 'vidharm_train.json'), 
        'val': os.path.join(project_root_dir, 'vidharm', 'vidharm_val.json'), 
        'test': os.path.join(project_root_dir, 'vidharm', 'vidharm_test.json') 
    }

    print("Extracting features and labeling data")

    for name, json_path in datasets_to_process.items(): 
        if os.path.exists(json_path): 
            process_vidharm_dataset(json_path, vidharm_video_base_path, dataset_name=name, project_root_dir=project_root_dir) 
        else:
            print(f"Warning: not found file JSON for {name} at {json_path}. Skip!") 
                
    gc.collect()

if __name__ == "__main__":
    main()