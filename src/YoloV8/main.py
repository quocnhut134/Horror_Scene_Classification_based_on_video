import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

import gpu_check
import data_preparation
import train_model
import visualize_results

def main():
    project_root_dir = os.getcwd()
    
    gpu_check.check_gpu_availability()

    yolo_cls_dataset_root_path, data_yaml_file, classes = data_preparation.prepare_yolo_classification_data(project_root_dir)
    print(f"Dataset: {yolo_cls_dataset_root_path}")
    print(f"File data.yaml: {data_yaml_file}")

    final_trained_model_path, training_results_obj = train_model.train_yolov8_classification_model(
        yolo_cls_dataset_root_path, project_root_dir
    )
    print(f"Best trained model at: {final_trained_model_path}")

    visualize_results.visualize_training_results(project_root_dir, model_name='yolov8n_violence_classification_model')

if __name__ == "__main__":
    main()