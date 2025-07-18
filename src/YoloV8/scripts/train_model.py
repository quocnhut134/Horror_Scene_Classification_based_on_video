import os
from ultralytics import YOLO 

def train_yolov8_classification_model(yolo_cls_dataset_root_path, project_root_dir):
    
    model = YOLO('yolov8n-cls.pt')

    results = model.train(
        data=yolo_cls_dataset_root_path, 
        epochs=200, 
        imgsz=224, 
        batch=32, 
        name='yolov8n_violence_classification_model', 
        project=project_root_dir, 
        exist_ok=False, 
        task='classify', 
        optimizer='AdamW', 
        lr0=0.001, 
        patience=20, 
        device='0'
    )

    final_trained_cls_model_path = os.path.join(
        project_root_dir, 'runs', 'classify', 'yolov8n_violence_classification_model', 'weights', 'best.pt'
    ) 
    print(f"Best trained model: {final_trained_cls_model_path}") 

    return final_trained_cls_model_path, results

if __name__ == "__main__":
    project_root_dir = os.getcwd() 
    yolo_cls_dataset_root_path = os.path.join(project_root_dir, 'yolo_classification_data_organized')
    
    if not os.path.exists(yolo_cls_dataset_root_path):
        print(f"Error: Cannot find dataset '{yolo_cls_dataset_root_path}', run data_preparation.py")
    else:
        best_model_path, training_results = train_yolov8_classification_model(yolo_cls_dataset_root_path, project_root_dir)
        print(f"Best model link: {best_model_path}")