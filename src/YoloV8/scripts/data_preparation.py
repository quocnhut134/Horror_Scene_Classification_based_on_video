import os
import cv2 
import shutil
from sklearn.model_selection import train_test_split 
from tqdm.notebook import tqdm 
import glob

def prepare_yolo_classification_data(project_root_dir, dataset_name='violence_dataset'):

    source_images_base_path = os.path.join(project_root_dir, dataset_name)
    
    if not os.path.exists(source_images_base_path): 
        raise FileNotFoundError(f"Error: Cannot find dataset at {source_images_base_path}") 

    print(f"Root dataset:{source_images_base_path}") 

    all_image_paths_with_labels = [] 

    label_folders = ['violence', 'non_violence']

    for label_name_raw in tqdm(label_folders, desc="Collecting images"): 
        label_name_standard = label_name_raw.lower().replace(' ', '_') 
        current_label_folder_path = os.path.join(source_images_base_path, label_name_raw) 

        if not os.path.exists(current_label_folder_path): 
            print(f"Warning: cannot find folder at '{current_label_folder_path}', Skipping...") 
            continue

        image_files = glob.glob(os.path.join(current_label_folder_path, '*.jpg')) + \
                      glob.glob(os.path.join(current_label_folder_path, '*.jpeg')) + \
                      glob.glob(os.path.join(current_label_folder_path, '*.png')) 

        if not image_files: 
            print(f"Warning: Cannot find images in '{current_label_folder_path}'.")
            continue

        for img_path in image_files: 
            all_image_paths_with_labels.append((img_path, label_name_standard)) 

    print(f"\nTotal of found images: {len(all_image_paths_with_labels)}") 

    yolo_cls_dataset_root_path = os.path.join(project_root_dir, 'yolo_classification_data_organized')
    os.makedirs(yolo_cls_dataset_root_path, exist_ok=True)

    train_data_dir = os.path.join(yolo_cls_dataset_root_path, 'train')
    val_data_dir = os.path.join(yolo_cls_dataset_root_path, 'val')

    unique_class_names = sorted(list(set([label for _, label in all_image_paths_with_labels]))) 
    print(f"Found layers: {unique_class_names}") 

    for cls_name in unique_class_names: 
        os.makedirs(os.path.join(train_data_dir, cls_name), exist_ok=True) 
        os.makedirs(os.path.join(val_data_dir, cls_name), exist_ok=True) 

    image_paths_list = [info[0] for info in all_image_paths_with_labels] 
    labels_list = [info[1] for info in all_image_paths_with_labels] 

    train_image_paths, val_image_paths, train_labels, val_labels = train_test_split(
        image_paths_list, labels_list, test_size=0.2, random_state=42, stratify=labels_list 
    )

    print(f"Train images: {len(train_image_paths)}") 
    print(f"Val images: {len(val_image_paths)}") 

    for img_src_path, label in tqdm(zip(train_image_paths, train_labels), total=len(train_image_paths), desc="Copying Train data"): 
        shutil.copy(img_src_path, os.path.join(train_data_dir, label, os.path.basename(img_src_path))) 

    for img_src_path, label in tqdm(zip(val_image_paths, val_labels), total=len(val_image_paths), desc="Copying Val data"): 
        shutil.copy(img_src_path, os.path.join(val_data_dir, label, os.path.basename(img_src_path))) 

    # Create file data.yaml
    data_yaml_content = f"""
    # Root link to cls_dataset: {yolo_cls_dataset_root_path}

    # Name of layers:
    0: non_violence
    1: violence
    """ 

    data_yaml_path = os.path.join(project_root_dir, 'classification_data.yaml') 
    with open(data_yaml_path, 'w', encoding='utf-8') as f: 
        f.write(data_yaml_content) 

    print(f"\nCreated `data.yaml` at: {data_yaml_path}") 
    print("data.yaml content:") 
    print(data_yaml_content) 

    return yolo_cls_dataset_root_path, data_yaml_path, unique_class_names

if __name__ == "__main__":
    project_root_dir = os.getcwd() 
    yolo_dataset_path, data_yaml_file, classes = prepare_yolo_classification_data(project_root_dir)
    print(f"Dataset YOLO: {yolo_dataset_path}")
    print(f"File data.yaml: {data_yaml_file}")
    print(f"Layers: {classes}")