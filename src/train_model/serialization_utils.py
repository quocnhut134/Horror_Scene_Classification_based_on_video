import os
import joblib

def save_model_artifacts(model, scaler, label_encoder, project_root_dir, model_name=""):

    save_dir = os.path.join(project_root_dir, 'src', 'train_model', 'saved_models')
    os.makedirs(save_dir, exist_ok=True)

    model_file_name = f'horror_classifier_{model_name.replace(" ", "_").lower()}.pkl' if model_name else 'horror_classifier_model.pkl'
    scaler_file_name = 'scaler.pkl'
    label_encoder_file_name = 'label_encoder.pkl'

    model_save_path = os.path.join(save_dir, model_file_name)
    scaler_save_path = os.path.join(save_dir, scaler_file_name)
    label_encoder_save_path = os.path.join(save_dir, label_encoder_file_name)

    if model is not None:
        joblib.dump(model, model_save_path)
        joblib.dump(scaler, scaler_save_path)
        joblib.dump(label_encoder, label_encoder_save_path)
        print(f"\nModel ({model_name}), Scaler, and LabelEncoder have been saved to: {save_dir}")
    else:
        print("\nNo best model to save.")

def load_model_artifacts(project_root_dir, model_name=""):

    save_dir = os.path.join(project_root_dir, 'src', 'train_model', 'saved_models')
    
    model_file_name = f'horror_classifier_{model_name.replace(" ", "_").lower()}.pkl' if model_name else 'horror_classifier_model.pkl'
    scaler_file_name = 'scaler.pkl'
    label_encoder_file_name = 'label_encoder.pkl'

    model_load_path = os.path.join(save_dir, model_file_name)
    scaler_load_path = os.path.join(save_dir, scaler_file_name)
    label_encoder_load_path = os.path.join(save_dir, label_encoder_file_name)

    model = None
    scaler = None
    label_encoder = None

    try:
        if os.path.exists(model_load_path):
            model = joblib.load(model_load_path)
            print(f"Model loaded from {model_load_path}")
        else:
            print(f"Model file not found at {model_load_path}")

        if os.path.exists(scaler_load_path):
            scaler = joblib.load(scaler_load_path)
            print(f"Scaler loaded from {scaler_load_path}")
        else:
            print(f"Scaler file not found at {scaler_load_path}")

        if os.path.exists(label_encoder_load_path):
            label_encoder = joblib.load(label_encoder_load_path)
            print(f"LabelEncoder loaded from {label_encoder_load_path}")
        else:
            print(f"LabelEncoder file not found at {label_encoder_load_path}")

    except Exception as e:
        print(f"Error loading model artifacts: {e}")
        model, scaler, label_encoder = None, None, None

    return model, scaler, label_encoder