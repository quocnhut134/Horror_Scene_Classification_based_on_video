import os
import sys

current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.abspath(os.path.join(current_file_dir, '..', '..'))
sys.path.append(project_root_dir)

from src.train_model.data_loader import load_and_preprocess_data
from src.train_model.model_trainer import train_models
from src.train_model.model_evaluator import evaluate_model
from src.train_model.serialization_utils import save_model_artifacts


def main():

    print(f"Project Root Directory: {project_root_dir}")

    X_train_scaled, y_train_encoded, X_val_scaled, y_val_encoded, \
    X_test_scaled, y_test_encoded, scaler, label_encoder = \
        load_and_preprocess_data(project_root_dir)

    best_model, best_model_name, best_accuracy = \
        train_models(X_train_scaled, y_train_encoded, X_val_scaled, y_val_encoded, label_encoder)

    evaluate_model(best_model, X_test_scaled, y_test_encoded, label_encoder, best_model_name)

    save_model_artifacts(best_model, scaler, label_encoder, project_root_dir, best_model_name)

if __name__ == "__main__":
    main()