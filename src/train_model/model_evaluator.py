import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def evaluate_model(model, X_test_scaled, y_test_encoded, label_encoder, model_name):
    if model is None:
        print("No model was trained successfully.")
        return

    print("\n=== Evaluation on Test Set ===")
    y_test_pred = model.predict(X_test_scaled)

    print(f"\nClassification Report for the Final Model ({model_name} on Test Set):")
    print(classification_report(y_test_encoded, y_test_pred, target_names=label_encoder.classes_, zero_division=0))
    
    final_accuracy = accuracy_score(y_test_encoded, y_test_pred)
    print(f"FINAL Accuracy (Test Set - {model_name}): {final_accuracy:.4f}")

    print("\nConfusion Matrix (Test Set):")
    cm_df = pd.DataFrame(confusion_matrix(y_test_encoded, y_test_pred),
                         index=[f'Actual {l}' for l in label_encoder.classes_],
                         columns=[f'Predicted {l}' for l in label_encoder.classes_])
    print(cm_df)