import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold 
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import joblib 
import numpy as np

def train_models(X_train_scaled, y_train_encoded, X_val_scaled, y_val_encoded, label_encoder):
      
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVC': SVC(random_state=42, class_weight='balanced', probability=True),
    }

    param_grids = {
        'Logistic Regression': {
            'C': [0.1, 1, 10], 
            'solver': ['liblinear', 'lbfgs']
        },
        'Random Forest': {
            'n_estimators': [10, 20, 30, 50, 100, 200],
            'max_depth': [None, 2, 5, 7, 10, 20],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 5, 10]
        },
        'Gradient Boosting': {
            'n_estimators': [10, 20, 30, 50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [None, 2, 3, 5, 7, 10, 20]
        },
        'SVC': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        }
    }

    best_model = None
    best_f1_weighted = 0.0
    best_model_name = ""

    for name, model in models.items():
        
        cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) 

        if name in param_grids:
            grid_search = GridSearchCV(model, param_grids[name], cv=cv_splitter, scoring='f1_weighted', n_jobs=-1, verbose=1)
            grid_search.fit(X_train_scaled, y_train_encoded) 
            
            current_model = grid_search.best_estimator_
            print(f"Best parameters for {name}: {grid_search.best_params_}")
        else:  
            current_model = model
            current_model.fit(X_train_scaled, y_train_encoded) 
        
        print(f"\nEvaluating {name} on Validation Set:")
        y_val_pred = current_model.predict(X_val_scaled)

        val_f1_weighted = f1_score(y_val_encoded, y_val_pred, average='weighted')
        print(f"F1-score Weighted (Validation - {name}): {val_f1_weighted:.4f}")
        
        val_accuracy = accuracy_score(y_val_encoded, y_val_pred)
        print(f"Accuracy (Validation - {name}): {val_accuracy:.4f}")
        print("Classification Report (Validation - {}):\n{}".format(name, classification_report(y_val_encoded, y_val_pred, target_names=label_encoder.classes_)))

        if val_f1_weighted > best_f1_weighted: 
            best_f1_weighted = val_f1_weighted
            best_model = current_model
            best_model_name = name
            
    print(f"\nBest model on Validation set: {best_model_name} with F1-weighted score: {best_f1_weighted:.4f}")

    return best_model, best_model_name, best_f1_weighted