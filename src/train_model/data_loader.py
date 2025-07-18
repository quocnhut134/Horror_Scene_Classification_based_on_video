import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_preprocess_data(project_root_dir):
    
    train_csv_path = os.path.join(project_root_dir, 'relabeled_data', 'vidharm_horror_features_and_labels_train.csv')
    val_csv_path = os.path.join(project_root_dir, 'relabeled_data', 'vidharm_horror_features_and_labels_val.csv')
    test_csv_path = os.path.join(project_root_dir, 'relabeled_data', 'vidharm_horror_features_and_labels_test.csv')

    try:
        df_train = pd.read_csv(train_csv_path)
        df_val = pd.read_csv(val_csv_path)
        df_test = pd.read_csv(test_csv_path)
    except FileNotFoundError as e:
        print(f"Missing file: {e.filename}")
        exit(1) 

    valid_labels = ['Disturbing Visuals', 'Jump scare', 'Psychological Tension', 'Calm Neutral']
    df_train = df_train[df_train['horror_label_threshold_based'].isin(valid_labels)].copy()
    df_val = df_val[df_val['horror_label_threshold_based'].isin(valid_labels)].copy()
    df_test = df_test[df_test['horror_label_threshold_based'].isin(valid_labels)].copy()

    print(f"Number of accepted train samples: {len(df_train)}")
    print(f"Number of accepted val samples: {len(df_val)}")
    print(f"Number of accepted test samples: {len(df_test)}")

    features = [
        'violence_score',
        'brightness_std',
        'scream_score',
        'silence_score',
        'noise_score',
        'music_score',
        'hum_score',
        'speech_score',
        'tension_score'
    ]
    target_label = 'horror_label_threshold_based'

    X_train = df_train[features]
    y_train = df_train[target_label]

    X_val = df_val[features]
    y_val = df_val[target_label]

    X_test = df_test[features]
    y_test = df_test[target_label]

    label_encoder = LabelEncoder()
    all_unique_labels = sorted(pd.concat([y_train, y_val, y_test]).unique())
    label_encoder.fit(all_unique_labels)

    y_train_encoded = label_encoder.transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    y_test_encoded = label_encoder.transform(y_test)

    print("Label Mapping (String -> Number):", list(label_encoder.classes_))
    print("Label Mapping (Number -> String):", {i: label for i, label in enumerate(label_encoder.classes_)})

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print(f"Size of trainset (scaled): {X_train_scaled.shape}")
    print(f"Size of valset (scaled): {X_val_scaled.shape}")
    print(f"Size of testset (scaled): {X_test_scaled.shape}")

    return X_train_scaled, y_train_encoded, X_val_scaled, y_val_encoded, \
           X_test_scaled, y_test_encoded, scaler, label_encoder