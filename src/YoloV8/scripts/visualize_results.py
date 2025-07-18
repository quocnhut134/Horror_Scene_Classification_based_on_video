import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_training_results(project_root_dir, model_name='yolov8n_violence_classification_model'):
    results_dir = os.path.join(project_root_dir, model_name)
    results_csv_path = os.path.join(results_dir, 'results.csv')

    if os.path.exists(results_csv_path):
        df_results = pd.read_csv(results_csv_path, comment='#')

        df_results.columns = [col.strip() for col in df_results.columns] 

        df_results['epoch'] = df_results['epoch'].astype(int)

        plt.figure(figsize=(16, 8))

        # Loss
        plt.subplot(1, 2, 1) 
        plt.plot(df_results['epoch'], df_results['train/loss'], label='Training Loss')
        plt.plot(df_results['epoch'], df_results['val/loss'], label='Validation Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(visible=True, which='both', color='#999999', linestyle='-', alpha=0.2)


        # Accuracy
        plt.subplot(1, 2, 2) 
        plt.plot(df_results['epoch'], df_results['metrics/accuracy_top1'], label='Top-1 Accuracy')
        plt.plot(df_results['epoch'], df_results['metrics/accuracy_top5'], label='Top-5 Accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(visible=True, which='both', color='#999999', linestyle='-', alpha=0.2)

        plt.tight_layout()
        plt.show()
        
        print("\nView graphs and confusion matrix at:")
        print(results_dir)

    else:
        print(f"File results.csv not found: {results_csv_path}")

if __name__ == "__main__":
    project_root_dir = os.getcwd() 
    visualize_training_results(project_root_dir)