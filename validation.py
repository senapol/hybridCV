from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import yaml

data_yaml_path = 'CustomTennis-4/data.yaml'

with open(data_yaml_path, 'r') as file:
    data_config = yaml.safe_load(file)

base_dir = os.path.join(os.getcwd(), 'CustomTennis-4')

data_config['val'] = os.path.join(base_dir, 'valid', 'images')
data_config['train'] = os.path.join(base_dir, 'train', 'images')
data_config['test'] = os.path.join(base_dir, 'test', 'images')

temp_data_yaml = 'temp_data.yaml'
with open(temp_data_yaml, 'w') as file:
    yaml.dump(data_config, file)

data_yaml = temp_data_yaml

models = {
    'Best': 'models/best.pt',
    'Last': 'models/last.pt',
    'Best2': 'models/best2.pt',
    'Last2': 'models/last2.pt',
    'Best3': 'models/best3.pt',
    'Last3': 'models/last3.pt'
}

def compare_all_models():
    results = {}
    
    for model_name, model_path in models.items():
        print(f"\nValidating {model_name} model...")
        
        if not os.path.exists(model_path):
            print(f"ERROR: Model file not found at {model_path}")
            continue
            
        try:
            model = YOLO(model_path)
            
            print(f"Running validation with data config: {data_yaml}")
            metrics = model.val(data=data_yaml)
            
            results[model_name] = {
                'mAP50': metrics.box.map50,
                'mAP50-95': metrics.box.map,
                'Precision': metrics.box.mp,
                'Recall': metrics.box.mr,
                'Inference Time (ms)': metrics.speed['inference']
            }
        except Exception as e:
            print(f"Error validating {model_name} model: {str(e)}")
            results[model_name] = {
                'mAP50': float('nan'),
                'mAP50-95': float('nan'),
                'Precision': float('nan'),
                'Recall': float('nan'),
                'Inference Time (ms)': float('nan')
            }
    
    df = pd.DataFrame(results).round(3)
    
    print("\nModel Comparison Results:")
    print("=" * 80)
    print(df)
    
    print("\nBest Model for Each Metric:")
    print("=" * 80)
    for metric in ['mAP50', 'mAP50-95', 'Precision', 'Recall', 'Inference Time (ms)']:
        if metric == 'Inference Time (ms)':
            best_model = df.loc[metric].idxmin()  # Lower is better for inference time
            best_value = df.loc[metric].min()
        else:
            best_model = df.loc[metric].idxmax()  # Higher is better for other metrics
            best_value = df.loc[metric].max()
        
        print(f"{metric}: {best_model} ({best_value:.3f})")
    
    print("\nPairwise Comparisons:")
    print("=" * 80)
    model_names = list(models.keys())
    for i in range(len(model_names)):
        for j in range(i+1, len(model_names)):
            model_a = model_names[i]
            model_b = model_names[j]
            
            print(f"\nComparing {model_a} vs {model_b}:")
            for metric in ['mAP50', 'mAP50-95', 'Precision', 'Recall']:
                diff = results[model_b][metric] - results[model_a][metric]
                better = f"{model_b} is better" if diff > 0 else f"{model_a} is better"
                print(f"  {metric}: Difference of {abs(diff):.3f} ({better})")
    
    return df

def visualize_comparison(df):
    """Create visualizations for model comparison"""
    df_viz = df.drop('Inference Time (ms)').transpose()
    
    plt.figure(figsize=(12, 8))
    df_viz.plot(kind='bar', figsize=(12, 8))
    plt.title('Model Performance Comparison (CustomTennis-4)')
    plt.ylabel('Score')
    plt.xlabel('Model')
    plt.legend(title='Metrics')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('model_comparison_custom.png')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_viz, annot=True, cmap='YlGnBu', fmt='.3f')
    plt.title('Model Performance Heatmap (CustomTennis-4)')
    plt.tight_layout()
    plt.savefig('model_heatmap_custom.png')
    
    plt.figure(figsize=(10, 6))
    inference_times = df.loc['Inference Time (ms)']
    inference_times.plot(kind='bar', color='skyblue')
    plt.title('Inference Time Comparison (CustomTennis-4)')
    plt.ylabel('Time (ms)')
    plt.xlabel('Model')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('inference_time_comparison_custom.png')
    
    print("Visualizations saved as 'model_comparison_custom.png', 'model_heatmap_custom.png', and 'inference_time_comparison_custom.png'")

def main():
    print("Starting model comparison on CustomTennis-4 dataset...")
    
    print(f"Using data configuration from: {data_yaml}")
    print("Validation data path:", data_config['val'])
    print("Training data path:", data_config['train'])
    print("Test data path:", data_config['test'])
    
    if not os.path.exists(data_config['val']):
        print(f"WARNING: Validation directory not found at: {data_config['val']}")
        print("Please verify your directory structure.")
        
        possible_val_paths = [
            os.path.join(base_dir, 'valid', 'images'),
            os.path.join(base_dir, 'validation', 'images'),
            os.path.join(base_dir, 'val', 'images')
        ]
        
        for path in possible_val_paths:
            if os.path.exists(path):
                print(f"Found validation directory at: {path}")
                data_config['val'] = path
                with open(temp_data_yaml, 'w') as file:
                    yaml.dump(data_config, file)
                print("Updated data.yaml with correct path")
                break
    
    df = compare_all_models()
    
    df.to_csv('model_comparison_customtennis4_results.csv')
    print("Results saved to model_comparison_customtennis4_results.csv")
    
    try:
        visualize_comparison(df)
    except ImportError:
        print("Matplotlib or Seaborn not available. Skipping visualization.")
    
    if os.path.exists(temp_data_yaml):
        os.remove(temp_data_yaml)
    
    print("Comparison complete!")

if __name__ == "__main__":
    main()