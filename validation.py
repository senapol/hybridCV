from ultralytics import YOLO
import pandas as pd

data_yaml = 'training/tennis-ball-detection-6/data.yaml'

models = {
    'Best': 'models/best.pt',
    'Last': 'models/last.pt',
    'Best2': 'models/best2.pt',
    'Last2': 'models/last2.pt'
}


def compare_models():
    model1 = YOLO('models/best2.pt')  # load an official model
    model2 = YOLO('models/last2.pt')  # load a custom model

    results = {}
    print("\nValidating First model...")
    # Validate the model
    first_metrics = model1.val(data=data_yaml)  # no arguments needed, dataset and settings remembered
    results['First'] = {
        'mAP50': first_metrics.box.map50,
        'mAP50-95': first_metrics.box.map,
        'Precision': first_metrics.box.mp,
        'Recall': first_metrics.box.mr,
        'Inference Time (ms)': first_metrics.speed['inference']
    }

    print("\nValidating Second Model...")
    # Validate the model
    second_metrics = model2.val(data=data_yaml)
    results['Second'] = {
        'mAP50': second_metrics.box.map50,
        'mAP50-95': second_metrics.box.map,
        'Precision': second_metrics.box.mp,
        'Recall': second_metrics.box.mr,
        'Inference Time (ms)': second_metrics.speed['inference']
    }

    # Create comparison DataFrame
    df = pd.DataFrame(results).round(3)

    print("\nModel Comparison Results:")
    print("=" * 60)
    print(df)

    print("\nDetailed Analysis:")
    print("=" * 50)
    for metric in ['mAP50', 'mAP50-95', 'Precision', 'Recall']:
        diff = results['Second'][metric] - results['First'][metric]
        better = "better" if diff > 0 else "worse"
        print(f"{metric}: Custom model is {abs(diff):.3f} {better} than YOLOv8x")
    
    return results

def main():
    results = compare_models()

if __name__ == "__main__":
    main()
    