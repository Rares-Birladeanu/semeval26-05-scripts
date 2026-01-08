import json
from typing import Dict
import argparse

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from scipy.stats import spearmanr

def load_json_or_jsonl(filepath: str) -> Dict[str, dict]:
    """Load a JSON split file."""
    if filepath.endswith('.jsonl'):
        with open(filepath, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    else:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

def labels_to_dict(labels: list) -> dict:
    """
    Convert labels to a dictionary.
    """
    return {l['id']: l['label'] for l in labels}

def labels_to_sorted_list(labels: list) -> list:
    return sorted(labels, key=lambda x: x['id'])

def compute_true_labels_as_labels(ground_truth_filepath: str = '../data/dev.json') -> dict:
    """
    Compute true labels as labels.
    """
    gt_data = load_json_or_jsonl(ground_truth_filepath)
    true_labels = []
    for k, v in gt_data.items():
        average = v['average']
        rounded = round(average)
        true_labels.append({'id': k, 'label': rounded})
    return true_labels

def compute_metrics(pred_labels: list, true_labels: list) -> dict:
    """
    Compute metrics.
    """

    pred_labels = [p['label'] for p in labels_to_sorted_list(pred_labels)]
    true_labels = [p['label'] for p in labels_to_sorted_list(true_labels)]

    mse = mean_squared_error(true_labels, pred_labels)
    mae = mean_absolute_error(true_labels, pred_labels)
    rmse = np.sqrt(mse)
    spearman = spearmanr(true_labels, pred_labels)
    accuracy = accuracy_score(true_labels, pred_labels)
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'spearman': spearman,
        'accuracy': accuracy,
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate ultimate solution for Semeval 2026 Task 5")
    parser.add_argument("--augmented", action="store_true", help="Use augmented training data")
    parser.add_argument("--augmented_v2", action="store_true", help="Use augmented training data v2")
    parser.add_argument("--inference", action="store_true", help="Use inference data")
    args = parser.parse_args()

    if args.augmented:
        train_true_labels = compute_true_labels_as_labels('../data/train_augmented.json')
    elif args.augmented_v2:
        train_true_labels = compute_true_labels_as_labels('../data/train_augmented_v2.json')
    else:
        train_true_labels = compute_true_labels_as_labels('../data/train.json')

    # Evaluate dev data
    print("="*60)
    print("EVALUATING DEV DATA")
    print("="*60)
    dev_true_labels = compute_true_labels_as_labels('../data/dev.json')
    if args.inference:
        dev_predictions_filepath = '../predictions/ultimate_predictions_dev_inference.jsonl'
    else:
        dev_predictions_filepath = '../predictions/ultimate_predictions_dev.jsonl'
    dev_predictions = load_json_or_jsonl(dev_predictions_filepath)
    dev_metrics = compute_metrics(dev_predictions, dev_true_labels)
    print("Dev Metrics:")
    print(f"  MSE: {dev_metrics['mse']:.4f}")
    print(f"  MAE: {dev_metrics['mae']:.4f}")
    print(f"  RMSE: {dev_metrics['rmse']:.4f}")
    print(f"  Spearman: {dev_metrics['spearman'].correlation:.4f} (p={dev_metrics['spearman'].pvalue:.4f})")
    print(f"  Accuracy: {dev_metrics['accuracy']:.4f}")
    
    # Evaluate training data
    print("\n" + "="*60)
    print("EVALUATING TRAINING DATA")
    print("="*60)
    if args.inference:
        train_predictions_filepath = '../predictions/ultimate_predictions_train_inference.jsonl'
    else:
        train_predictions_filepath = '../predictions/ultimate_predictions_train.jsonl'
    train_predictions = load_json_or_jsonl(train_predictions_filepath)
    train_metrics = compute_metrics(train_predictions, train_true_labels)
    print("Train Metrics:")
    print(f"  MSE: {train_metrics['mse']:.4f}")
    print(f"  MAE: {train_metrics['mae']:.4f}")
    print(f"  RMSE: {train_metrics['rmse']:.4f}")
    print(f"  Spearman: {train_metrics['spearman'].correlation:.4f} (p={train_metrics['spearman'].pvalue:.4f})")
    print(f"  Accuracy: {train_metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
