import json
from typing import Dict
import argparse
import statistics

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

def load_gold_data_with_choices(ground_truth_filepath: str = '../data/dev.json') -> dict:
    """
    Load gold data with choices for standard deviation calculation.
    """
    gt_data = load_json_or_jsonl(ground_truth_filepath)
    return gt_data

def is_within_standard_deviation(prediction: int, choices: list) -> bool:
    """
    Check if prediction is within standard deviation of gold labels.
    Based on scoring.py logic:
    - Prediction is within (avg - stdev) < prediction < (avg + stdev)
    - OR abs(avg - prediction) < 1
    """
    if not choices or len(choices) == 0:
        return False
    
    avg = statistics.mean(choices)
    stdev = statistics.stdev(choices) if len(choices) > 1 else 0.0
    
    # Is prediction within the range of the average +/- the standard deviation?
    if (avg - stdev) < prediction < (avg + stdev):
        return True
    
    # Is the distance between average and prediction less than one?
    if abs(avg - prediction) < 1:
        return True
    
    return False

def compute_metrics(pred_labels: list, true_labels: list, gold_data_with_choices: dict = None) -> dict:
    """
    Compute metrics.
    """
    pred_labels_sorted = labels_to_sorted_list(pred_labels)
    true_labels_sorted = labels_to_sorted_list(true_labels)

    pred_labels = [p['label'] for p in pred_labels_sorted]
    true_labels = [p['label'] for p in true_labels_sorted]

    mse = mean_squared_error(true_labels, pred_labels)
    mae = mean_absolute_error(true_labels, pred_labels)
    rmse = np.sqrt(mse)
    spearman = spearmanr(true_labels, pred_labels)
    accuracy = accuracy_score(true_labels, pred_labels)
    
    # Calculate accuracy within standard deviation if gold data with choices is provided
    accuracy_within_sd = None
    if gold_data_with_choices:
        within_sd_correct = 0
        within_sd_total = 0
        for pred_item, true_item in zip(pred_labels_sorted, true_labels_sorted):
            sample_id = pred_item['id']
            if sample_id in gold_data_with_choices:
                choices = gold_data_with_choices[sample_id].get('choices', [])
                if choices:
                    within_sd_total += 1
                    if is_within_standard_deviation(pred_item['label'], choices):
                        within_sd_correct += 1
        if within_sd_total > 0:
            accuracy_within_sd = within_sd_correct / within_sd_total
    
    result = {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'spearman': spearman,
        'accuracy': accuracy,
    }
    if accuracy_within_sd is not None:
        result['accuracy_within_sd'] = accuracy_within_sd
    
    return result

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
    dev_gold_data = load_gold_data_with_choices('../data/dev.json')
    if args.inference:
        dev_predictions_filepath = '../predictions/ultimate_predictions_dev_inference.jsonl'
    else:
        dev_predictions_filepath = '../predictions/ultimate_predictions_dev.jsonl'
    dev_predictions = load_json_or_jsonl(dev_predictions_filepath)
    dev_metrics = compute_metrics(dev_predictions, dev_true_labels, dev_gold_data)
    print("Dev Metrics:")
    print(f"  MSE: {dev_metrics['mse']:.4f}")
    print(f"  MAE: {dev_metrics['mae']:.4f}")
    print(f"  RMSE: {dev_metrics['rmse']:.4f}")
    print(f"  Spearman: {dev_metrics['spearman'].correlation:.4f} (p={dev_metrics['spearman'].pvalue:.4f})")
    print(f"  Accuracy: {dev_metrics['accuracy']:.4f}")
    if 'accuracy_within_sd' in dev_metrics:
        print(f"  Accuracy within SD: {dev_metrics['accuracy_within_sd']:.4f}")
    
    # Evaluate training data
    print("\n" + "="*60)
    print("EVALUATING TRAINING DATA")
    print("="*60)
    if args.augmented:
        train_gold_data = load_gold_data_with_choices('../data/train_augmented.json')
    elif args.augmented_v2:
        train_gold_data = load_gold_data_with_choices('../data/train_augmented_v2.json')
    else:
        train_gold_data = load_gold_data_with_choices('../data/train.json')
    if args.inference:
        train_predictions_filepath = '../predictions/ultimate_predictions_train_inference.jsonl'
    else:
        train_predictions_filepath = '../predictions/ultimate_predictions_train.jsonl'
    train_predictions = load_json_or_jsonl(train_predictions_filepath)
    train_metrics = compute_metrics(train_predictions, train_true_labels, train_gold_data)
    print("Train Metrics:")
    print(f"  MSE: {train_metrics['mse']:.4f}")
    print(f"  MAE: {train_metrics['mae']:.4f}")
    print(f"  RMSE: {train_metrics['rmse']:.4f}")
    print(f"  Spearman: {train_metrics['spearman'].correlation:.4f} (p={train_metrics['spearman'].pvalue:.4f})")
    print(f"  Accuracy: {train_metrics['accuracy']:.4f}")
    if 'accuracy_within_sd' in train_metrics:
        print(f"  Accuracy within SD: {train_metrics['accuracy_within_sd']:.4f}")


if __name__ == "__main__":
    main()
