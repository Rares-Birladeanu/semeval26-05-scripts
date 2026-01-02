"""
Comprehensive evaluation script for Semeval 2026 Task 5 predictions
Computes multiple metrics: Spearman, MAE, RMSE, Accuracy, and more
"""

import sys
import os
import json
import statistics
from collections import Counter
from typing import Dict, List

import numpy as np
from scipy.stats import spearmanr, pearsonr

from format_check import check_formatting


def get_standard_deviation(l):
    """Calculate standard deviation."""
    return statistics.stdev(l) if len(l) > 1 else 0.0


def get_average(l):
    """Calculate average."""
    return sum(l) / len(l) if l else 0.0


def is_within_standard_deviation(prediction, labels):
    """Check if prediction is within standard deviation of average."""
    avg = get_average(labels)
    stdev = get_standard_deviation(labels)

    # Is prediction within the range of the average +/- the standard deviation?
    if (avg - stdev) < prediction < (avg + stdev):
        return True

    # Is the distance between average and prediction less than one?
    if abs(avg - prediction) < 1:
        return True

    return False


def load_predictions(predictions_filepath: str) -> Dict[str, int]:
    """Load predictions from JSONL file."""
    predictions = {}
    with open(predictions_filepath, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            predictions[str(data["id"])] = int(data["prediction"])
    return predictions


def compute_comprehensive_metrics(predictions_filepath: str, gold_data: dict):
    """
    Compute comprehensive evaluation metrics.
    """
    predictions = load_predictions(predictions_filepath)
    
    # Collect data
    gold_means = []
    gold_rounded_means = []
    pred_values = []
    gold_choices = []
    pred_rounded = []
    
    # Per-class metrics
    class_correct = {i: 0 for i in range(1, 6)}
    class_total = {i: 0 for i in range(1, 6)}
    within_sd_correct = 0
    within_sd_total = 0
    exact_match = 0
    
    for sample_id, sample_data in gold_data.items():
        if sample_id not in predictions:
            continue
            
        if "choices" not in sample_data or not sample_data["choices"]:
            continue
        
        choices = sample_data["choices"]
        pred = predictions[sample_id]
        
        # Calculate gold metrics
        gold_mean = get_average(choices)
        gold_rounded = round(gold_mean)
        gold_rounded = min(5, max(1, gold_rounded))
        
        gold_means.append(gold_mean)
        gold_rounded_means.append(gold_rounded)
        pred_values.append(pred)
        gold_choices.append(choices)
        pred_rounded.append(pred)
        
        # Per-class accuracy
        class_total[gold_rounded] += 1
        if pred == gold_rounded:
            class_correct[gold_rounded] += 1
            exact_match += 1
        
        # Within standard deviation
        within_sd_total += 1
        if is_within_standard_deviation(pred, choices):
            within_sd_correct += 1
    
    # Convert to numpy arrays
    gold_means = np.array(gold_means)
    gold_rounded_means = np.array(gold_rounded_means)
    pred_values = np.array(pred_values)
    
    # ========== CORRELATION METRICS ==========
    print("\n" + "="*60)
    print("CORRELATION METRICS")
    print("="*60)
    
    # Spearman correlation (with continuous means)
    spearman_corr, spearman_p = spearmanr(pred_values, gold_means)
    print(f"Spearman Correlation (vs continuous means): {spearman_corr:.4f} (p={spearman_p:.6f})")
    
    # Spearman correlation (with rounded means)
    spearman_rounded_corr, spearman_rounded_p = spearmanr(pred_values, gold_rounded_means)
    print(f"Spearman Correlation (vs rounded means): {spearman_rounded_corr:.4f} (p={spearman_rounded_p:.6f})")
    
    # Pearson correlation
    pearson_corr, pearson_p = pearsonr(pred_values, gold_means)
    print(f"Pearson Correlation: {pearson_corr:.4f} (p={pearson_p:.6f})")
    
    # ========== REGRESSION METRICS ==========
    print("\n" + "="*60)
    print("REGRESSION METRICS")
    print("="*60)
    
    # MAE (Mean Absolute Error)
    mae_continuous = np.mean(np.abs(pred_values - gold_means))
    mae_rounded = np.mean(np.abs(pred_values - gold_rounded_means))
    print(f"MAE (vs continuous means): {mae_continuous:.4f}")
    print(f"MAE (vs rounded means): {mae_rounded:.4f}")
    
    # RMSE (Root Mean Squared Error)
    rmse_continuous = np.sqrt(np.mean((pred_values - gold_means) ** 2))
    rmse_rounded = np.sqrt(np.mean((pred_values - gold_rounded_means) ** 2))
    print(f"RMSE (vs continuous means): {rmse_continuous:.4f}")
    print(f"RMSE (vs rounded means): {rmse_rounded:.4f}")
    
    # ========== ACCURACY METRICS ==========
    print("\n" + "="*60)
    print("ACCURACY METRICS")
    print("="*60)
    
    # Exact match accuracy
    n_samples = len(pred_values)
    exact_accuracy = exact_match / n_samples if n_samples > 0 else 0.0
    print(f"Exact Match Accuracy: {exact_accuracy:.4f} ({exact_match}/{n_samples})")
    
    # Within 1 point accuracy
    within_one = np.sum(np.abs(pred_values - gold_rounded_means) <= 1)
    within_one_accuracy = within_one / n_samples if n_samples > 0 else 0.0
    print(f"Within 1 Point Accuracy: {within_one_accuracy:.4f} ({within_one}/{n_samples})")
    
    # Within standard deviation accuracy
    within_sd_accuracy = within_sd_correct / within_sd_total if within_sd_total > 0 else 0.0
    print(f"Within Standard Deviation Accuracy: {within_sd_accuracy:.4f} ({within_sd_correct}/{within_sd_total})")
    
    # ========== PER-CLASS METRICS ==========
    print("\n" + "="*60)
    print("PER-CLASS ACCURACY")
    print("="*60)
    for class_label in range(1, 6):
        if class_total[class_label] > 0:
            accuracy = class_correct[class_label] / class_total[class_label]
            print(f"Class {class_label}: {accuracy:.4f} ({class_correct[class_label]}/{class_total[class_label]})")
        else:
            print(f"Class {class_label}: N/A (0 samples)")
    
    # ========== DISTRIBUTION ANALYSIS ==========
    print("\n" + "="*60)
    print("DISTRIBUTION ANALYSIS")
    print("="*60)
    
    pred_dist = Counter(pred_values)
    gold_dist = Counter(gold_rounded_means)
    
    n_samples = len(pred_values)
    print("Prediction Distribution:")
    for i in range(1, 6):
        count = pred_dist.get(i, 0)
        pct = 100 * count / n_samples if n_samples > 0 else 0
        print(f"  Class {i}: {count:4d} ({pct:5.1f}%)")
    
    print("\nGold Distribution (rounded means):")
    for i in range(1, 6):
        count = gold_dist.get(i, 0)
        pct = 100 * count / n_samples if n_samples > 0 else 0
        print(f"  Class {i}: {count:4d} ({pct:5.1f}%)")
    
    # ========== ERROR ANALYSIS ==========
    print("\n" + "="*60)
    print("ERROR ANALYSIS")
    print("="*60)
    
    errors = pred_values - gold_rounded_means
    print(f"Mean Error: {np.mean(errors):.4f}")
    print(f"Median Error: {np.median(errors):.4f}")
    print(f"Std Dev of Errors: {np.std(errors):.4f}")
    
    # Error distribution
    n_errors = len(errors)
    error_dist = Counter(errors.astype(int))
    print("\nError Distribution (prediction - gold):")
    for err in sorted(error_dist.keys()):
        count = error_dist[err]
        pct = 100 * count / n_errors if n_errors > 0 else 0
        print(f"  {err:+3d}: {count:4d} ({pct:5.1f}%)")
    
    # ========== SUMMARY ==========
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total Samples: {len(pred_values)}")
    print(f"Best Metric - Spearman: {spearman_corr:.4f}")
    print(f"Best Metric - MAE: {mae_rounded:.4f}")
    print(f"Best Metric - Exact Accuracy: {exact_accuracy:.4f}")
    
    # Return metrics dictionary
    return {
        "spearman_continuous": float(spearman_corr),
        "spearman_rounded": float(spearman_rounded_corr),
        "pearson": float(pearson_corr),
        "mae_continuous": float(mae_continuous),
        "mae_rounded": float(mae_rounded),
        "rmse_continuous": float(rmse_continuous),
        "rmse_rounded": float(rmse_rounded),
        "exact_accuracy": float(exact_accuracy),
        "within_one_accuracy": float(within_one_accuracy),
        "within_sd_accuracy": float(within_sd_accuracy),
        "total_samples": len(pred_values),
    }


if __name__ == "__main__":
    arguments = sys.argv
    if len(arguments) < 3:
        print("Usage: python evaluate.py <predictions_filepath> <set>")
        print("  predictions_filepath: Path to predictions .jsonl file")
        print("  set: The split to evaluate on (dev/test/train)")
        print("\nExample: python evaluate.py predictions/my_predictions.jsonl dev")
        sys.exit(1)

    predictions_filepath = arguments[1]
    if not os.path.exists(predictions_filepath):
        print(f"Error: Predictions file not found: {predictions_filepath}")
        sys.exit(1)

    testset = arguments[2]
    gold_filepath = f"data/{testset}.json"
    if not os.path.exists(gold_filepath):
        print(f"Error: Gold data file not found: {gold_filepath}")
        print("Make sure the set name (dev/test/train) is correct")
        sys.exit(1)

    try:
        with open(gold_filepath, "r", encoding="utf-8") as f:
            gold_data = json.load(f)
    except Exception as e:
        print(f"Error loading gold data: {e}")
        sys.exit(1)

    print(f"Evaluating: {predictions_filepath}")
    print(f"Gold data: {gold_filepath}")
    print(f"Gold samples: {len(gold_data)}")

    # check_formatting expects a list of dicts with "id" key
    # Convert dict format ({"0": {...}, "1": {...}}) to list format ([{"id": "0", ...}, {"id": "1", ...}])
    if isinstance(gold_data, dict):
        gold_data_list = [{"id": str(k)} for k in gold_data.keys()]
    else:
        gold_data_list = gold_data
    
    if not check_formatting(predictions_filepath, gold_data_list):
        sys.exit(1)

    print("\n" + "="*60)
    print("COMPREHENSIVE EVALUATION")
    print("="*60)
    
    metrics = compute_comprehensive_metrics(predictions_filepath, gold_data)
    
    # Optionally save metrics to file
    if len(arguments) >= 4:
        output_file = arguments[3]
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to: {output_file}")
