"""
Script to calculate Spearman correlation when using rounded means as predictions
Shows the baseline performance if we just predict the rounded mean for each instance
"""

import json
from statistics import mean
from typing import Dict

import numpy as np
from scipy.stats import spearmanr


def load_split(filepath: str) -> Dict[str, dict]:
    """Load a JSON split file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_rounded_mean(choices: list) -> int:
    """Calculate rounded mean of choices, clipped to 1-5."""
    avg = round(mean(choices))
    return min(5, max(1, int(avg)))


def main():
    # Load validation data (dev set)
    print("Loading validation data...")
    dev_data = load_split("data/dev.json")
    
    # Check if dev set has labels
    has_labels = any("choices" in v and v["choices"] for v in dev_data.values())
    
    if not has_labels:
        print("Dev set doesn't have labels. Using train set with validation split...")
        train_data = load_split("data/train.json")
        
        # Create 90/10 split
        items = list(train_data.items())
        np.random.seed(42)
        np.random.shuffle(items)
        
        split_idx = int(0.9 * len(items))
        train_items = dict(items[:split_idx])
        val_items = dict(items[split_idx:])
        
        print(f"Using {len(val_items)} samples for validation")
        data_to_use = val_items
    else:
        print(f"Using dev set with {len(dev_data)} samples")
        data_to_use = dev_data
    
    # Calculate predictions (rounded means) and actual labels
    predictions = []
    actuals = []
    sample_info = []
    
    print("\nCalculating rounded means for each instance...")
    for k, v in data_to_use.items():
        if "choices" in v and v["choices"]:
            # Prediction: rounded mean
            pred = calculate_rounded_mean(v["choices"])
            predictions.append(pred)
            
            # Actual: also rounded mean (what we're comparing against)
            actual = calculate_rounded_mean(v["choices"])
            actuals.append(actual)
            
            sample_info.append({
                "id": k,
                "choices": v["choices"],
                "mean": mean(v["choices"]),
                "rounded_mean": actual,
                "prediction": pred
            })
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculate Spearman correlation
    spearman_corr, p_value = spearmanr(predictions, actuals)
    
    # Also calculate other metrics
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS: Using Rounded Means as Predictions")
    print("="*60)
    print(f"Number of samples: {len(predictions)}")
    print(f"\nSpearman Correlation: {spearman_corr:.4f}")
    print(f"P-value: {p_value:.6f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    # Distribution analysis
    print(f"\nPrediction distribution:")
    unique_pred, counts_pred = np.unique(predictions, return_counts=True)
    for u, c in zip(unique_pred, counts_pred):
        print(f"  Class {int(u)}: {c} ({100*c/len(predictions):.1f}%)")
    
    print(f"\nActual distribution:")
    unique_actual, counts_actual = np.unique(actuals, return_counts=True)
    for u, c in zip(unique_actual, counts_actual):
        print(f"  Class {int(u)}: {c} ({100*c/len(actuals):.1f}%)")
    
    # Show some examples
    print(f"\nSample examples (first 10):")
    print("-" * 60)
    for i, info in enumerate(sample_info[:10]):
        print(f"ID: {info['id']}")
        print(f"  Choices: {info['choices']}")
        print(f"  Mean: {info['mean']:.2f}")
        print(f"  Rounded mean (prediction): {info['rounded_mean']}")
        print()
    
    # Interpretation
    print("="*60)
    print("INTERPRETATION:")
    print("="*60)
    if spearman_corr == 1.0:
        print("Perfect correlation (1.0) - all predictions match actuals exactly")
    elif spearman_corr > 0.8:
        print(f"Very strong correlation ({spearman_corr:.4f}) - excellent baseline")
    elif spearman_corr > 0.6:
        print(f"Strong correlation ({spearman_corr:.4f}) - good baseline")
    elif spearman_corr > 0.4:
        print(f"Moderate correlation ({spearman_corr:.4f}) - decent baseline")
    else:
        print(f"Weak correlation ({spearman_corr:.4f}) - poor baseline")
    
    print(f"\nNote: This is the 'perfect' baseline - if you just predict")
    print(f"the rounded mean of each sample's choices, this is the best")
    print(f"you can do. Your model should beat this!")


if __name__ == "__main__":
    main()

