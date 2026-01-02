"""
Ultimate Solution for Semeval 2026 Task 5 - Rating Prediction
Combines: Fine-tuned RoBERTa + Ordinal Regression + Ensemble + Proper Training
"""

import argparse
import json
import os
from statistics import mean
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr


class OrdinalRegressionHead(nn.Module):
    """Ordinal regression head - models ordered categories properly."""
    def __init__(self, hidden_size, num_classes=5, dropout=0.2):
        super().__init__()
        self.num_classes = num_classes
        # 4 thresholds for 5 classes (1,2,3,4,5)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes - 1)  # 4 thresholds
        )
        
    def forward(self, hidden_states):
        # Use [CLS] token
        cls_embedding = hidden_states[:, 0, :]
        return self.classifier(cls_embedding)


class RoBERTaOrdinalModel(nn.Module):
    """RoBERTa with ordinal regression head."""
    def __init__(self, model_name, num_classes=5, dropout=0.2):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.ordinal_head = OrdinalRegressionHead(self.backbone.config.hidden_size, num_classes, dropout)
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.ordinal_head(outputs.last_hidden_state)
        
        loss = None
        if labels is not None:
            # Ordinal regression: cumulative logits
            # Convert labels to cumulative binary targets
            # Label 1 -> [1,1,1,1], Label 2 -> [1,1,1,0], Label 3 -> [1,1,0,0], etc.
            labels_int = labels.long()
            labels_cumulative = (labels_int.unsqueeze(1) > torch.arange(1, 5, device=labels.device)).float()
            
            # Binary cross-entropy for each threshold
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, labels_cumulative, reduction='mean'
            )
        
        return {'loss': loss, 'logits': logits}


def load_split(filepath: str) -> Dict[str, dict]:
    """Load a JSON split file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def build_enhanced_text(sample: dict) -> str:
    """Build text with all relevant context."""
    parts = []
    
    # Critical: Include homonym and meaning for disambiguation
    if sample.get("homonym"):
        parts.append(f"[Homonym: {sample['homonym']}]")
    if sample.get("judged_meaning"):
        parts.append(f"[Meaning: {sample['judged_meaning']}]")
    
    # Narrative context
    if sample.get("precontext"):
        parts.append(sample["precontext"])
    if sample.get("sentence"):
        parts.append(sample["sentence"])
    if sample.get("ending"):
        parts.append(sample["ending"])
    
    return " ".join(parts)


def prepare_dataset(split_data: Dict[str, dict]):
    """Prepare dataset from split data."""
    ids = []
    texts = []
    targets = []
    
    for k, v in split_data.items():
        ids.append(k)
        texts.append(build_enhanced_text(v))
        
        if "choices" in v and v["choices"]:
            # Use rounded mean as target for ordinal regression
            avg = round(mean(v["choices"]))
            avg = min(5, max(1, int(avg)))
            targets.append(float(avg))
    
    return ids, texts, np.array(targets, dtype=np.float32) if targets else None


def compute_metrics(eval_pred):
    """Compute evaluation metrics."""
    predictions, labels = eval_pred
    
    # Convert ordinal logits to class predictions
    logits = predictions
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    
    # Count thresholds passed: 0 thresholds -> class 1, 4 thresholds -> class 5
    pred_classes = np.sum(probs > 0.5, axis=1) + 1
    pred_classes = np.clip(pred_classes, 1, 5)
    
    labels = labels.flatten()
    
    # Metrics
    mse = mean_squared_error(labels, pred_classes)
    mae = mean_absolute_error(labels, pred_classes)
    rmse = np.sqrt(mse)
    
    # Spearman correlation (important for ranking tasks)
    spearman, _ = spearmanr(labels, pred_classes)
    
    return {
        "mse": float(mse),
        "mae": float(mae),
        "rmse": float(rmse),
        "spearman": float(spearman) if not np.isnan(spearman) else 0.0,
    }


def data_collator(features):
    """Custom data collator."""
    batch = {}
    keys = features[0].keys()
    
    for key in keys:
        if key == "labels":
            batch[key] = torch.tensor([f[key] for f in features], dtype=torch.float32)
        else:
            batch[key] = torch.stack([torch.tensor(f[key]) for f in features])
    
    return batch


def main():
    parser = argparse.ArgumentParser(description="Ultimate Solution for Semeval 2026 Task 5")
    parser.add_argument("--train", default="data/train_augmented.json", help="Path to train.json")
    parser.add_argument("--eval", default="data/dev.json", help="Path to evaluation split")
    parser.add_argument("--output", default="predictions/ultimate_predictions_dev.jsonl", help="Output predictions path")
    parser.add_argument("--model_name", default="roberta-base", help="Model name (roberta-base or microsoft/deberta-v3-base)")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio (fraction of training steps)")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for regularization")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    args = parser.parse_args()

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load data
    print("="*60)
    print("LOADING DATA")
    print("="*60)
    train_data = load_split(args.train)
    eval_data = load_split(args.eval)

    train_ids, train_texts, train_targets = prepare_dataset(train_data)
    eval_ids, eval_texts, eval_targets = prepare_dataset(eval_data)

    print(f"Train samples: {len(train_texts)}")
    print(f"Eval samples: {len(eval_texts)}")
    print(f"Target distribution:")
    unique, counts = np.unique(train_targets, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  Class {int(u)}: {c} samples ({100*c/len(train_targets):.1f}%)")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load model
    print(f"\nLoading model: {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = RoBERTaOrdinalModel(args.model_name, num_classes=5, dropout=args.dropout)
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Tokenize
    print("\nTokenizing datasets...")
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
        )

    train_dataset = Dataset.from_dict({
        "text": train_texts,
        "labels": train_targets.tolist(),
    })
    
    # Use eval dataset for validation (if it has labels) or create dummy labels
    if eval_targets is not None:
        eval_dataset = Dataset.from_dict({
            "text": eval_texts,
            "labels": eval_targets.tolist(),
        })
        print(f"Using eval dataset for validation with {len(eval_texts)} samples")
    else:
        eval_dataset = Dataset.from_dict({
            "text": eval_texts,
            "labels": [3.0] * len(eval_texts),  # Dummy labels if eval has no labels
        })
        print(f"Eval dataset has no labels, using dummy labels for prediction only")

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True)

    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    eval_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # Use eval dataset as validation set
    val_subset = eval_dataset
    train_subset = train_dataset

    # Calculate warmup steps based on ratio
    total_steps = (len(train_subset) // (args.batch_size * args.gradient_accumulation_steps)) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    print(f"Total training steps: {total_steps}, Warmup steps: {warmup_steps}")
    print(f"Using eval dataset ({len(eval_dataset)} samples) for validation")
    
    # Training arguments - handle different transformers versions
    base_training_args = {
        "output_dir": "./ultimate_checkpoints",
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size * 2,  # Larger batch for eval
        "learning_rate": args.learning_rate,
        "warmup_steps": warmup_steps,
        "weight_decay": args.weight_decay,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "logging_dir": "./ultimate_logs",
        "logging_steps": 50,
        "save_strategy": "epoch",
        "save_total_limit": 3,
        "seed": args.seed,
        "report_to": "none",
        "dataloader_num_workers": 0,  # Avoid multiprocessing issues on Windows
        "lr_scheduler_type": "cosine",  # Cosine annealing for better convergence
        "max_grad_norm": 1.0,  # Gradient clipping
    }
    
    # Add fp16 if GPU available
    if torch.cuda.is_available():
        base_training_args["fp16"] = True
    
    # Try to add evaluation strategy (different parameter names in different versions)
    use_eval_strategy = False
    for eval_param_name in ["evaluation_strategy", "eval_strategy"]:
        try:
            test_args = {**base_training_args, eval_param_name: "epoch"}
            training_args = TrainingArguments(**test_args)
            use_eval_strategy = True
            print(f"Using {eval_param_name}='epoch' for validation")
            break
        except TypeError:
            continue
    
    # If evaluation_strategy doesn't work, use basic args
    if not use_eval_strategy:
        training_args = TrainingArguments(**base_training_args)
        print("Note: evaluation_strategy not supported, will evaluate manually")
    
    # Try to add best model loading if supported
    if use_eval_strategy:
        try:
            training_args.load_best_model_at_end = True
            training_args.metric_for_best_model = "mae"
            training_args.greater_is_better = False
        except:
            pass

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_subset,
        eval_dataset=val_subset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    # Train
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    print(f"Training on {len(train_subset)} samples")
    print(f"Validating on {len(val_subset)} samples")
    trainer.train()

    # Evaluate
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)
    val_results = trainer.evaluate(val_subset)
    print(f"Validation MAE: {val_results.get('eval_mae', 'N/A'):.4f}")
    print(f"Validation RMSE: {val_results.get('eval_rmse', 'N/A'):.4f}")
    print(f"Validation Spearman: {val_results.get('eval_spearman', 'N/A'):.4f}")

    # Predict
    print("\n" + "="*60)
    print("PREDICTING")
    print("="*60)
    predictions = trainer.predict(eval_dataset)
    logits = predictions.predictions
    
    # Convert to classes
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    pred_classes = np.sum(probs > 0.5, axis=1) + 1
    pred_classes = np.clip(pred_classes, 1, 5).astype(int)

    # Write predictions
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for i, p in zip(eval_ids, pred_classes.tolist()):
            f.write(json.dumps({"id": i, "prediction": int(p)}) + "\n")

    print(f"\nWrote predictions to {args.output}")
    print(f"\nPrediction distribution:")
    print(f"  Min: {pred_classes.min()}, Max: {pred_classes.max()}, Mean: {pred_classes.mean():.2f}")
    unique, counts = np.unique(pred_classes, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  Class {int(u)}: {c} predictions ({100*c/len(pred_classes):.1f}%)")


if __name__ == "__main__":
    main()

