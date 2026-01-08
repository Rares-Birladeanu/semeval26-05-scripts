"""
Inference script for loading a checkpoint and making predictions on dev and train data.
"""

import argparse
import json
import os
from typing import Dict, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from datasets import Dataset
from config import DATA_ROOT, PREDICTIONS_ROOT

from ultimate_solution import RoBERTaOrdinalModel, build_enhanced_text, prepare_dataset


def load_checkpoint(checkpoint_path: str, model_name: Optional[str], device: torch.device):
    """Load model and tokenizer from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}...")
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Try to load tokenizer from checkpoint first
    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        print(f"✓ Loaded tokenizer from checkpoint")
    except:
        # If tokenizer not in checkpoint, try to infer from config or use provided model_name
        config_path = os.path.join(checkpoint_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                # Try to get model name from config
                if '_name_or_path' in config:
                    inferred_model_name = config['_name_or_path']
                elif 'model_type' in config:
                    model_type = config['model_type']
                    if model_type == 'roberta':
                        inferred_model_name = 'roberta-base'
                    elif model_type == 'deberta-v2' or model_type == 'deberta-v3':
                        inferred_model_name = 'microsoft/deberta-v3-base'
                    else:
                        inferred_model_name = model_name or 'roberta-base'
                else:
                    inferred_model_name = model_name or 'roberta-base'
                
                print(f"Loading tokenizer from {inferred_model_name}...")
                tokenizer = AutoTokenizer.from_pretrained(inferred_model_name)
        else:
            if model_name:
                print(f"Loading tokenizer from {model_name}...")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
            else:
                raise ValueError("Could not load tokenizer. Please specify --model_name")
    
    # Load model
    print("Loading model...")
    
    # Try to infer model name from config if not provided
    if model_name is None:
        config_path = os.path.join(checkpoint_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                if '_name_or_path' in config:
                    model_name = config['_name_or_path']
                elif 'model_type' in config:
                    model_type = config['model_type']
                    if model_type == 'roberta':
                        model_name = 'roberta-base'
                    elif model_type == 'deberta-v2' or model_type == 'deberta-v3':
                        model_name = 'microsoft/deberta-v3-base'
                    else:
                        model_name = 'roberta-base'
                else:
                    model_name = 'roberta-base'
        else:
            model_name = 'roberta-base'
    
    print(f"Using model architecture: {model_name}")
    
    # Load model architecture
    model = RoBERTaOrdinalModel(model_name, num_classes=5, dropout=0.3)
    
    # Load weights - try different formats
    import glob
    state_dict_path = None
    
    # Try pytorch_model.bin first
    if os.path.exists(os.path.join(checkpoint_path, "pytorch_model.bin")):
        state_dict_path = os.path.join(checkpoint_path, "pytorch_model.bin")
    # Try safetensors
    elif glob.glob(os.path.join(checkpoint_path, "*.safetensors")):
        state_dict_path = glob.glob(os.path.join(checkpoint_path, "*.safetensors"))[0]
    else:
        raise FileNotFoundError(f"No model weights found in {checkpoint_path}")
    
    print(f"Loading weights from {os.path.basename(state_dict_path)}...")
    
    try:
        if state_dict_path.endswith('.safetensors'):
            try:
                from safetensors.torch import load_file
                state_dict = load_file(state_dict_path)
                model.load_state_dict(state_dict, strict=False)
            except ImportError:
                raise ImportError("safetensors not installed. Install with: pip install safetensors")
        else:
            checkpoint = torch.load(state_dict_path, map_location=device)
            if isinstance(checkpoint, dict):
                # Try different keys
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'], strict=False)
                elif 'model' in checkpoint:
                    model.load_state_dict(checkpoint['model'], strict=False)
                else:
                    # Assume the dict itself is the state dict
                    model.load_state_dict(checkpoint, strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
        
        model.to(device)
        model.eval()
        print(f"✓ Loaded model from checkpoint")
        
    except Exception as e:
        print(f"Error loading model weights: {e}")
        print("Attempting to load with Trainer...")
        # Fallback: try using Trainer's from_pretrained
        from transformers import Trainer
        try:
            model = RoBERTaOrdinalModel.from_pretrained(checkpoint_path)
            model.to(device)
            model.eval()
            print(f"✓ Loaded model using Trainer.from_pretrained")
        except:
            raise RuntimeError(f"Failed to load model from checkpoint: {e}")
    
    return model, tokenizer


def predict(model, tokenizer, texts, ids, device, batch_size=32, max_length=512):
    """Make predictions on texts."""
    print(f"Making predictions on {len(texts)} samples...")
    
    # Create dataset
    dataset = Dataset.from_dict({
        "text": texts,
        "id": ids,
    })
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
    
    dataset = dataset.map(tokenize_function, batched=True)
    dataset.set_format("torch", columns=["input_ids", "attention_mask"])
    
    # Predict
    predictions = []
    model.eval()
    
    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs['logits'].cpu().numpy()
            
            # Convert to classes (same logic as training script)
            probs = torch.sigmoid(torch.tensor(logits)).numpy()
            pred_classes = np.sum(probs > 0.5, axis=1) + 1
            pred_classes = np.clip(pred_classes, 1, 5).astype(int)
            
            predictions.extend(pred_classes.tolist())
            
            if (i + batch_size) % 100 == 0:
                print(f"  Processed {min(i + batch_size, len(dataset))}/{len(dataset)} samples...")
    
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Inference script for checkpoint predictions")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint directory")
    parser.add_argument("--model_name", default=None, help="Model name (auto-detected if not provided)")
    parser.add_argument("--dev", default=os.path.join(DATA_ROOT, "dev.json"), help="Path to dev.json")
    parser.add_argument("--train", default=os.path.join(DATA_ROOT, "train.json"), help="Path to train.json")
    parser.add_argument("--output_dir", default=PREDICTIONS_ROOT, help="Output directory for predictions")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--clean_format", action="store_true", default=True, help="Use clean format (default: True)")
    parser.add_argument("--structured_format", action="store_true", help="Use structured format (overrides clean_format)")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu, auto-detected if not provided)")
    args = parser.parse_args()
    
    # Determine format
    use_clean_format = not args.structured_format if args.structured_format else args.clean_format
    
    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load checkpoint
    model, tokenizer = load_checkpoint(args.checkpoint, args.model_name, device)
    
    # Load data
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    
    # Dev data
    print(f"Loading dev data from {args.dev}...")
    with open(args.dev, 'r', encoding='utf-8') as f:
        dev_data = json.load(f)
    dev_ids, dev_texts, _ = prepare_dataset(dev_data, clean_format=use_clean_format)
    print(f"Loaded {len(dev_texts)} dev samples")
    
    # Train data
    print(f"Loading train data from {args.train}...")
    with open(args.train, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    train_ids, train_texts, _ = prepare_dataset(train_data, clean_format=use_clean_format)
    print(f"Loaded {len(train_texts)} train samples")
    
    # Make predictions
    print("\n" + "="*60)
    print("MAKING PREDICTIONS")
    print("="*60)
    
    # Dev predictions
    print("\nPredicting on dev data...")
    dev_predictions = predict(model, tokenizer, dev_texts, dev_ids, device, 
                             batch_size=args.batch_size, max_length=args.max_length)
    
    # Train predictions
    print("\nPredicting on train data...")
    train_predictions = predict(model, tokenizer, train_texts, train_ids, device,
                               batch_size=args.batch_size, max_length=args.max_length)
    
    # Save predictions
    print("\n" + "="*60)
    print("SAVING PREDICTIONS")
    print("="*60)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save dev predictions
    dev_output_path = os.path.join(args.output_dir, "ultimate_predictions_dev_inference.jsonl")
    with open(dev_output_path, "w", encoding="utf-8") as f:
        for i, p in zip(dev_ids, dev_predictions):
            f.write(json.dumps({"id": i, "label": int(p)}) + "\n")
    print(f"✓ Saved dev predictions to {dev_output_path}")
    print(f"  Dev prediction distribution:")
    unique, counts = np.unique(dev_predictions, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"    Class {int(u)}: {c} predictions ({100*c/len(dev_predictions):.1f}%)")
    
    # Save train predictions
    train_output_path = os.path.join(args.output_dir, "ultimate_predictions_train_inference.jsonl")
    with open(train_output_path, "w", encoding="utf-8") as f:
        for i, p in zip(train_ids, train_predictions):
            f.write(json.dumps({"id": i, "label": int(p)}) + "\n")
    print(f"\n✓ Saved train predictions to {train_output_path}")
    print(f"  Train prediction distribution:")
    unique, counts = np.unique(train_predictions, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"    Class {int(u)}: {c} predictions ({100*c/len(train_predictions):.1f}%)")
    
    print("\n✓ Inference complete!")


if __name__ == "__main__":
    main()

