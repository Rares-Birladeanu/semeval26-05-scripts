"""
Improved Data Augmentation Strategy for Semeval 2026 Task 5

Key insight: Train and dev have ZERO homonym overlap. The model needs to learn
general patterns of coherence, not memorize homonym-specific patterns.

Strategies:
1. Cross-homonym augmentation: Swap homonyms/meanings while preserving narrative structure
2. Context variation: Vary precontext/sentence/ending while keeping same homonym-meaning-rating
3. Synthetic dev homonyms: Create training samples with dev homonyms
4. Rating-preserving transformations: Change surface form but preserve semantic coherence
"""

import argparse
import json
import random
import os
from typing import Dict, List, Tuple
import copy
from collections import defaultdict, Counter
from config import DATA_ROOT

# Try to import augmentation libraries
try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import nltk
    from nltk.corpus import wordnet
    from nltk.tokenize import word_tokenize
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False


def load_split(filepath: str) -> Dict[str, dict]:
    """Load a JSON split file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_split(data: Dict[str, dict], filepath: str):
    """Save data to JSON file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


class ImprovedDataAugmenter:
    """Improved augmentation focusing on cross-homonym generalization."""
    
    def __init__(self, train_data: Dict[str, dict], dev_data: Dict[str, dict]):
        self.train_data = train_data
        self.dev_data = dev_data
        
        # Extract homonym sets
        self.train_homonyms = set(v['homonym'] for v in train_data.values())
        self.dev_homonyms = set(v['homonym'] for v in dev_data.values())
        self.dev_only_homonyms = self.dev_homonyms - self.train_homonyms
        
        print(f"Train homonyms: {len(self.train_homonyms)}")
        print(f"Dev homonyms: {len(self.dev_homonyms)}")
        print(f"Dev-only homonyms: {len(self.dev_only_homonyms)}")
        
        # Group by homonym for easy lookup
        self.train_by_homonym = defaultdict(list)
        for k, v in train_data.items():
            self.train_by_homonym[v['homonym']].append((k, v))
        
        self.dev_by_homonym = defaultdict(list)
        for k, v in dev_data.items():
            self.dev_by_homonym[v['homonym']].append((k, v))
        
        # Group by rating for balanced augmentation
        self.train_by_rating = defaultdict(list)
        for k, v in train_data.items():
            rating = round(v['average'])
            self.train_by_rating[rating].append((k, v))
        
        # Initialize paraphrase model if available
        self.paraphrase_model = None
        if HAS_TRANSFORMERS:
            try:
                print("Loading paraphrase model...")
                self.paraphrase_model = pipeline(
                    "text2text-generation",
                    model="tuner007/pegasus_paraphrase",
                    device=-1  # CPU
                )
                print("✓ Paraphrase model loaded")
            except Exception as e:
                print(f"⚠ Could not load paraphrase model: {e}")
    
    def paraphrase_text(self, text: str, max_length=512):
        """Paraphrase text using a model."""
        if not self.paraphrase_model or not text.strip():
            return text
        
        try:
            if len(text) > max_length:
                text = text[:max_length]
            result = self.paraphrase_model(
                f"paraphrase: {text}",
                max_length=max_length,
                num_return_sequences=1,
                num_beams=5,
                temperature=0.7
            )
            return result[0]['generated_text']
        except Exception as e:
            return text
    
    def strategy_1_cross_homonym_swap(self, sample: dict, target_homonym: str, target_meaning: str) -> dict:
        """
        Strategy 1: Swap homonym/meaning while preserving narrative structure.
        This teaches the model that coherence patterns are homonym-agnostic.
        """
        augmented = copy.deepcopy(sample)
        augmented['homonym'] = target_homonym
        augmented['judged_meaning'] = target_meaning
        
        # Update sample_id
        if 'sample_id' in augmented:
            augmented['sample_id'] = f"{augmented['sample_id']}_cross_swap"
        
        return augmented
    
    def strategy_2_context_variation(self, sample: dict) -> dict:
        """
        Strategy 2: Vary context while keeping same homonym-meaning-rating.
        Paraphrase precontext/sentence/ending to create variations.
        """
        augmented = copy.deepcopy(sample)
        
        # Paraphrase different parts
        if sample.get('precontext') and random.random() < 0.7:
            augmented['precontext'] = self.paraphrase_text(sample['precontext'])
        
        if sample.get('sentence') and random.random() < 0.7:
            augmented['sentence'] = self.paraphrase_text(sample['sentence'])
        
        if sample.get('ending') and random.random() < 0.7 and sample.get('ending'):
            augmented['ending'] = self.paraphrase_text(sample['ending'])
        
        if 'sample_id' in augmented:
            augmented['sample_id'] = f"{augmented['sample_id']}_ctx_var"
        
        return augmented
    
    def strategy_3_synthetic_dev_homonyms(self, sample: dict) -> List[dict]:
        """
        Strategy 3: Create synthetic samples with dev-only homonyms.
        Find dev samples with similar rating/context structure, swap homonym.
        """
        augmented_samples = []
        rating = round(sample['average'])
        
        # Find dev samples with similar rating
        similar_dev_samples = [
            (k, v) for k, v in self.dev_data.items()
            if abs(round(v['average']) - rating) <= 1
            and v['homonym'] in self.dev_only_homonyms
        ]
        
        if not similar_dev_samples:
            return []
        
        # Pick 1-2 dev homonyms to create synthetic samples
        num_swaps = min(2, len(similar_dev_samples))
        selected = random.sample(similar_dev_samples, num_swaps)
        
        for dev_key, dev_sample in selected:
            synthetic = copy.deepcopy(sample)
            synthetic['homonym'] = dev_sample['homonym']
            synthetic['judged_meaning'] = dev_sample['judged_meaning']
            # Keep original rating - the coherence pattern should be similar
            if 'sample_id' in synthetic:
                synthetic['sample_id'] = f"{synthetic['sample_id']}_synth_dev_{dev_sample['homonym']}"
            augmented_samples.append(synthetic)
        
        return augmented_samples
    
    def strategy_4_rating_preserving_transform(self, sample: dict) -> dict:
        """
        Strategy 4: Transform text in ways that preserve rating.
        Change character names, locations, but keep semantic relationships.
        """
        augmented = copy.deepcopy(sample)
        
        # Simple transformations that preserve meaning
        text_fields = ['precontext', 'sentence', 'ending']
        for field in text_fields:
            if not sample.get(field):
                continue
            
            text = sample[field]
            
            # Replace common names (simple heuristic)
            name_replacements = {
                'Tommy': random.choice(['Johnny', 'Bobby', 'Mikey', 'Danny']),
                'Clara': random.choice(['Sarah', 'Emma', 'Lisa', 'Anna']),
                'John': random.choice(['Mark', 'Paul', 'David', 'Chris']),
                'Mary': random.choice(['Jane', 'Kate', 'Amy', 'Beth']),
            }
            
            for old_name, new_name in name_replacements.items():
                if old_name in text and random.random() < 0.5:
                    text = text.replace(old_name, new_name)
            
            augmented[field] = text
        
        if 'sample_id' in augmented:
            augmented['sample_id'] = f"{augmented['sample_id']}_rating_preserve"
        
        return augmented
    
    def strategy_5_same_context_different_homonym(self, sample: dict) -> List[dict]:
        """
        Strategy 5: Find other samples with same context structure but different homonym.
        This forces the model to focus on coherence, not homonym identity.
        """
        augmented_samples = []
        
        # Find samples with similar precontext/sentence structure
        sample_precontext = sample.get('precontext', '')
        sample_sentence = sample.get('sentence', '')
        
        # Look for samples with similar context but different homonym
        candidates = []
        for k, v in self.train_data.items():
            if (v['homonym'] != sample['homonym'] and
                abs(len(v.get('precontext', '')) - len(sample_precontext)) < 50 and
                abs(round(v['average']) - round(sample['average'])) <= 1):
                candidates.append((k, v))
        
        if len(candidates) < 2:
            return []
        
        # Pick 1-2 candidates
        num_swaps = min(2, len(candidates))
        selected = random.sample(candidates, num_swaps)
        
        for cand_key, cand_sample in selected:
            # Create hybrid: use candidate's homonym/meaning but original's context structure
            hybrid = copy.deepcopy(sample)
            hybrid['homonym'] = cand_sample['homonym']
            hybrid['judged_meaning'] = cand_sample['judged_meaning']
            # Adjust rating slightly based on candidate's rating (weighted average)
            original_rating = sample['average']
            cand_rating = cand_sample['average']
            hybrid['average'] = 0.7 * original_rating + 0.3 * cand_rating
            # Recalculate choices to match new average
            hybrid['choices'] = [
                max(1, min(5, round(hybrid['average'] + random.gauss(0, 0.5))))
                for _ in range(5)
            ]
            hybrid['stdev'] = (sum((c - hybrid['average'])**2 for c in hybrid['choices']) / len(hybrid['choices']))**0.5
            
            if 'sample_id' in hybrid:
                hybrid['sample_id'] = f"{hybrid['sample_id']}_ctx_swap_{cand_sample['homonym']}"
            augmented_samples.append(hybrid)
        
        return augmented_samples
    
    def augment_dataset(self, 
                       augmentation_factor=1.0,
                       use_strategies=None,
                       samples_per_strategy=None):
        """
        Augment dataset using multiple strategies.
        
        Args:
            augmentation_factor: Fraction of samples to augment
            use_strategies: List of strategy numbers to use [1,2,3,4,5]
            samples_per_strategy: Dict mapping strategy -> number of augmented samples per original
        """
        if use_strategies is None:
            use_strategies = [1, 2, 3, 4, 5]
        
        if samples_per_strategy is None:
            samples_per_strategy = {
                1: 1,  # Cross-homonym swap: 1 per sample
                2: 1,  # Context variation: 1 per sample
                3: 2,  # Synthetic dev homonyms: 2 per sample
                4: 1,  # Rating-preserving: 1 per sample
                5: 1,  # Same context different homonym: 1 per sample
            }
        
        print(f"\nUsing strategies: {use_strategies}")
        print(f"Samples per strategy: {samples_per_strategy}")
        
        augmented_data = copy.deepcopy(self.train_data)
        original_keys = list(self.train_data.keys())
        n_augment = int(len(original_keys) * augmentation_factor)
        samples_to_augment = random.sample(original_keys, min(n_augment, len(original_keys)))
        
        print(f"Augmenting {len(samples_to_augment)} samples...")
        
        new_key = len(original_keys)
        total_augmented = 0
        
        for i, key in enumerate(samples_to_augment):
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(samples_to_augment)} samples ({total_augmented} total augmented)...")
            
            sample = self.train_data[key]
            
            # Strategy 1: Cross-homonym swap
            if 1 in use_strategies:
                # Find a different homonym from train set
                other_homonyms = [h for h in self.train_by_homonym.keys() if h != sample['homonym']]
                if other_homonyms:
                    for _ in range(samples_per_strategy.get(1, 1)):
                        target_homonym = random.choice(other_homonyms)
                        # Find a meaning for this homonym
                        target_samples = self.train_by_homonym[target_homonym]
                        if target_samples:
                            _, target_sample = random.choice(target_samples)
                            augmented_sample = self.strategy_1_cross_homonym_swap(
                                sample, target_homonym, target_sample['judged_meaning']
                            )
                            augmented_data[str(new_key)] = augmented_sample
                            new_key += 1
                            total_augmented += 1
            
            # Strategy 2: Context variation
            if 2 in use_strategies:
                for _ in range(samples_per_strategy.get(2, 1)):
                    augmented_sample = self.strategy_2_context_variation(sample)
                    augmented_data[str(new_key)] = augmented_sample
                    new_key += 1
                    total_augmented += 1
            
            # Strategy 3: Synthetic dev homonyms
            if 3 in use_strategies:
                synthetic_samples = self.strategy_3_synthetic_dev_homonyms(sample)
                for aug_sample in synthetic_samples[:samples_per_strategy.get(3, 2)]:
                    augmented_data[str(new_key)] = aug_sample
                    new_key += 1
                    total_augmented += 1
            
            # Strategy 4: Rating-preserving transform
            if 4 in use_strategies:
                for _ in range(samples_per_strategy.get(4, 1)):
                    augmented_sample = self.strategy_4_rating_preserving_transform(sample)
                    augmented_data[str(new_key)] = augmented_sample
                    new_key += 1
                    total_augmented += 1
            
            # Strategy 5: Same context, different homonym
            if 5 in use_strategies:
                hybrid_samples = self.strategy_5_same_context_different_homonym(sample)
                for aug_sample in hybrid_samples[:samples_per_strategy.get(5, 1)]:
                    augmented_data[str(new_key)] = aug_sample
                    new_key += 1
                    total_augmented += 1
        
        print(f"\n✓ Augmentation complete: {len(self.train_data)} -> {len(augmented_data)} samples")
        print(f"  Original: {len(self.train_data)}, Augmented: {total_augmented}, Total: {len(augmented_data)}")
        
        return augmented_data


def main():
    parser = argparse.ArgumentParser(description="Improved data augmentation for Semeval 2026 Task 5")
    parser.add_argument("--train", default=os.path.join(DATA_ROOT, "train.json"), help="Input training data")
    parser.add_argument("--dev", default=os.path.join(DATA_ROOT, "dev.json"), help="Dev data (for homonym analysis)")
    parser.add_argument("--output", default=os.path.join(DATA_ROOT, "train_augmented_v2.json"), help="Output file")
    parser.add_argument("--factor", type=float, default=1.0, help="Augmentation factor (1.0 = all samples)")
    parser.add_argument("--strategies", nargs="+", type=int, default=[1, 2, 3, 4, 5],
                       help="Strategies to use: 1=cross-homonym, 2=context-var, 3=synthetic-dev, 4=rating-preserve, 5=context-swap")
    parser.add_argument("--per_sample", type=int, default=None,
                       help="Number of augmented samples per original sample (multiplies all strategies). If not set, uses default samples_per_strategy.")
    parser.add_argument("--per_strategy", nargs="+", type=int, default=None,
                       help="Samples per strategy in order [1,2,3,4,5]. Example: --per_strategy 2 1 3 1 1")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    # Load data
    print(f"Loading train data from {args.train}...")
    train_data = load_split(args.train)
    print(f"Loaded {len(train_data)} train samples")
    
    print(f"Loading dev data from {args.dev}...")
    dev_data = load_split(args.dev)
    print(f"Loaded {len(dev_data)} dev samples")
    
    # Initialize augmenter
    augmenter = ImprovedDataAugmenter(train_data, dev_data)
    
    # Determine samples_per_strategy
    samples_per_strategy = None
    if args.per_strategy:
        # User specified per-strategy counts
        if len(args.per_strategy) != len(args.strategies):
            print(f"Warning: --per_strategy has {len(args.per_strategy)} values but {len(args.strategies)} strategies.")
            print("Using first N values or padding with 1.")
        samples_per_strategy = {}
        for i, strategy in enumerate(args.strategies):
            if i < len(args.per_strategy):
                samples_per_strategy[strategy] = args.per_strategy[i]
            else:
                samples_per_strategy[strategy] = 1
    elif args.per_sample:
        # User wants N samples per strategy per original sample
        samples_per_strategy = {}
        for strategy in args.strategies:
            samples_per_strategy[strategy] = args.per_sample
        print(f"Creating {args.per_sample} augmented samples per strategy per original sample")
        print(f"Total: {len(args.strategies) * args.per_sample} augmented samples per original (with {len(args.strategies)} strategies)")
    
    # Augment
    print(f"\nAugmenting with factor {args.factor}...")
    augmented_data = augmenter.augment_dataset(
        augmentation_factor=args.factor,
        use_strategies=args.strategies,
        samples_per_strategy=samples_per_strategy
    )
    
    # Save
    print(f"\nSaving augmented data to {args.output}...")
    save_split(augmented_data, args.output)
    print(f"✓ Saved {len(augmented_data)} samples")
    
    # Show example
    print("\nExample augmented sample:")
    original_key = list(train_data.keys())[0]
    augmented_key = str(len(train_data))
    if augmented_key in augmented_data:
        print(f"\nOriginal (key {original_key}):")
        orig = train_data[original_key]
        print(f"  Homonym: {orig['homonym']}")
        print(f"  Meaning: {orig['judged_meaning'][:80]}...")
        print(f"  Rating: {orig['average']:.2f}")
        
        aug = augmented_data[augmented_key]
        print(f"\nAugmented (key {augmented_key}):")
        print(f"  Homonym: {aug['homonym']}")
        print(f"  Meaning: {aug['judged_meaning'][:80]}...")
        print(f"  Rating: {aug['average']:.2f}")


if __name__ == "__main__":
    main()

