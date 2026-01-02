"""
Data Augmentation Script for Semeval 2026 Task 5
Uses multiple strategies: paraphrasing, synonym replacement, back-translation
"""

import argparse
import json
import random
from typing import Dict, List
import copy

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
    nltk.download('punkt_tab')
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


class DataAugmenter:
    """Data augmentation for narrative rating task."""
    
    def __init__(self, use_paraphrase=True, use_synonyms=True, use_backtranslation=False):
        self.use_paraphrase = use_paraphrase and HAS_TRANSFORMERS
        self.use_synonyms = use_synonyms and HAS_NLTK
        self.use_backtranslation = use_backtranslation and HAS_TRANSFORMERS
        
        # Initialize models if available
        self.paraphrase_model = None
        if self.use_paraphrase:
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
                self.use_paraphrase = False
        
        # Download NLTK data if needed
        if self.use_synonyms:
            try:
                nltk.data.find('tokenizers/punkt')
                nltk.data.find('corpora/wordnet')
            except LookupError:
                print("Downloading NLTK data (first time only)...")
                nltk.download('punkt', quiet=True)
                nltk.download('wordnet', quiet=True)
                nltk.download('omw-1.4', quiet=True)
    
    def get_synonyms(self, word: str, pos=None):
        """Get synonyms for a word."""
        if not HAS_NLTK:
            return []
        
        synonyms = set()
        for syn in wordnet.synsets(word, pos=pos):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ').lower()
                if synonym != word.lower():
                    synonyms.add(synonym)
        return list(synonyms)[:3]  # Return up to 3 synonyms
    
    def synonym_replacement(self, text: str, homonym: str = None, replace_ratio=0.1):
        """Replace words with synonyms, avoiding the homonym."""
        if not self.use_synonyms:
            return text
        
        words = word_tokenize(text)
        homonym_lower = homonym.lower() if homonym else ""
        n_replace = max(1, int(len(words) * replace_ratio))
        
        replaced = 0
        new_words = []
        for word in words:
            if replaced >= n_replace:
                new_words.append(word)
                continue
            
            # Skip homonym and common words
            if (word.lower() == homonym_lower or 
                word.lower() in ['the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been'] or
                not word.isalpha()):
                new_words.append(word)
                continue
            
            # Try to replace with synonym
            synonyms = self.get_synonyms(word.lower())
            if synonyms:
                new_words.append(random.choice(synonyms))
                replaced += 1
            else:
                new_words.append(word)
        
        return ' '.join(new_words)
    
    def paraphrase_text(self, text: str, max_length=512):
        """Paraphrase text using a model."""
        if not self.use_paraphrase or self.paraphrase_model is None:
            return text
        
        try:
            # Truncate if too long
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
            print(f"Paraphrase failed: {e}")
            return text
    
    def simple_rephrasing(self, text: str):
        """Simple rule-based rephrasing."""
        # Safe transformations that preserve meaning
        replacements = {
            " couldn't ": " could not ",
            " can't ": " cannot ",
            " won't ": " will not ",
            " didn't ": " did not ",
            " isn't ": " is not ",
            " aren't ": " are not ",
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Reverse some replacements randomly
        for new, old in replacements.items():
            if random.random() < 0.3:
                text = text.replace(new, old)
        
        return text
    
    def augment_sample(self, sample: dict, strategy="paraphrase"):
        """Augment a single sample."""
        augmented = copy.deepcopy(sample)
        
        homonym = sample.get("homonym", "")
        
        # Augment text fields
        if "precontext" in sample and sample["precontext"]:
            if strategy == "paraphrase":
                augmented["precontext"] = self.paraphrase_text(sample["precontext"])
            elif strategy == "synonym":
                augmented["precontext"] = self.synonym_replacement(sample["precontext"], homonym)
            elif strategy == "rephrase":
                augmented["precontext"] = self.simple_rephrasing(sample["precontext"])
        
        if "sentence" in sample and sample["sentence"]:
            if strategy == "paraphrase":
                augmented["sentence"] = self.paraphrase_text(sample["sentence"])
            elif strategy == "synonym":
                augmented["sentence"] = self.synonym_replacement(sample["sentence"], homonym)
            elif strategy == "rephrase":
                augmented["sentence"] = self.simple_rephrasing(sample["sentence"])
        
        if "ending" in sample and sample["ending"]:
            if strategy == "paraphrase":
                augmented["ending"] = self.paraphrase_text(sample["ending"])
            elif strategy == "synonym":
                augmented["ending"] = self.synonym_replacement(sample["ending"], homonym)
            elif strategy == "rephrase":
                augmented["ending"] = self.simple_rephrasing(sample["ending"])
        
        # Update sample_id to indicate augmentation
        if "sample_id" in augmented:
            augmented["sample_id"] = f"{augmented['sample_id']}_aug_{strategy}"
        
        return augmented
    
    def augment_dataset(self, data: Dict[str, dict], augmentation_factor=1.0, strategies=None, augment_per_sample=1):
        """Augment entire dataset.
        
        Args:
            augmentation_factor: Fraction of samples to augment (1.0 = all samples)
            strategies: List of strategies to use
            augment_per_sample: Number of augmented versions to create per sample
        """
        if strategies is None:
            strategies = []
            if self.use_paraphrase:
                strategies.append("paraphrase")
            if self.use_synonyms:
                strategies.append("synonym")
            strategies.append("rephrase")  # Always available
        
        if not strategies:
            print("No augmentation strategies available!")
            return data
        
        print(f"Using augmentation strategies: {strategies}")
        print(f"Creating {augment_per_sample} augmented version(s) per sample")
        
        augmented_data = copy.deepcopy(data)
        original_keys = list(data.keys())
        n_augment = int(len(original_keys) * augmentation_factor)
        
        # Select random samples to augment
        samples_to_augment = random.sample(original_keys, min(n_augment, len(original_keys)))
        
        print(f"Augmenting {len(samples_to_augment)} samples...")
        
        new_key = len(original_keys)
        total_augmented = 0
        
        for i, key in enumerate(samples_to_augment):
            if (i + 1) % 100 == 0:
                print(f"  Augmented {i + 1}/{len(samples_to_augment)} samples ({total_augmented} total augmented samples)...")
            
            sample = data[key]
            
            # Create multiple augmented versions
            for aug_idx in range(augment_per_sample):
                # Apply random strategy (can be different for each version)
                strategy = random.choice(strategies)
                augmented_sample = self.augment_sample(sample, strategy=strategy)
                
                # Add with new key
                augmented_data[str(new_key)] = augmented_sample
                new_key += 1
                total_augmented += 1
        
        print(f"✓ Augmentation complete: {len(data)} -> {len(augmented_data)} samples")
        print(f"  Original: {len(data)}, Augmented: {total_augmented}, Total: {len(augmented_data)}")
        return augmented_data


def main():
    parser = argparse.ArgumentParser(description="Augment training data for Semeval 2026 Task 5")
    parser.add_argument("--input", default="data/train.json", help="Input training data file")
    parser.add_argument("--output", default="data/train_augmented.json", help="Output augmented data file")
    parser.add_argument("--factor", type=float, default=1.0, help="Augmentation factor (1.0 = augment all samples)")
    parser.add_argument("--per_sample", type=int, default=2, help="Number of augmented versions per sample")
    parser.add_argument("--strategies", nargs="+", default=None, 
                       choices=["paraphrase", "synonym", "rephrase", "all"],
                       help="Augmentation strategies to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    # Determine strategies
    strategies = args.strategies
    if strategies and "all" in strategies:
        strategies = ["paraphrase", "synonym", "rephrase"]
    
    # Load data
    print(f"Loading data from {args.input}...")
    data = load_split(args.input)
    print(f"Loaded {len(data)} samples")
    
    # Initialize augmenter
    use_paraphrase = strategies is None or "paraphrase" in strategies
    use_synonyms = strategies is None or "synonym" in strategies
    augmenter = DataAugmenter(
        use_paraphrase=use_paraphrase,
        use_synonyms=use_synonyms,
        use_backtranslation=False  # Too slow for now
    )
    
    # Augment
    print(f"\nAugmenting with factor {args.factor}...")
    augmented_data = augmenter.augment_dataset(
        data, 
        augmentation_factor=args.factor, 
        strategies=strategies,
        augment_per_sample=args.per_sample
    )
    
    # Save
    print(f"\nSaving augmented data to {args.output}...")
    save_split(augmented_data, args.output)
    print(f"✓ Saved {len(augmented_data)} samples (original: {len(data)}, augmented: {len(augmented_data) - len(data)})")
    
    # Show example
    print("\nExample augmented sample:")
    original_key = list(data.keys())[0]
    augmented_key = str(len(data))
    if augmented_key in augmented_data:
        print(f"\nOriginal (key {original_key}):")
        orig = data[original_key]
        print(f"  Precontext: {orig.get('precontext', '')[:100]}...")
        print(f"  Sentence: {orig.get('sentence', '')}")
        
        aug = augmented_data[augmented_key]
        print(f"\nAugmented (key {augmented_key}):")
        print(f"  Precontext: {aug.get('precontext', '')[:100]}...")
        print(f"  Sentence: {aug.get('sentence', '')}")


if __name__ == "__main__":
    main()

