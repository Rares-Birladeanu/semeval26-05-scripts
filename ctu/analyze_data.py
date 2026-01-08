"""Analyze differences between train and dev data to inform better augmentation strategies."""

import json
from collections import Counter, defaultdict
import statistics
from config import DATA_ROOT
import os

def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_data(data, name):
    print(f"\n{'='*60}")
    print(f"ANALYZING {name.upper()}")
    print(f"{'='*60}")
    
    # Basic stats
    print(f"\nTotal samples: {len(data)}")
    
    # Rating distribution
    ratings = [v['average'] for v in data.values()]
    print(f"\nRating Statistics:")
    print(f"  Mean: {statistics.mean(ratings):.2f}")
    print(f"  Median: {statistics.median(ratings):.2f}")
    print(f"  StdDev: {statistics.stdev(ratings):.2f}")
    print(f"  Min: {min(ratings):.2f}")
    print(f"  Max: {max(ratings):.2f}")
    
    # Rating distribution bins
    bins = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for r in ratings:
        bins[round(r)] += 1
    print(f"\nRating Distribution (rounded):")
    for k, v in sorted(bins.items()):
        print(f"  {k}: {v} ({100*v/len(ratings):.1f}%)")
    
    # Standard deviation of choices (disagreement)
    stdevs = [v['stdev'] for v in data.values()]
    print(f"\nDisagreement (stdev of choices):")
    print(f"  Mean stdev: {statistics.mean(stdevs):.2f}")
    print(f"  High disagreement (>1.5): {sum(1 for s in stdevs if s > 1.5)} ({100*sum(1 for s in stdevs if s > 1.5)/len(stdevs):.1f}%)")
    print(f"  Low disagreement (<0.5): {sum(1 for s in stdevs if s < 0.5)} ({100*sum(1 for s in stdevs if s < 0.5)/len(stdevs):.1f}%)")
    
    # Homonym distribution
    homonyms = Counter(v['homonym'] for v in data.values())
    print(f"\nTop 10 Homonyms:")
    for homonym, count in homonyms.most_common(10):
        print(f"  {homonym}: {count}")
    
    # Empty endings
    empty_endings = sum(1 for v in data.values() if not v.get('ending', '').strip())
    print(f"\nEmpty endings: {empty_endings} ({100*empty_endings/len(data):.1f}%)")
    
    # Text length statistics
    precontext_lens = [len(v.get('precontext', '')) for v in data.values()]
    sentence_lens = [len(v.get('sentence', '')) for v in data.values()]
    ending_lens = [len(v.get('ending', '')) for v in data.values()]
    
    print(f"\nText Length Statistics:")
    print(f"  Precontext: mean={statistics.mean(precontext_lens):.1f}, median={statistics.median(precontext_lens):.1f}")
    print(f"  Sentence: mean={statistics.mean(sentence_lens):.1f}, median={statistics.median(sentence_lens):.1f}")
    print(f"  Ending: mean={statistics.mean(ending_lens):.1f}, median={statistics.median(ending_lens):.1f}")
    
    # Analyze patterns: same precontext/sentence, different endings
    context_to_endings = defaultdict(list)
    for k, v in data.items():
        key = (v.get('precontext', ''), v.get('sentence', ''), v.get('homonym', ''))
        context_to_endings[key].append((v.get('ending', ''), v['average']))
    
    multi_endings = {k: v for k, v in context_to_endings.items() if len(v) > 1}
    print(f"\nContexts with multiple endings: {len(multi_endings)}")
    if multi_endings:
        # Show rating variance for same context
        rating_vars = []
        for endings in multi_endings.values():
            ratings = [e[1] for e in endings]
            if len(ratings) > 1:
                rating_vars.append(statistics.stdev(ratings))
        if rating_vars:
            print(f"  Mean rating stddev for multi-ending contexts: {statistics.mean(rating_vars):.2f}")
    
    # Analyze: same context, different meanings
    context_to_meanings = defaultdict(set)
    for k, v in data.items():
        key = (v.get('precontext', ''), v.get('sentence', ''), v.get('ending', ''))
        context_to_meanings[key].add(v.get('judged_meaning', ''))
    
    multi_meanings = {k: v for k, v in context_to_meanings.items() if len(v) > 1}
    print(f"\nContexts with multiple meanings: {len(multi_meanings)}")
    
    return {
        'ratings': ratings,
        'stdevs': stdevs,
        'homonyms': homonyms,
        'empty_endings': empty_endings,
        'precontext_lens': precontext_lens,
        'sentence_lens': sentence_lens,
        'ending_lens': ending_lens,
        'multi_endings': multi_endings,
        'multi_meanings': multi_meanings
    }

def compare_datasets(train_stats, dev_stats):
    print(f"\n{'='*60}")
    print(f"COMPARISON: TRAIN vs DEV")
    print(f"{'='*60}")
    
    print(f"\nRating Distribution Difference:")
    train_mean = statistics.mean(train_stats['ratings'])
    dev_mean = statistics.mean(dev_stats['ratings'])
    print(f"  Train mean: {train_mean:.2f}")
    print(f"  Dev mean: {dev_mean:.2f}")
    print(f"  Difference: {abs(train_mean - dev_mean):.2f}")
    
    print(f"\nDisagreement Difference:")
    train_stdev_mean = statistics.mean(train_stats['stdevs'])
    dev_stdev_mean = statistics.mean(dev_stats['stdevs'])
    print(f"  Train mean stdev: {train_stdev_mean:.2f}")
    print(f"  Dev mean stdev: {dev_stdev_mean:.2f}")
    print(f"  Difference: {abs(train_stdev_mean - dev_stdev_mean):.2f}")
    
    print(f"\nEmpty Endings:")
    print(f"  Train: {train_stats['empty_endings']} ({100*train_stats['empty_endings']/len(train_stats['ratings']):.1f}%)")
    print(f"  Dev: {dev_stats['empty_endings']} ({100*dev_stats['empty_endings']/len(dev_stats['ratings']):.1f}%)")
    
    # Homonym overlap
    train_homonyms = set(train_stats['homonyms'].keys())
    dev_homonyms = set(dev_stats['homonyms'].keys())
    overlap = train_homonyms & dev_homonyms
    train_only = train_homonyms - dev_homonyms
    dev_only = dev_homonyms - train_homonyms
    print(f"\nHomonym Overlap:")
    print(f"  Train unique: {len(train_homonyms)}")
    print(f"  Dev unique: {len(dev_homonyms)}")
    print(f"  Overlap: {len(overlap)}")
    print(f"  Train only: {len(train_only)}")
    print(f"  Dev only: {len(dev_only)}")
    if dev_only:
        print(f"  Dev-only homonyms: {list(dev_only)[:10]}")

def main():
    train_data = load_data(os.path.join(DATA_ROOT, 'train.json'))
    dev_data = load_data(os.path.join(DATA_ROOT, 'dev.json'))
    
    train_stats = analyze_data(train_data, 'TRAIN')
    dev_stats = analyze_data(dev_data, 'DEV')
    compare_datasets(train_stats, dev_stats)
    
    # Show some dev examples
    print(f"\n{'='*60}")
    print(f"SAMPLE DEV EXAMPLES")
    print(f"{'='*60}")
    for i, (k, v) in enumerate(list(dev_data.items())[:5]):
        print(f"\nExample {i+1}:")
        print(f"  Homonym: {v['homonym']}")
        print(f"  Meaning: {v['judged_meaning']}")
        print(f"  Precontext: {v.get('precontext', '')[:100]}...")
        print(f"  Sentence: {v.get('sentence', '')}")
        print(f"  Ending: {v.get('ending', '')[:100]}...")
        print(f"  Rating: {v['average']:.2f} (stdev: {v['stdev']:.2f})")

if __name__ == "__main__":
    main()

