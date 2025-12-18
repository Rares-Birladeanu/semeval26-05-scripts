import argparse
import json
import os
import random
from statistics import mean
from typing import Dict, List

import numpy as np

# Import TensorFlow lazily so the script can still be imported without TF present
import tensorflow as tf
from tensorflow.keras import layers, models


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_split(filepath: str) -> Dict[str, dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def build_text(sample: dict) -> str:
    # Keep it simple: only use the main narrative fields
    parts = [
        str(sample.get("precontext", "")).strip(),
        str(sample.get("sentence", "")).strip(),
        str(sample.get("ending", "")).strip(),
    ]
    # Join non-empty parts with spaces
    return " ".join([p for p in parts if p])


def labels_to_class(choices: List[int]) -> int:
    # Convert a list of 1..5 ratings to a single class 0..4 via rounded mean
    avg = round(mean(choices))
    avg = min(5, max(1, int(avg)))
    return avg - 1


def prepare_dataset(split_data: Dict[str, dict]):
    ids = []
    texts = []
    classes = []
    for k, v in split_data.items():
        ids.append(k)
        texts.append(build_text(v))
        # Some splits (e.g., test) may not contain labels
        if "choices" in v and v["choices"]:
            classes.append(labels_to_class(v["choices"]))
    return ids, texts, np.array(classes, dtype=np.int32) if classes else None


def build_model(vectorizer: layers.TextVectorization, num_classes: int = 5) -> tf.keras.Model:
    inp = layers.Input(shape=(), dtype=tf.string, name="text")
    x = vectorizer(inp)
    # Minimal MLP on top of TF-IDF features
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="TensorFlow MLP baseline for Semeval 2026 Task 5")
    parser.add_argument("--train", default="data/train.json", help="Path to train.json")
    parser.add_argument("--eval", default="data/dev.json", help="Path to evaluation split (dev/test).json")
    parser.add_argument("--output", default="predictions/tf_mlp_predictions_dev.jsonl", help="Output predictions .jsonl path")
    parser.add_argument("--max_tokens", type=int, default=10000, help="Max vocabulary size for vectorizer")
    parser.add_argument("--epochs", type=int, default=5, help="Max training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)

    # Load data
    train_data = load_split(args.train)
    eval_data = load_split(args.eval)

    train_ids, train_texts, train_classes = prepare_dataset(train_data)
    eval_ids, eval_texts, _ = prepare_dataset(eval_data)

    # Build vectorizer
    vectorizer = layers.TextVectorization(
        max_tokens=args.max_tokens,
        output_mode="tf-idf",
        standardize="lower_and_strip_punctuation",
        name="tfidf_vec",
    )
    text_ds = tf.data.Dataset.from_tensor_slices(np.array(train_texts, dtype=object)).batch(256)
    vectorizer.adapt(text_ds)

    # Build model
    model = build_model(vectorizer)

    # Train with simple validation split (no tf.data pipelines)
    x_train = np.array(train_texts, dtype=object)
    y_train = train_classes
    model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle=True,
        verbose=2,
    )

    # Predict on eval
    x_eval = np.array(eval_texts, dtype=object)
    probs = model.predict(x_eval, batch_size=args.batch_size, verbose=0)
    preds = np.argmax(probs, axis=-1) + 1  # back to 1..5

    # Write predictions
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for i, p in zip(eval_ids, preds.tolist()):
            f.write(json.dumps({"id": i, "prediction": int(p)}) + "\n")

    print(f"Wrote predictions to {args.output}")


if __name__ == "__main__":
    main()
