# semeval26-05-scripts
Some scripts for Semeval 2026 Task 5. Equivalent to the scoring script on the CodaBench task. More baselines to be added later.

Link to submission website: https://www.codabench.org/competitions/10877/?secret_key=e3c13419-88c6-4c13-9989-8e694a2bc5c0

# How to evaluate predictions

First, remember to install the requirements.

To evaluate a prediction, please format it like the "predictions/[...].jsonl" files.
Each prediction must be in its own line. The "id" key corresponds to the keys of the samples in the gold data ("0", "1", etc).
The prediction key should be an integer between 1 and 5.

Once you prepared your prediction data, put it in the input/res/ folder (replacing the existing file) and call the evaluation script like this:

```
python scoring.py input/ref/solution.jsonl input/res/predictions.jsonl output/scores.json
```

Scores will be printed and written on output/scores.json. If your predictions file contains bad formatting or is incomplete, it will print an error.

To submit to CodaBench, zip the predictions.jsonl up and upload it to the "My Submissions" tab on the task website.

Test set is yet unreleased, so you can only test on the dev set for now. The samples (including labels) are public here: https://github.com/Janosch-Gehring/ambistory

# TensorFlow MLP Baseline

We provide a simple text MLP baseline implemented with TensorFlow/Keras that trains on `data/train.json` and predicts on `data/dev.json` by default. It concatenates the available text fields and learns a 5‑class classifier (labels 1–5), using the rounded mean of human judgments as the training target.

1) Install requirements (includes CPU‑only TensorFlow):

```
pip install -r requirements.txt
```

2) Run the baseline (writes `predictions/tf_mlp_predictions_dev.jsonl`):

```
python baselines/tf_mlp_baseline.py --train data/train.json --eval data/dev.json --output predictions/tf_mlp_predictions_dev.jsonl
```

Useful flags:
- `--epochs` (default 5)
- `--batch_size` (default 64)
- `--max_tokens` vocabulary size for TF‑IDF vectorizer (default 10000)

3) Evaluate the produced predictions on the dev set:

```
python evaluate.py predictions/tf_mlp_predictions_dev.jsonl dev
```

Or evaluate with the Codabench scoring format:

```
python scoring.py input/ref/solution.jsonl predictions/tf_mlp_predictions_dev.jsonl output/scores.json
```

Note: TensorFlow may be a large dependency. If you prefer, run this baseline in a virtual environment.