"""Train a BiLSTM sentiment classifier on the IMDb 50K movie review dataset.

Usage:  python train.py
Input:  data/IMDB Dataset.csv  (columns: review, sentiment)
Output: models/ (model, tokenizer, config) and outputs/ (metrics, plots)
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "IMDB Dataset.csv"
MODEL_DIR = BASE_DIR / "models"
OUT_DIR = BASE_DIR / "outputs"

# ---- Hyperparameters / experimental settings ----
SEED = 42
VOCAB_SIZE = 20_000
EMBED_DIM = 128
LSTM_UNITS = 64
DENSE_UNITS = 64
DROPOUT = 0.4
MAX_LEN = 200
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-3
PATIENCE = 2
TEST_SIZE = 0.15
VAL_SIZE = 0.15  # fraction of the full dataset
OOV_TOKEN = "<OOV>"

_BR = re.compile(r"<\s*br\s*/?\s*>", re.IGNORECASE)
_TAG = re.compile(r"<[^>]+>")
_URL = re.compile(r"(https?://\S+|www\.\S+)")
_BAD = re.compile(r"[^a-z0-9\s'!?.,]")
_WS = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase, strip HTML/URLs/odd characters, normalise whitespace.

    Shared with app.py so training and inference preprocess identically.
    """
    text = str(text).lower()
    text = _BR.sub(" ", text)
    text = _TAG.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _BAD.sub(" ", text)
    # separate punctuation so '!' or '?' become their own tokens
    text = re.sub(r"([!?.,])", r" \1 ", text)
    return _WS.sub(" ", text).strip()


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}.\n"
            "Download it from https://www.kaggle.com/datasets/lakshmi25npathi/"
            "imdb-dataset-of-50k-movie-reviews and save it as data/IMDB Dataset.csv"
        )
    df = pd.read_csv(DATA_PATH)
    raw_n = len(df)
    df = df.dropna(subset=["review", "sentiment"])
    df["sentiment"] = df["sentiment"].str.strip().str.lower()
    df = df[df["sentiment"].isin(["positive", "negative"])]
    df["review"] = df["review"].map(clean_text)
    df = df[df["review"].str.len() > 0].drop_duplicates(subset="review")
    df["label"] = (df["sentiment"] == "positive").astype(int)
    print(f"Records: {raw_n} raw -> {len(df)} after cleaning/deduplication")
    print(df["sentiment"].value_counts().to_string())
    return df.reset_index(drop=True)


def plot_class_distribution(df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(5, 4))
    sns.countplot(x="sentiment", data=df, order=["negative", "positive"])
    plt.title("Class distribution (after cleaning)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "class_distribution.png", dpi=150)
    plt.close()


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import tensorflow as tf
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix, f1_score,
                                 precision_score, recall_score)
    from sklearn.model_selection import train_test_split
    from tensorflow.keras import layers, models
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer

    MODEL_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    df = load_data()
    plot_class_distribution(df)

    # ---- Stratified 70/15/15 split ----
    train_df, temp_df = train_test_split(
        df, test_size=TEST_SIZE + VAL_SIZE, stratify=df["label"], random_state=SEED)
    val_df, test_df = train_test_split(
        temp_df, test_size=TEST_SIZE / (TEST_SIZE + VAL_SIZE),
        stratify=temp_df["label"], random_state=SEED)
    print(f"Train/Val/Test: {len(train_df)}/{len(val_df)}/{len(test_df)}")

    # ---- Tokenizer fitted on training text only (avoids leakage) ----
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOKEN)
    tokenizer.fit_on_texts(train_df["review"])

    def encode(texts):
        seqs = tokenizer.texts_to_sequences(texts)
        return pad_sequences(seqs, maxlen=MAX_LEN, padding="pre", truncating="post")

    X_train, y_train = encode(train_df["review"]), train_df["label"].values
    X_val, y_val = encode(val_df["review"]), val_df["label"].values
    X_test, y_test = encode(test_df["review"]), test_df["label"].values

    # ---- Model ----
    model = models.Sequential([
        layers.Input(shape=(MAX_LEN,)),
        layers.Embedding(VOCAB_SIZE, EMBED_DIM, mask_zero=True),
        layers.Bidirectional(layers.LSTM(LSTM_UNITS)),
        layers.Dense(DENSE_UNITS, activation="relu"),
        layers.Dropout(DROPOUT),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
                  loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    model_path = MODEL_DIR / "sentiment_lstm.keras"
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
        ModelCheckpoint(str(model_path), monitor="val_loss", save_best_only=True),
    ]
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks)

    # ---- Evaluate best model on held-out test set ----
    best = tf.keras.models.load_model(model_path)
    probs = best.predict(X_test, batch_size=256).ravel()
    preds = (probs >= 0.5).astype(int)

    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds)),
        "recall": float(recall_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df),
        "epochs_run": len(history.history["loss"]),
    }
    report = classification_report(y_test, preds, target_names=["negative", "positive"])
    print(report)
    print(json.dumps(metrics, indent=2))

    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (OUT_DIR / "classification_report.txt").write_text(report)

    # ---- Plots ----
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["negative", "positive"], yticklabels=["negative", "positive"])
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion matrix (test set)")
    plt.tight_layout(); plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=150); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, key, title in zip(ax, ["loss", "accuracy"], ["Loss", "Accuracy"]):
        a.plot(history.history[key], label="train")
        a.plot(history.history["val_" + key], label="validation")
        a.set_title(title); a.set_xlabel("Epoch"); a.legend()
    plt.tight_layout(); plt.savefig(OUT_DIR / "training_curves.png", dpi=150); plt.close()

    # ---- Save tokenizer + config for the app ----
    (MODEL_DIR / "tokenizer.json").write_text(tokenizer.to_json(), encoding="utf-8")
    config = {
        "max_len": MAX_LEN, "vocab_size": VOCAB_SIZE, "embed_dim": EMBED_DIM,
        "lstm_units": LSTM_UNITS, "dense_units": DENSE_UNITS, "dropout": DROPOUT,
        "batch_size": BATCH_SIZE, "epochs": EPOCHS, "learning_rate": LEARNING_RATE,
        "patience": PATIENCE, "seed": SEED, "oov_token": OOV_TOKEN,
        "padding": "pre", "truncating": "post",
        "labels": {"0": "negative", "1": "positive"},
    }
    (MODEL_DIR / "config.json").write_text(json.dumps(config, indent=2))
    print("Done. Artifacts saved to models/ and outputs/.")


if __name__ == "__main__":
    main()
