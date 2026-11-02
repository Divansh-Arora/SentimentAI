"""Streamlit interface for SentimentAI.  Run: streamlit run app.py"""
import json
from pathlib import Path

import streamlit as st

from train import clean_text

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
OUT_DIR = BASE_DIR / "outputs"

st.set_page_config(page_title="SentimentAI", page_icon="🎬", layout="centered")


@st.cache_resource
def load_artifacts():
    import tensorflow as tf
    from tensorflow.keras.preprocessing.text import tokenizer_from_json

    model = tf.keras.models.load_model(MODEL_DIR / "sentiment_lstm.keras")
    tokenizer = tokenizer_from_json((MODEL_DIR / "tokenizer.json").read_text(encoding="utf-8"))
    config = json.loads((MODEL_DIR / "config.json").read_text())
    return model, tokenizer, config


def predict(text: str, model, tokenizer, config) -> float:
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    seq = tokenizer.texts_to_sequences([clean_text(text)])
    padded = pad_sequences(seq, maxlen=config["max_len"],
                           padding=config["padding"], truncating=config["truncating"])
    return float(model.predict(padded, verbose=0)[0][0])


st.title("🎬 SentimentAI")
st.caption("Movie review sentiment analysis with a Bidirectional LSTM trained on IMDb reviews.")

required = [MODEL_DIR / n for n in ("sentiment_lstm.keras", "tokenizer.json", "config.json")]
if not all(p.exists() for p in required):
    st.error("Model files not found. Place the dataset in `data/` and run `python train.py` first.")
    st.stop()

model, tokenizer, config = load_artifacts()

examples = {
    "(choose an example)": "",
    "Positive": "An absolutely wonderful film with brilliant acting and a moving story.",
    "Negative": "A boring, predictable mess. I regret wasting two hours on this.",
    "Mixed": "The visuals were stunning, but the plot was weak and the ending fell flat.",
}
choice = st.selectbox("Try an example", list(examples))
review = st.text_area("Enter a movie review", value=examples[choice], height=180)

if st.button("Analyze sentiment", type="primary"):
    if not review.strip():
        st.warning("Please enter a review first.")
    elif not clean_text(review):
        st.warning("The review has no usable text after cleaning. Try plain words.")
    else:
        p_pos = predict(review, model, tokenizer, config)
        p_neg = 1 - p_pos
        label = "Positive" if p_pos >= 0.5 else "Negative"
        (st.success if label == "Positive" else st.error)(
            f"Predicted sentiment: **{label}** ({max(p_pos, p_neg):.1%} confidence)")
        c1, c2 = st.columns(2)
        c1.metric("Positive", f"{p_pos:.1%}")
        c2.metric("Negative", f"{p_neg:.1%}")
        st.progress(p_pos)
        st.caption("Probabilities are model estimates, not guarantees. Sarcasm, mixed "
                   "opinions and non-movie text may be misclassified.")

with st.sidebar:
    st.header("Model info")
    st.write(f"Sequence length: {config['max_len']}  \nVocabulary: {config['vocab_size']:,}")
    mpath = OUT_DIR / "metrics.json"
    if mpath.exists():
        m = json.loads(mpath.read_text())
        st.subheader("Test-set results")
        st.write(f"Accuracy: {m['accuracy']:.3f}  \nPrecision: {m['precision']:.3f}  \n"
                 f"Recall: {m['recall']:.3f}  \nF1: {m['f1']:.3f}")
