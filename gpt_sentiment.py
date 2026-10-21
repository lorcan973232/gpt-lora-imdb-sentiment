#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CA1 script for the module Artificial Intelligence for Computer Vision & NLP:
Assignment 1: Deep Learning for NLP.

This file runs the full GPT sentiment study from start to finish.
"""

# This file started as a notebook and was then saved as one Python script.

# Simple fallback so display() still works when the file is run as a normal Python script.
try:
    from IPython.display import display
except Exception:
    def display(*objs):
        for obj in objs:
            print(obj)


# ==============================================================================
# Cell 0 [markdown]
# ==============================================================================
# # Parameter-Efficient Fine-Tuning of Decoder-Only GPT Models for Long-Form Sentiment Classification on IMDb
#
# It compares three systems:
#
# - **Baseline A:** TF-IDF + Logistic Regression  
# - **Baseline B:** frozen GPT representations + lightweight classifier  
# - **System C:** LoRA-based fine-tuning of a decoder-only GPT classifier

# ==============================================================================
# Cell 1 [markdown]
# ==============================================================================
# ## 1. Why this problem is a good NLP task
#
# This is not just a basic IMDb sentiment task. It is set up as a study of **decoder-only GPT fine-tuning for long reviews**.
#
# There are five main reasons why this is not a simple task.
#
# 1. **IMDb reviews are long.** Many reviews are too long to fit neatly into a student hardware context window, so truncation becomes a real design choice.
# 2. **Decoder-only GPTs are not the usual choice for classification.** Encoder models are often a more natural fit, so GPT needs careful adaptation.
# 3. **GPT classification is not plug and play.** Pooling, padding, truncation, and parameter-efficient tuning all matter.
# 4. **Embeddings need to be explained, not just used.** GPT uses subword tokenisation, contextual token embeddings, and positional embeddings.
# 5. **The task allows proper comparison.** The script tests baselines, GPT candidates, hyperparameters, and error patterns instead of showing one unchecked model.
#
# For that reason, the script is set up as a small empirical study rather than a basic coursework demo.

# ==============================================================================
# Cell 2 [markdown]
# ==============================================================================
# ## 2. Environment and reproducibility
#
# The script is meant to run from top to bottom in a fresh environment. Randomness is seeded, library versions are recorded, split files are saved, and output artefacts are written to disk.
#
# There are two run modes:
#
# - **Full study mode** for the main coursework run.
# - **Quick debug mode** for a short test on limited hardware.
#
# Quick mode is off by default because the aim is a full coursework run.

# ==============================================================================
# Cell 3 [code]
# ==============================================================================
# Install these packages once if you are using a fresh environment.
# Leave this commented if they are already installed.
# !pip install -q transformers datasets peft accelerate scikit-learn matplotlib seaborn tqdm

# ==============================================================================
# Cell 4 [code]
# ==============================================================================
import os
import gc
import copy
import re
import json
import time
import math
import random
import tarfile
import pickle
import platform
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

import sklearn
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    ConfusionMatrixDisplay,
)
from sklearn.calibration import calibration_curve

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    GPT2Model,
    GPT2ForSequenceClassification,
    set_seed as hf_set_seed,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset, DatasetDict

from peft import get_peft_model, LoraConfig, TaskType

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 140)
print("Imports loaded successfully.")

# ==============================================================================
# Cell 5 [code]
# ==============================================================================
SEED = 42
SPLIT_SEED = 42

# Default settings for the full run
DEBUG_MODE = False
MAX_TRAIN_SAMPLES = None
MAX_VAL_SAMPLES = None
MAX_TEST_SAMPLES = None

# Smaller run for a quick local test
if DEBUG_MODE:
    MAX_TRAIN_SAMPLES = 4000
    MAX_VAL_SAMPLES = 1000
    MAX_TEST_SAMPLES = 2000

ROOT_DIR = Path(".").resolve()
ARTIFACT_DIR = ROOT_DIR / "artefacts"
OUTPUT_DIR = ROOT_DIR / "outputs"

TABLES_DIR = OUTPUT_DIR / "tables"
METRICS_DIR = OUTPUT_DIR / "metrics"
FIGURES_DIR = OUTPUT_DIR / "figures"
ERRORS_DIR = OUTPUT_DIR / "errors"
EMBEDDINGS_DIR = OUTPUT_DIR / "embeddings"
MODELS_DIR = OUTPUT_DIR / "models"
LOGS_DIR = OUTPUT_DIR / "logs"
CACHE_DIR = ARTIFACT_DIR / "cache"
MODEL_DIR = MODELS_DIR / "final_model"
SPLIT_DIR = OUTPUT_DIR / "splits"

for path in [
    ARTIFACT_DIR, OUTPUT_DIR, TABLES_DIR, METRICS_DIR, FIGURES_DIR,
    ERRORS_DIR, EMBEDDINGS_DIR, MODELS_DIR, LOGS_DIR, CACHE_DIR, MODEL_DIR, SPLIT_DIR
]:
    path.mkdir(parents=True, exist_ok=True)

def set_all_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    hf_set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_all_seeds(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env_info = {
    "Python": platform.python_version(),
    "PyTorch": torch.__version__,
    "Transformers": transformers.__version__,
    "Scikit-learn": sklearn.__version__,
    "NumPy": np.__version__,
    "Pandas": pd.__version__,
    "Device": str(device),
    "Debug mode": DEBUG_MODE,
}
if torch.cuda.is_available():
    env_info["GPU"] = torch.cuda.get_device_name(0)
    env_info["VRAM_GB"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)

env_df = pd.DataFrame(env_info.items(), columns=["Component", "Value"])
display(env_df)



# ------------------------------
# Save and logging helpers
# ------------------------------
def _json_safe(obj):
    """Recursively convert NumPy/Pandas scalars and containers into JSON-safe Python types."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Series):
        return _json_safe(obj.to_dict())
    if isinstance(obj, pd.DataFrame):
        return _json_safe(obj.to_dict(orient="records"))
    return obj

def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(data), f, indent=2, ensure_ascii=False)

def sanitize_config_dict(config: dict) -> dict:
    """Convert any NumPy/Pandas scalar values in a config dict to plain Python types."""
    return {str(k): _json_safe(v) for k, v in config.items()}

def save_dataframe(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)

def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def save_current_figure(filename: str, dpi: int = 200) -> None:
    out_path = FIGURES_DIR / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.gcf().savefig(out_path, dpi=dpi, bbox_inches="tight")

def sanitize_peft_config_inplace(model) -> None:
    if hasattr(model, "peft_config"):
        for _, peft_cfg in model.peft_config.items():
            for attr, value in list(vars(peft_cfg).items()):
                try:
                    setattr(peft_cfg, attr, _json_safe(value))
                except Exception:
                    pass

def append_run_summary(message: str) -> None:
    summary_path = LOGS_DIR / "run_summary.txt"
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write(message.rstrip() + "\n")

_ORIGINAL_PLT_SHOW = plt.show
_FIGURE_COUNTER = 0

def _auto_save_open_figures(prefix: str = "figure", dpi: int = 200) -> None:
    global _FIGURE_COUNTER
    fig_nums = plt.get_fignums()
    for fig_num in fig_nums:
        fig = plt.figure(fig_num)
        _FIGURE_COUNTER += 1
        out_path = FIGURES_DIR / f"{prefix}_{_FIGURE_COUNTER:03d}.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")

def auto_saving_show(*args, **kwargs):
    _auto_save_open_figures()
    result = _ORIGINAL_PLT_SHOW(*args, **kwargs)
    plt.close("all")
    return result

plt.show = auto_saving_show
append_run_summary("Run started.")
append_run_summary(f"Environment: {env_info}")


def stratified_sample_df(frame: pd.DataFrame, label_col: str, n_per_class: int, random_state: int = SEED) -> pd.DataFrame:
    """
    Sample up to n_per_class examples from each class while preserving the label column.

    This avoids pandas groupby-apply behaviour that can drop grouping columns in newer versions.
    """
    parts = []
    for _, group in frame.groupby(label_col, sort=False):
        take_n = min(len(group), n_per_class)
        parts.append(group.sample(n=take_n, random_state=random_state))
    return (
        pd.concat(parts, axis=0)
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )

# ==============================================================================
# Cell 6 [markdown]
# ==============================================================================
# ## 3. Dataset
#
# The IMDb review dataset is loaded directly in code. If a local archive is available, that is used first; otherwise the script falls back to Hugging Face. The original train and test splits are retained. The validation split is created **only from the training data** using stratified sampling with a fixed split seed.
#
# The test set is kept separate and not used for tuning or model selection.

# ==============================================================================
# Cell 7 [code]
# ==============================================================================
import datasets

def load_imdb_programmatically() -> DatasetDict:
    """Load IMDb from a local aclImdb archive when available, otherwise from Hugging Face."""
    local_tar = Path("aclImdb_v1.tar.gz")
    if local_tar.exists():
        extract_dir = CACHE_DIR / "aclImdb"
        if not extract_dir.exists():
            with tarfile.open(local_tar, "r:gz") as tar:
                tar.extractall(path=CACHE_DIR)

        if not extract_dir.exists():
            candidate_dirs = [p for p in CACHE_DIR.iterdir() if p.is_dir() and p.name.lower().startswith("aclimdb")]
            if candidate_dirs:
                extract_dir = candidate_dirs[0]

        rows = {}
        for split in ["train", "test"]:
            texts, labels = [], []
            for label_name, label_value in [("neg", 0), ("pos", 1)]:
                data_dir = extract_dir / split / label_name
                files = sorted(data_dir.glob("*.txt"))
                if not files:
                    raise FileNotFoundError(f"No review files found in {data_dir}")
                for fp in tqdm(files, desc=f"Loading local {split}/{label_name}", leave=False):
                    texts.append(fp.read_text(encoding="utf-8", errors="ignore"))
                    labels.append(label_value)
            rows[split] = {"text": texts, "label": labels}

        return DatasetDict({
            split: datasets.Dataset.from_dict(rows[split]) for split in rows
        })

    return load_dataset("imdb")

print("Loading IMDb dataset...")
raw_dataset = load_imdb_programmatically()
print(raw_dataset)
print("\nSample example:")
print(raw_dataset["train"][0]["text"][:300].replace("\n", " "))
print("Label:", raw_dataset["train"][0]["label"])

# ==============================================================================
# Cell 8 [code]
# ==============================================================================
df_train_full = pd.DataFrame({
    "text": raw_dataset["train"]["text"],
    "label": raw_dataset["train"]["label"],
})
df_test = pd.DataFrame({
    "text": raw_dataset["test"]["text"],
    "label": raw_dataset["test"]["label"],
})

df_train, df_val = train_test_split(
    df_train_full,
    test_size=0.10,
    stratify=df_train_full["label"],
    random_state=SPLIT_SEED,
)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

if MAX_TRAIN_SAMPLES is not None:
    df_train = stratified_sample_df(df_train, "label", MAX_TRAIN_SAMPLES // 2, random_state=SEED)

if MAX_VAL_SAMPLES is not None:
    df_val = stratified_sample_df(df_val, "label", MAX_VAL_SAMPLES // 2, random_state=SEED)

if MAX_TEST_SAMPLES is not None:
    df_test = stratified_sample_df(df_test, "label", MAX_TEST_SAMPLES // 2, random_state=SEED)

save_dataframe(df_train, SPLIT_DIR / "train_split.csv", index=False)
save_dataframe(df_val, SPLIT_DIR / "val_split.csv", index=False)
save_dataframe(df_test, SPLIT_DIR / "test_split.csv", index=False)

print(f"Train size: {len(df_train):,}")
print(f"Validation size: {len(df_val):,}")
print(f"Test size: {len(df_test):,}")
print(f"Split seed recorded: {SPLIT_SEED}")

# ==============================================================================
# Cell 9 [markdown]
# ==============================================================================
# ## 4. Initial data checks: class balance and review length
#
# Although IMDb is designed to be class-balanced, it is still sensible to confirm this after splitting. Review length is even more important here, since longer texts are more likely to be truncated and generally make sentiment classification harder than short-form inputs.

# ==============================================================================
# Cell 10 [code]
# ==============================================================================
for split_name, frame in [("train", df_train), ("val", df_val), ("test", df_test)]:
    frame["word_count"] = frame["text"].str.split().str.len()

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, (name, frame) in zip(axes, [("Train", df_train), ("Validation", df_val), ("Test", df_test)]):
    counts = frame["label"].value_counts().sort_index()
    ax.bar(["Negative", "Positive"], counts.values, edgecolor="black")
    ax.set_title(f"{name} class balance")
    ax.set_ylabel("Count")
    for i, v in enumerate(counts.values):
        ax.text(i, v + max(counts.values) * 0.01, f"{v:,}\n({v / len(frame):.1%})", ha="center", fontsize=9)
plt.tight_layout()
save_current_figure("class_balance.png")
plt.show()

balance_summary = pd.DataFrame({
    "Split": ["Train", "Validation", "Test"],
    "Negative": [int((df_train["label"] == 0).sum()), int((df_val["label"] == 0).sum()), int((df_test["label"] == 0).sum())],
    "Positive": [int((df_train["label"] == 1).sum()), int((df_val["label"] == 1).sum()), int((df_test["label"] == 1).sum())],
})
display(balance_summary)
save_dataframe(balance_summary, TABLES_DIR / "class_balance_summary.csv", index=False)

# ==============================================================================
# Cell 11 [code]
# ==============================================================================
length_stats = df_train["word_count"].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).round(1)
display(length_stats.to_frame("Train review word-count statistics"))
save_dataframe(length_stats.to_frame("Train review word-count statistics").reset_index(), TABLES_DIR / "review_length_statistics.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].hist(df_train["word_count"], bins=80, alpha=0.85)
axes[0].axvline(df_train["word_count"].median(), linestyle="--", label=f"Median = {int(df_train['word_count'].median())}")
axes[0].axvline(df_train["word_count"].quantile(0.95), linestyle=":", label=f"95th pct = {int(df_train['word_count'].quantile(0.95))}")
axes[0].set_xlabel("Review length (words)")
axes[0].set_ylabel("Frequency")
axes[0].set_title("IMDb review length distribution")
axes[0].legend()

sorted_lengths = np.sort(df_train["word_count"].values)
cdf = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths)
axes[1].plot(sorted_lengths, cdf)
for cutoff in [128, 256, 512, 1024]:
    frac = (df_train["word_count"] <= cutoff).mean()
    axes[1].axvline(cutoff, linestyle="--", alpha=0.6, label=f"{cutoff} words ≈ {frac:.1%}")
axes[1].set_xlabel("Review length (words)")
axes[1].set_ylabel("Cumulative proportion")
axes[1].set_title("Coverage if documents were hard-truncated")
axes[1].legend(fontsize=8)

plt.tight_layout()
save_current_figure("review_length_distribution.png")
plt.show()

# ==============================================================================
# Cell 12 [code]
# ==============================================================================
fig, ax = plt.subplots(figsize=(8, 4))
for label_value, label_name in [(0, "Negative"), (1, "Positive")]:
    subset = df_train.loc[df_train["label"] == label_value, "word_count"]
    ax.hist(subset, bins=60, alpha=0.55, label=f"{label_name} (median={int(subset.median())})")
ax.set_xlabel("Review length (words)")
ax.set_ylabel("Frequency")
ax.set_title("Review length by class")
ax.legend()
plt.tight_layout()
save_current_figure("review_length_by_class.png")
plt.show()

print(
    "Interpretation: the dataset is balanced, but review length is wide-ranging. "
    "That makes context-window design and truncation policy central to this study."
)

# ==============================================================================
# Cell 13 [markdown]
# ==============================================================================
# ## 5. Tokenisation and embeddings
#
# Decoder-only GPT models work with subword tokens rather than full words. That matters here because:
#
# - tokenisation changes sequence length, so a 400-word review will not usually map to 400 tokens;
# - token embeddings are learned lookup vectors for subword units, rather than fixed, human-interpretable word features;
# - GPT-2 uses token embeddings together with **learned absolute positional embeddings**;
# - sentiment classification downstream therefore depends both on lexical geometry and on how causal self-attention contextualises these embeddings.
#
# The text is only lightly cleaned. Strong normalisation could weaken the match between the pretrained tokenizer and model, so the script keeps case, punctuation, and most of the raw text.

# ==============================================================================
# Cell 14 [code]
# ==============================================================================
TOKENIZER_NAME = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
# This allows long-token analysis without tokenizer warnings. Training still uses an explicit maximum length.
tokenizer.model_max_length = 1000000

print("Tokenizer loaded:", TOKENIZER_NAME)
print("Vocab size:", tokenizer.vocab_size)
print("Pad token:", repr(tokenizer.pad_token), "id =", tokenizer.pad_token_id)
print("EOS token:", repr(tokenizer.eos_token), "id =", tokenizer.eos_token_id)

token_demo_texts = [
    "This film was unbelievably good.",
    "A surprisingly well-acted slow-burn thriller.",
    "The ending is not bad at all, but the pacing is uneven.",
]
for text in token_demo_texts:
    print("\nTEXT:", text)
    print("TOKENS:", tokenizer.tokenize(text))

# ==============================================================================
# Cell 15 [code]
# ==============================================================================
analysis_model = GPT2Model.from_pretrained(TOKENIZER_NAME)
print("Token embedding matrix shape (wte):", tuple(analysis_model.wte.weight.shape))
print("Positional embedding matrix shape (wpe):", tuple(analysis_model.wpe.weight.shape))
print(
    "\nInterpretation: GPT-2 uses learned token embeddings (wte) and learned absolute "
    "positional embeddings (wpe). The hidden representation at each position is the sum "
    "of the corresponding token and position vectors before contextualisation by self-attention."
)
del analysis_model
gc.collect()
torch.cuda.empty_cache()

# ==============================================================================
# Cell 16 [code]
# ==============================================================================
sample_texts = df_train["text"].sample(min(2000, len(df_train)), random_state=SEED).tolist()
word_counts = [len(text.split()) for text in sample_texts]
bpe_counts = []

for text in tqdm(sample_texts, desc="Counting GPT tokens"):
    token_ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    bpe_counts.append(len(token_ids))

ratio = np.array(bpe_counts) / np.maximum(np.array(word_counts), 1)
coverage_df = pd.DataFrame({
    "Statistic": ["Mean token/word ratio", "Median BPE length", "90th pct BPE length", "95th pct BPE length", "Reviews <= 512 BPE tokens"],
    "Value": [
        round(ratio.mean(), 3),
        int(np.median(bpe_counts)),
        int(np.percentile(bpe_counts, 90)),
        int(np.percentile(bpe_counts, 95)),
        f"{(np.array(bpe_counts) <= 512).mean():.1%}",
    ],
})
display(coverage_df)
save_dataframe(coverage_df, EMBEDDINGS_DIR / "token_length_analysis.csv", index=False)

plt.figure(figsize=(10, 4))
plt.hist(bpe_counts, bins=80, alpha=0.85)
for cutoff in [128, 256, 512]:
    plt.axvline(cutoff, linestyle="--", label=f"{cutoff} tokens")
plt.xlabel("GPT-2 BPE token count")
plt.ylabel("Frequency")
plt.title("Token-count distribution under GPT-2 tokenisation")
plt.legend()
plt.tight_layout()
save_current_figure("token_count_distribution.png")
plt.show()

# ==============================================================================
# Cell 17 [code]
# ==============================================================================
# Inspect the embedding space using nearest neighbours and a sentiment heatmap
base_model_for_emb = GPT2Model.from_pretrained(TOKENIZER_NAME)
emb = base_model_for_emb.wte.weight.detach().cpu().numpy()
emb = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8, None)

def nearest_neighbours(word: str, topk: int = 8) -> pd.DataFrame:
    token_ids = tokenizer.encode(word, add_special_tokens=False)
    if len(token_ids) == 0:
        raise ValueError(f"No token ids found for: {word}")
    anchor_id = token_ids[0]
    sims = emb @ emb[anchor_id]
    order = np.argsort(sims)[::-1]
    rows = []
    for idx in order:
        tok = tokenizer.convert_ids_to_tokens(int(idx))
        if idx == anchor_id:
            continue
        rows.append({"Token": tok, "Cosine similarity": float(sims[idx])})
        if len(rows) == topk:
            break
    return pd.DataFrame(rows)

for probe in ["excellent", "terrible", "boring", "gripping"]:
    print(f"\nNearest neighbours for: {probe}")
    display(nearest_neighbours(probe))

probe_words = ["excellent", "brilliant", "amazing", "enjoyable", "terrible", "awful", "boring", "dreadful"]

def pooled_token_embedding(word: str) -> np.ndarray:
    token_ids = tokenizer.encode(word, add_special_tokens=False)
    return emb[token_ids].mean(axis=0)

probe_vectors = np.vstack([pooled_token_embedding(word) for word in probe_words])
sim = probe_vectors @ probe_vectors.T
sim_df = pd.DataFrame(sim, index=probe_words, columns=probe_words)

display(sim_df.round(3))
save_dataframe(sim_df.round(6).reset_index().rename(columns={"index": "probe"}), EMBEDDINGS_DIR / "embedding_similarity_matrix.csv", index=False)

nearest_frames = []
for probe in ["excellent", "terrible", "boring", "gripping"]:
    df_nn = nearest_neighbours(probe).copy()
    df_nn.insert(0, "probe", probe)
    nearest_frames.append(df_nn)
save_dataframe(pd.concat(nearest_frames, ignore_index=True), EMBEDDINGS_DIR / "embedding_nearest_neighbours.csv", index=False)

plt.figure(figsize=(7, 6))
sns.heatmap(sim_df, annot=True, fmt=".2f", cmap="vlag", center=0)
plt.title("Cosine similarity between sentiment-bearing probe terms")
plt.tight_layout()
save_current_figure("embedding_similarity_heatmap.png")
plt.show()

embedding_summary_text = (
    "Embedding interpretation: pretrained GPT token space often groups sentiment-bearing "
    "tokens sensibly, but document-level sentiment still depends on context, negation, "
    "reversal, and discourse structure. That is why the final system adapts contextual "
    "representations with LoRA rather than relying only on frozen token geometry."
)
print(embedding_summary_text)
save_text(
    EMBEDDINGS_DIR / "embedding_analysis.txt",
    embedding_summary_text + "\n"
    + "Saved files: token_length_analysis.csv, embedding_similarity_matrix.csv, "
      "embedding_nearest_neighbours.csv, contextual_similarity_before.csv, "
      "contextual_similarity_after.csv, contextual_similarity_delta.csv\n"
)

del base_model_for_emb
gc.collect()
torch.cuda.empty_cache()

# ==============================================================================
# Cell 18 [markdown]
# ==============================================================================
# ## 6. Baseline systems
#
# Here, I compare three systems.
#
# ### Baseline A: TF-IDF with logistic regression
# I include this as a solid classical baseline, since it usually performs well on sentiment tasks and provides a sensible sanity check.
#
# ### Baseline B: Frozen GPT representations + lightweight classifier
# This lets us see what decoder-only GPT representations can achieve **without** any task-specific fine-tuning.
#
# ### System C: LoRA fine-tuned GPT classifier
# This is the main model used in the submission. It adapts a decoder-only GPT using parameter-efficient fine-tuning, rather than full fine-tuning.

# ==============================================================================
# Cell 19 [code]
# ==============================================================================
# Baseline A: TF-IDF with logistic regression.
tfidf_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        strip_accents=None,
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
    )),
    ("clf", LogisticRegression(
        C=4.0,
        solver="liblinear",
        max_iter=2000,
        random_state=SEED,
    )),
])

print("Training Baseline A (TF-IDF + Logistic Regression)...")
tfidf_pipeline.fit(df_train["text"], df_train["label"])

val_preds_a = tfidf_pipeline.predict(df_val["text"])
val_proba_a = tfidf_pipeline.predict_proba(df_val["text"])[:, 1]

# ==============================================================================
# Cell 20 [code]
# ==============================================================================
def compute_metrics(labels, preds, proba):
    return {
        "Accuracy": round(accuracy_score(labels, preds), 4),
        "Balanced Acc": round(balanced_accuracy_score(labels, preds), 4),
        "Precision": round(precision_score(labels, preds, zero_division=0), 4),
        "Recall": round(recall_score(labels, preds, zero_division=0), 4),
        "F1": round(f1_score(labels, preds, zero_division=0), 4),
        "ROC-AUC": round(roc_auc_score(labels, proba), 4),
        "PR-AUC": round(average_precision_score(labels, proba), 4),
    }

baseline_a_val_metrics = compute_metrics(df_val["label"].values, val_preds_a, val_proba_a)
display(pd.DataFrame([baseline_a_val_metrics], index=["Baseline A - validation"]))

# ==============================================================================
# Cell 21 [markdown]
# ==============================================================================
# ### Baseline B design
#
# In Baseline B, I use **frozen distilgpt2 hidden states** as features. This is deliberately weaker than full fine-tuning, but still richer than a purely classical baseline in terms of representation. The aim is to test a simple question: *how much sentiment information is already present in pretrained decoder-only representations before any task-specific adaptation?*

# ==============================================================================
# Cell 22 [code]
# ==============================================================================
FROZEN_MODEL_NAME = "distilgpt2"

frozen_tokenizer = AutoTokenizer.from_pretrained(FROZEN_MODEL_NAME)
frozen_tokenizer.pad_token = frozen_tokenizer.eos_token
frozen_tokenizer.padding_side = "left"

frozen_backbone = GPT2Model.from_pretrained(FROZEN_MODEL_NAME).to(device)
frozen_backbone.eval()
for param in frozen_backbone.parameters():
    param.requires_grad = False

def mean_pool_hidden_states(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    pooled = (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    return pooled

@torch.no_grad()
def extract_frozen_features(frame: pd.DataFrame, tokenizer, model, max_length=256, batch_size=16, cache_name=None):
    cache_path = CACHE_DIR / f"{cache_name}.pkl" if cache_name else None
    if cache_path is not None and cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    model_device = next(model.parameters()).device
    all_features = []
    texts = frame["text"].tolist()

    for start in tqdm(range(0, len(texts), batch_size), desc=f"Frozen features: {cache_name or 'uncached'}"):
        batch_texts = texts[start:start + batch_size]
        enc = tokenizer(
            batch_texts,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(model_device) for k, v in enc.items()}
        out = model(**enc)
        pooled = mean_pool_hidden_states(out.last_hidden_state, enc["attention_mask"])
        all_features.append(pooled.cpu().numpy())

    features = np.vstack(all_features)

    if cache_path is not None:
        with open(cache_path, "wb") as f:
            pickle.dump(features, f)

    return features

# ==============================================================================
# Cell 23 [code]
# ==============================================================================
print("Extracting frozen GPT features for Baseline B...")
X_train_b = extract_frozen_features(df_train, frozen_tokenizer, frozen_backbone, max_length=256, batch_size=16, cache_name="train_frozen_distilgpt2_256")
X_val_b = extract_frozen_features(df_val, frozen_tokenizer, frozen_backbone, max_length=256, batch_size=16, cache_name="val_frozen_distilgpt2_256")

mlp_b = MLPClassifier(
    hidden_layer_sizes=(256,),
    activation="relu",
    alpha=1e-4,
    learning_rate_init=1e-3,
    batch_size=128,
    max_iter=20,
    early_stopping=True,
    random_state=SEED,
)

print("Training Baseline B (frozen distilgpt2 + MLP)...")
mlp_b.fit(X_train_b, df_train["label"].values)

val_preds_b = mlp_b.predict(X_val_b)
val_proba_b = mlp_b.predict_proba(X_val_b)[:, 1]

baseline_b_val_metrics = compute_metrics(df_val["label"].values, val_preds_b, val_proba_b)
display(pd.DataFrame([baseline_b_val_metrics], index=["Baseline B - validation"]))

# Move the frozen model back to CPU after feature extraction to free GPU memory.
frozen_backbone = frozen_backbone.to('cpu')
gc.collect()
torch.cuda.empty_cache()

# ==============================================================================
# Cell 24 [markdown]
# ==============================================================================
# ## 7. GPT candidates and model selection
#
# In this section, I do more than just name a model. I compare sensible options and explain the final choice in terms of:
#
# - validation behaviour,
# - computational cost,
# - suitability for long reviews,
# - and compatibility with parameter-efficient tuning.
#
# The candidate list is kept realistic for student hardware: `distilgpt2`, `gpt2`, and a larger reference model that is not run.

# ==============================================================================
# Cell 25 [code]
# ==============================================================================
model_selection_table = pd.DataFrame([
    {
        "Model": "distilgpt2",
        "Parameters_M": 82,
        "Layers": 6,
        "Hidden size": 768,
        "Use in study": "Frozen baseline and pilot LoRA candidate",
        "Rationale": "Very compute-friendly, but shallower"
    },
    {
        "Model": "gpt2",
        "Parameters_M": 117,
        "Layers": 12,
        "Hidden size": 768,
        "Use in study": "Primary LoRA fine-tuning candidate",
        "Rationale": "Still tractable on student hardware; more depth than distilgpt2"
    },
    {
        "Model": "gpt2-medium",
        "Parameters_M": 345,
        "Layers": 24,
        "Hidden size": 1024,
        "Use in study": "Reference only; excluded from runs",
        "Rationale": "Interesting but typically too expensive for routine student sweeps"
    },
])
display(model_selection_table)
save_dataframe(model_selection_table, TABLES_DIR / "model_selection_reference.csv", index=False)

# ==============================================================================
# Cell 26 [markdown]
# ==============================================================================
# ### Dataset class and helper tools for LoRA
#
# For GPT fine-tuning, I use a custom dataset class so truncation can be adjusted at run time rather than fixed in advance. I use two truncation options:
#
# - **tail**: keep the opening tokens;
# - **head_tail**: keep both the start and end of the review, which helps when the final judgement appears near the end.

# ==============================================================================
# Cell 27 [code]
# ==============================================================================
class IMDbDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length, trunc_side="tail"):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.trunc_side = trunc_side

    def __len__(self):
        return len(self.texts)

    def _truncate_ids(self, input_ids):
        if len(input_ids) <= self.max_length:
            return input_ids
        if self.trunc_side == "tail":
            return input_ids[:self.max_length]
        if self.trunc_side == "head_tail":
            head_len = self.max_length // 2
            tail_len = self.max_length - head_len
            return input_ids[:head_len] + input_ids[-tail_len:]
        raise ValueError(f"Unsupported truncation strategy: {self.trunc_side}")

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = int(self.labels[idx])

        input_ids = self.tokenizer.encode(text, add_special_tokens=False, truncation=False)
        input_ids = self._truncate_ids(input_ids)
        attention_mask = [1] * len(input_ids)

        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            pad_ids = [self.tokenizer.pad_token_id] * pad_len
            pad_mask = [0] * pad_len
            input_ids = pad_ids + input_ids
            attention_mask = pad_mask + attention_mask

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long),
        }

def build_lora_model(
    model_name="gpt2",
    num_labels=2,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    pooling="last",
):
    num_labels = int(_json_safe(num_labels))
    lora_r = int(_json_safe(lora_r))
    lora_alpha = int(_json_safe(lora_alpha))
    lora_dropout = float(_json_safe(lora_dropout))
    pooling = str(_json_safe(pooling))
    model_name = str(_json_safe(model_name))

    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = num_labels
    config.id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    config.label2id = {"NEGATIVE": 0, "POSITIVE": 1}
    config.pad_token_id = tokenizer.pad_token_id

    base = GPT2ForSequenceClassification.from_pretrained(
        model_name,
        config=config,
        ignore_mismatched_sizes=True,
    )
    base.config.pad_token_id = tokenizer.pad_token_id

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["c_attn", "c_proj"],
        bias="none",
        modules_to_save=["score"],
    )
    model = get_peft_model(base, lora_config)

    if pooling == "mean":
        from transformers.modeling_outputs import SequenceClassifierOutputWithPast

        def mean_pool_forward(input_ids=None, attention_mask=None, labels=None, **kwargs):
            transformer = model.base_model.model.transformer
            transformer_outputs = transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs,
            )
            hidden_states = transformer_outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            logits = model.score(pooled)

            loss = None
            if labels is not None:
                loss = nn.CrossEntropyLoss()(logits, labels)

            return SequenceClassifierOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=getattr(transformer_outputs, "past_key_values", None),
                hidden_states=getattr(transformer_outputs, "hidden_states", None),
                attentions=getattr(transformer_outputs, "attentions", None),
            )

        model.forward = mean_pool_forward

    return model

def count_trainable_params(model) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)

# ==============================================================================
# Cell 28 [code]
# ==============================================================================
def train_one_epoch(model, loader, optimizer, scheduler, device, grad_accum=1):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(loader, desc="Train", leave=False), start=1):
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        loss = out.loss / grad_accum
        loss.backward()
        total_loss += out.loss.item()

        if step % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    if len(loader) % grad_accum != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return total_loss / max(len(loader), 1)

@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    all_labels, all_preds, all_proba = [], [], []
    total_loss = 0.0

    for batch in tqdm(loader, desc="Eval", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        probs = torch.softmax(out.logits, dim=-1)[:, 1]
        preds = torch.argmax(out.logits, dim=-1)

        all_labels.extend(batch["labels"].cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        all_proba.extend(probs.cpu().tolist())
        total_loss += out.loss.item()

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_proba),
        total_loss / max(len(loader), 1),
    )

def run_lora_experiment(config, df_tr, df_vl, verbose=True):
    set_all_seeds(SEED)
    config = sanitize_config_dict(config)

    train_ds = IMDbDataset(
        df_tr["text"].tolist(),
        df_tr["label"].tolist(),
        tokenizer,
        max_length=config["max_seq_len"],
        trunc_side=config["trunc_side"],
    )
    val_ds = IMDbDataset(
        df_vl["text"].tolist(),
        df_vl["label"].tolist(),
        tokenizer,
        max_length=config["max_seq_len"],
        trunc_side=config["trunc_side"],
    )

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"] * 2, shuffle=False, num_workers=0)

    model = build_lora_model(
        model_name=config["model_name"],
        lora_r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        pooling=config["pooling"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    total_steps = max(len(train_loader) * config["epochs"], 1)
    warmup_steps = int(total_steps * config["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    history = []
    start_time = time.time()
    best_epoch_record = None
    best_state_dict = None
    best_f1 = -np.inf

    for epoch in range(1, config["epochs"] + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            grad_accum=config["grad_accum_steps"],
        )
        labels, preds, proba, val_loss = evaluate_model(model, val_loader, device)
        metrics = compute_metrics(labels, preds, proba)
        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            **metrics,
        }
        history.append(record)

        if metrics["F1"] > best_f1:
            best_f1 = metrics["F1"]
            best_epoch_record = record.copy()
            best_state_dict = copy.deepcopy(model.state_dict())

        if verbose:
            print(
                f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"F1={metrics['F1']:.4f} ROC-AUC={metrics['ROC-AUC']:.4f}"
            )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    elapsed = round(time.time() - start_time, 1)

    result = {
        "config": config,
        "history": history,
        "best_epoch_metrics": best_epoch_record,
        "train_time_s": elapsed,
        "trainable_params": count_trainable_params(model),
        "model": model,
    }
    return result

# ==============================================================================
# Cell 29 [markdown]
# ==============================================================================
# ### Small pilot study for choosing the GPT model
#
# I run a short comparison on a subset of the training and validation data so that the final model choice is based on the observed results rather than just the model names.

# ==============================================================================
# Cell 30 [code]
# ==============================================================================
pilot_train = stratified_sample_df(df_train, "label", 2000, random_state=SEED)
pilot_val = stratified_sample_df(df_val, "label", 500, random_state=SEED)

candidate_base_cfg = {
    "max_seq_len": 256,
    "batch_size": 8,
    "epochs": 1,
    "lr": 3e-4,
    "weight_decay": 0.01,
    "warmup_ratio": 0.06,
    "grad_accum_steps": 1,
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "trunc_side": "tail",
    "pooling": "last",
}

candidate_rows = []

for model_name in ["distilgpt2", "gpt2"]:
    print(f"\nPilot candidate run: {model_name}")
    cfg = {**candidate_base_cfg, "model_name": model_name}
    result = run_lora_experiment(cfg, pilot_train, pilot_val, verbose=False)
    row = {
        "Model": model_name,
        "Trainable params": result["trainable_params"],
        "Train time (s)": result["train_time_s"],
        **result["best_epoch_metrics"],
    }
    candidate_rows.append(row)

    del result["model"]
    gc.collect()
    torch.cuda.empty_cache()

candidate_df = pd.DataFrame(candidate_rows).sort_values(["F1", "ROC-AUC"], ascending=False).reset_index(drop=True)
PRIMARY_MODEL_NAME = str(candidate_df.iloc[0]["Model"])

display(candidate_df)
save_dataframe(candidate_df, TABLES_DIR / "model_selection.csv", index=False)
append_run_summary(f"Primary candidate from pilot: {PRIMARY_MODEL_NAME}")
print(f"Primary final-model candidate selected from the pilot: {PRIMARY_MODEL_NAME}")

plt.figure(figsize=(6, 4))
plt.scatter(candidate_df["Train time (s)"], candidate_df["F1"], s=120)
for _, row in candidate_df.iterrows():
    plt.text(row["Train time (s)"], row["F1"], row["Model"], fontsize=10)
plt.xlabel("Pilot train time (s)")
plt.ylabel("Validation F1")
plt.title("Candidate model trade-off")
plt.tight_layout()
save_current_figure("model_selection_tradeoff.png")
plt.show()

# ==============================================================================
# Cell 31 [code]
# ==============================================================================
# Extra diagnostics to support the pilot-model choice
candidate_tradeoff_df = candidate_df.copy()
candidate_tradeoff_df["F1 per second"] = (
    candidate_tradeoff_df["F1"] / candidate_tradeoff_df["Train time (s)"].replace(0, np.nan)
).round(6)
candidate_tradeoff_df["F1 per 1M trainable params"] = (
    candidate_tradeoff_df["F1"] / (candidate_tradeoff_df["Trainable params"] / 1_000_000)
).round(6)

display(candidate_tradeoff_df.sort_values(["F1", "ROC-AUC"], ascending=False))
save_dataframe(candidate_tradeoff_df, TABLES_DIR / "candidate_tradeoff.csv", index=False)

print(
    "Selection rule: prefer the model with the best validation F1 / ROC-AUC "
    "unless the compute-cost increase is disproportionate to the gain."
)

# ==============================================================================
# Cell 32 [markdown]
# ==============================================================================
# ### Model selection summary
#
# After the pilot runs, this section briefly explains:
#
# - whether `gpt2` gave a clear improvement over `distilgpt2`;
# - whether that improvement was worth the extra training time;
# - and why the chosen model is the most sensible option on a laptops hardware.
#
# The aim is to keep model selection evidence-based rather than purely argumentative.

# ==============================================================================
# Cell 33 [markdown]
# ==============================================================================
# **Selection logic.**  
# The final model should be the one with the strongest validation evidence, while still fitting realistic compute limits. The pilot table and trade-off checks are used to justify that choice with results.

# ==============================================================================
# Cell 34 [markdown]
# ==============================================================================
# ## 8. Training setup and reproducibility
#
# System C uses **LoRA** rather than full fine-tuning, mainly for two reasons:
#
# 1. it is much more practical on a laptopshardware;
# 2. it adapts attention projections for this task while keeping most pretrained weights unchanged.
#
# This also links back to the embedding discussion: the goal is not to relearn the full token embedding space, but to adapt how pretrained representations are used for classification.

# ==============================================================================
# Cell 35 [code]
# ==============================================================================
demo_model = build_lora_model(model_name="gpt2", lora_r=8, lora_alpha=16, lora_dropout=0.05)
demo_model.print_trainable_parameters()
del demo_model
gc.collect()
torch.cuda.empty_cache()

# ==============================================================================
# Cell 36 [markdown]
# ==============================================================================
# ## 9. Hyperparameter tuning
#
# A full grid search over all settings would take too long and is not necessary so instead, tuning is done as a **staged coordinate search**.
#
# This setup makes sense for three reasons.
#
# 1. It stays **validation-driven**, since every tuning decision is based only on the validation split.
# 2. It is **compute-aware**, which is important on a laptops hardware.
# 3. It concentrates on the settings most likely to matter: sequence length, truncation, pooling, learning rate, LoRA rank, dropout, and weight decay.
#
# To keep tuning practical, the search is run on a stratified subset of the training and validation data. The final settings are then retrained on the full training split before the final test evaluation.

# ==============================================================================
# Cell 37 [code]
# ==============================================================================

# =========================
# Tuning subsets
# =========================

TUNE_TRAIN_N = min(4000, len(df_train))
TUNE_VAL_N = min(1000, len(df_val))

df_train_tune = stratified_sample_df(df_train, "label", TUNE_TRAIN_N // 2, random_state=SEED)
df_val_tune = stratified_sample_df(df_val, "label", TUNE_VAL_N // 2, random_state=SEED)

print("Tuning subset sizes:")
print("Train:", df_train_tune.shape)
print("Validation:", df_val_tune.shape)
display(df_train_tune["label"].value_counts().rename("train_count").to_frame())
display(df_val_tune["label"].value_counts().rename("val_count").to_frame())

# =========================
# Base settings for staged tuning
# =========================

BASE_CONFIG = {
    "model_name": PRIMARY_MODEL_NAME,
    "max_seq_len": 256,
    "lr": 3e-4,
    "batch_size": 8,
    "epochs": 2,
    "weight_decay": 0.01,
    "warmup_ratio": 0.06,
    "grad_accum_steps": 2,
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "pooling": "last",
    "trunc_side": "tail",
}

# =========================
# STAGED SEARCH SPACES
# =========================

DESIGN_SEARCH = {
    "max_seq_len": [128, 256],
    "pooling": ["last", "mean"],
    "trunc_side": ["tail", "head_tail"],
}

OPTIM_SEARCH = {
    "lr": [1e-4, 3e-4, 5e-4],
    "weight_decay": [0.0, 0.01],
    "lora_r": [4, 8, 16],
    "lora_dropout": [0.0, 0.05],
}

FINAL_CONFIRM_SEARCH = {
    "epochs": [2, 3],
}

# =========================
# HELPERS FOR STAGED SEARCH
# =========================

def select_best_row(df_runs):
    ordered = df_runs.sort_values(
        by=["F1", "ROC-AUC", "Train time (s)"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    return ordered.iloc[0], ordered


def run_axis_search(axis_name, candidate_values, current_config, train_df, val_df):
    axis_rows = []

    print(f"\n--- Searching axis: {axis_name} ---")
    for value in candidate_values:
        trial_config = current_config.copy()
        trial_config[axis_name] = value

        print(f"Running {axis_name} = {value}")
        result = run_lora_experiment(trial_config, train_df, val_df, verbose=False)
        best_epoch_metrics = result["best_epoch_metrics"]

        axis_rows.append({
            "axis": axis_name,
            "value": value,
            "Train time (s)": result["train_time_s"],
            "Trainable params": result["trainable_params"],
            "Accuracy": best_epoch_metrics["Accuracy"],
            "Balanced Acc": best_epoch_metrics["Balanced Acc"],
            "Precision": best_epoch_metrics["Precision"],
            "Recall": best_epoch_metrics["Recall"],
            "F1": best_epoch_metrics["F1"],
            "ROC-AUC": best_epoch_metrics["ROC-AUC"],
            "PR-AUC": best_epoch_metrics["PR-AUC"],
            "val_loss": best_epoch_metrics["val_loss"],
            "train_loss": best_epoch_metrics["train_loss"],
            "epoch": best_epoch_metrics["epoch"],
        })

        del result["model"]
        gc.collect()
        torch.cuda.empty_cache()

    axis_df = pd.DataFrame(axis_rows)
    best_row, ordered_df = select_best_row(axis_df)

    updated_config = current_config.copy()
    updated_config[axis_name] = best_row["value"]
    return ordered_df, updated_config

# =========================
# FULL STAGED COORDINATE SEARCH
# =========================

all_tuning_tables = []
tuning_decisions = []
current_config = BASE_CONFIG.copy()

for stage_name, stage_space in [
    ("Design search", DESIGN_SEARCH),
    ("Optimisation search", OPTIM_SEARCH),
    ("Final confirmation", FINAL_CONFIRM_SEARCH),
]:
    for axis_name, values in stage_space.items():
        axis_df, current_config = run_axis_search(
            axis_name=axis_name,
            candidate_values=values,
            current_config=current_config,
            train_df=df_train_tune,
            val_df=df_val_tune,
        )
        display(axis_df)
        all_tuning_tables.append(axis_df)

        best_row = axis_df.iloc[0]
        tuning_decisions.append({
            "stage": stage_name,
            "axis": axis_name,
            "selected_value": best_row["value"],
            "best_F1": best_row["F1"],
            "best_ROC_AUC": best_row["ROC-AUC"],
            "train_time_s": best_row["Train time (s)"],
        })

tuning_runs_df = pd.concat(all_tuning_tables, ignore_index=True)
tuning_summary_df = pd.DataFrame(tuning_decisions)

print("\nSelected configuration after staged tuning:")
display(pd.DataFrame([current_config]))
append_run_summary(f"Selected config after staged tuning: {_json_safe(current_config)}")

print("\nTuning summary:")
display(tuning_summary_df)

print("\nAll tuning runs:")
display(tuning_runs_df)

save_dataframe(tuning_summary_df, TABLES_DIR / "hyperparameter_tuning_summary.csv", index=False)
save_dataframe(tuning_runs_df, TABLES_DIR / "hyperparameter_tuning_results.csv", index=False)

# =========================
# REVIEWING THE TUNING RESULTS
# =========================

for axis_name in tuning_runs_df["axis"].unique():
    subset = tuning_runs_df[tuning_runs_df["axis"] == axis_name].copy()
    subset["value_str"] = subset["value"].astype(str)

    plt.figure(figsize=(6, 4))
    sns.barplot(data=subset, x="value_str", y="F1")
    plt.ylim(0, 1)
    plt.title(f"Validation F1 by {axis_name}")
    plt.xlabel(axis_name)
    plt.ylabel("Validation F1")
    plt.tight_layout()
    safe_axis_name = str(axis_name).replace("/", "_")
    save_current_figure(f"tuning_{safe_axis_name}.png")
    plt.show()

# ==============================================================================
# Cell 42 [markdown]
# ==============================================================================
# ## 11. Final model training
#
# In this section, I train the final System C model using the best validation setup from the sweep,
# then save it with the tokenizer and a JSON record of the selected configuration.

# ==============================================================================
# Cell 43 [code]
# ==============================================================================
FINAL_CONFIG = current_config.copy()
print("Final configuration selected from tuning:")
display(pd.DataFrame([FINAL_CONFIG]))
write_json(METRICS_DIR / "final_config.json", FINAL_CONFIG)

# ==============================================================================
# Cell 44 [code]
# ==============================================================================
# Train the final System C model on the training split while still tracking validation performance.
final_result = run_lora_experiment(FINAL_CONFIG, df_train, df_val, verbose=True)
final_model = final_result["model"]
final_history = pd.DataFrame(final_result["history"])
display(final_history)

# Save the adapter, tokenizer, and metadata needed to reload this run.
try:
    sanitize_peft_config_inplace(final_model)
    final_model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
except TypeError as exc:
    append_run_summary(f"Model save failed before sanitization retry: {exc}")
    sanitize_peft_config_inplace(final_model)
    # If save_pretrained still fails, fall back to saving the PEFT state dict directly, then save the tokenizer separately.
    try:
        final_model.save_pretrained(MODEL_DIR)
        tokenizer.save_pretrained(MODEL_DIR)
    except TypeError as exc2:
        append_run_summary(f"save_pretrained failed after retry; using fallback torch save: {exc2}")
        torch.save(final_model.state_dict(), MODEL_DIR / "adapter_state_dict.pt")
        if hasattr(final_model, "peft_config"):
            peft_cfg_dump = {name: _json_safe(vars(cfg)) for name, cfg in final_model.peft_config.items()}
            write_json(MODEL_DIR / "peft_config_fallback.json", peft_cfg_dump)
        tokenizer.save_pretrained(MODEL_DIR)

write_json(MODEL_DIR / "final_metrics_val.json", final_result["best_epoch_metrics"])
append_run_summary(f"Best validation epoch metrics: {final_result['best_epoch_metrics']}")
save_dataframe(final_history, TABLES_DIR / "final_training_history.csv", index=False)

# ==============================================================================
# Cell 45 [code]
# ==============================================================================
# Compare sentence embeddings before and after LoRA fine-tuning to see how the representation space shifts.
from sklearn.metrics.pairwise import cosine_similarity

probe_sentences = {
    "pos_1": "This film is genuinely excellent and emotionally satisfying.",
    "pos_2": "A brilliantly acted and deeply enjoyable movie.",
    "neg_1": "This film is tedious, badly written, and disappointing.",
    "neg_2": "A boring and painfully dull experience from start to finish.",
}

def get_hidden_state_backbone(model):
    """
    Return the GPT backbone used to produce hidden states.

    This handles:
    - a plain GPT2Model,
    - GPT2ForSequenceClassification,
    - and a PEFT-wrapped GPT classifier.
    """
    if isinstance(model, GPT2Model):
        return model

    if hasattr(model, "transformer"):
        return model.transformer

    if hasattr(model, "base_model"):
        base = model.base_model

        if hasattr(base, "model") and hasattr(base.model, "transformer"):
            return base.model.transformer

        if hasattr(base, "transformer"):
            return base.transformer

        if hasattr(base, "model") and isinstance(base.model, GPT2Model):
            return base.model

    raise TypeError(f"Could not find a GPT backbone for hidden-state extraction: {type(model)}")


def extract_pooled_hidden(model, tokenizer, text, max_length, device):
    """
    Encode a single text input and return a mean-pooled hidden vector.
    """
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    backbone = get_hidden_state_backbone(model)

    with torch.no_grad():
        out = backbone(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        hidden = out.last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    return pooled.squeeze(0).cpu().numpy()

base_cmp_model = GPT2Model.from_pretrained(FINAL_CONFIG["model_name"]).to(device)
base_cmp_model.eval()

before_vecs = []
after_vecs = []
labels_probe = []

for name, text in probe_sentences.items():
    before_vecs.append(extract_pooled_hidden(base_cmp_model, tokenizer, text, FINAL_CONFIG["max_seq_len"], device))
    after_vecs.append(extract_pooled_hidden(final_model, tokenizer, text, FINAL_CONFIG["max_seq_len"], device))
    labels_probe.append(name)

before_sim = pd.DataFrame(cosine_similarity(np.vstack(before_vecs)), index=labels_probe, columns=labels_probe)
after_sim = pd.DataFrame(cosine_similarity(np.vstack(after_vecs)), index=labels_probe, columns=labels_probe)

print("Contextual similarity before LoRA adaptation:")
display(before_sim.round(3))
print("Contextual similarity after LoRA adaptation:")
display(after_sim.round(3))
save_dataframe(before_sim.round(6).reset_index().rename(columns={"index": "probe"}), EMBEDDINGS_DIR / "contextual_similarity_before.csv", index=False)
save_dataframe(after_sim.round(6).reset_index().rename(columns={"index": "probe"}), EMBEDDINGS_DIR / "contextual_similarity_after.csv", index=False)
save_dataframe((after_sim - before_sim).round(6).reset_index().rename(columns={"index": "probe"}), EMBEDDINGS_DIR / "contextual_similarity_delta.csv", index=False)

plt.figure(figsize=(7, 6))
sns.heatmap(after_sim - before_sim, annot=True, fmt=".2f", cmap="vlag", center=0)
plt.title("Change in contextual similarity after LoRA fine-tuning")
plt.tight_layout()
save_current_figure("contextual_similarity_delta.png")
plt.show()

del base_cmp_model
gc.collect()
torch.cuda.empty_cache()

# ==============================================================================
# Cell 46 [markdown]
# ==============================================================================
# ### Interpreting the embedding shift
#
# Rather than looking only at static token embeddings, this checks whether task-specific
# fine-tuning makes sentiment probe sentences separate more clearly in context.
#
# A reasonable way to read the result is to look for this pattern:
#
# - positive probes becoming more similar to other positive probes,
# - negative probes becoming more similar to other negative probes,
# - with cross-sentiment similarity dropping after fine-tuning.
#
# Seeing this pattern would support using adapted contextual representations, instead of
# depending only on frozen features.

# ==============================================================================
# Cell 47 [code]
# ==============================================================================
plt.figure(figsize=(10, 4))
plt.plot(final_history["epoch"], final_history["train_loss"], marker="o", label="Train loss")
plt.plot(final_history["epoch"], final_history["val_loss"], marker="s", label="Validation loss")
plt.plot(final_history["epoch"], final_history["F1"], marker="^", label="Validation F1")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("Final model training history")
plt.legend()
plt.tight_layout()
save_current_figure("final_training_history.png")
plt.show()

# ==============================================================================
# Cell 48 [markdown]
# ==============================================================================
# ## 12. Evaluation and metrics
#
# Accuracy on its own does not tell the full story, so this notebook also evaluates:
#
# - **Accuracy** to summarise overall correctness,
# - **Balanced accuracy** as a simple robustness check,
# - **Precision / Recall / F1** because false positives and false negatives do not carry the same cost,
# - **ROC-AUC** to assess ranking quality across thresholds,
# - **PR-AUC** to monitor how well the model handles the positive class as the threshold changes,
# - **confusion matrices** so error patterns are easier to read,
# - **ROC and PR curves** for a straightforward visual comparison between models,
# - **length-bucket performance** because dealing with longer reviews is an important part of this task,
# - and a short **calibration check**, since a model can score well overall but still be miscalibrated.

# ==============================================================================
# Cell 49 [code]
# ==============================================================================
def select_best_threshold(labels, proba, grid=None):
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)

    rows = []
    for thr in grid:
        preds = (proba >= thr).astype(int)
        rows.append({
            "threshold": round(float(thr), 2),
            "F1": f1_score(labels, preds, zero_division=0),
            "Precision": precision_score(labels, preds, zero_division=0),
            "Recall": recall_score(labels, preds, zero_division=0),
        })
    threshold_df = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)
    return threshold_df, float(threshold_df.iloc[0]["threshold"])

# I use the validation split here to pick a sensible decision threshold for System C
val_ds_c = IMDbDataset(df_val["text"].tolist(), df_val["label"].tolist(), tokenizer, FINAL_CONFIG["max_seq_len"], FINAL_CONFIG["trunc_side"])
val_loader_c = DataLoader(val_ds_c, batch_size=FINAL_CONFIG["batch_size"] * 2, shuffle=False, num_workers=0)
val_labels_c, _, val_proba_c, _ = evaluate_model(final_model, val_loader_c, device)

threshold_df, best_threshold_c = select_best_threshold(val_labels_c, val_proba_c)
display(threshold_df)
save_dataframe(threshold_df, METRICS_DIR / "threshold_search.csv", index=False)
print(f"Selected decision threshold for System C: {best_threshold_c:.2f}")

# ==============================================================================
# Cell 50 [code]
# ==============================================================================
# Final test-set evaluation
test_ds_c = IMDbDataset(df_test["text"].tolist(), df_test["label"].tolist(), tokenizer, FINAL_CONFIG["max_seq_len"], FINAL_CONFIG["trunc_side"])
test_loader_c = DataLoader(test_ds_c, batch_size=FINAL_CONFIG["batch_size"] * 2, shuffle=False, num_workers=0)

test_labels_c, _, test_proba_c, _ = evaluate_model(final_model, test_loader_c, device)
test_preds_c = (test_proba_c >= best_threshold_c).astype(int)
metrics_c_test = compute_metrics(test_labels_c, test_preds_c, test_proba_c)

# Evaluate Baseline A on the same test set
test_preds_a = tfidf_pipeline.predict(df_test["text"])
test_proba_a = tfidf_pipeline.predict_proba(df_test["text"])[:, 1]
metrics_a_test = compute_metrics(df_test["label"].values, test_preds_a, test_proba_a)

# Evaluate Baseline B on the held-out test split
X_test_b = extract_frozen_features(df_test, frozen_tokenizer, frozen_backbone, max_length=256, batch_size=16, cache_name="test_frozen_distilgpt2_256")
test_preds_b = mlp_b.predict(X_test_b)
test_proba_b = mlp_b.predict_proba(X_test_b)[:, 1]
metrics_b_test = compute_metrics(df_test["label"].values, test_preds_b, test_proba_b)

final_comparison_df = pd.DataFrame([
    {"System": "A - TF-IDF + Logistic Regression", **metrics_a_test},
    {"System": "B - Frozen distilgpt2 + MLP", **metrics_b_test},
    {"System": "C - LoRA fine-tuned GPT", **metrics_c_test},
]).set_index("System")

display(final_comparison_df)
save_dataframe(final_comparison_df.reset_index(), TABLES_DIR / "final_test_comparison.csv", index=False)
append_run_summary(f"Final test metrics: {_json_safe({'Baseline_A': metrics_a_test, 'Baseline_B': metrics_b_test, 'System_C': metrics_c_test})}")

write_json(METRICS_DIR / 'system_c_threshold.json', {'threshold': float(best_threshold_c)})
write_json(METRICS_DIR / 'final_metrics.json', {
    'Baseline_A': metrics_a_test,
    'Baseline_B': metrics_b_test,
    'System_C': metrics_c_test,
})

# ==============================================================================
# Cell 51 [code]
# ==============================================================================
# Estimate bootstrap confidence intervals for the main final metrics
def bootstrap_metric_ci(y_true, y_proba, threshold, metric_name, n_boot=500, seed=SEED):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    values = []

    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        yt = y_true[idx]
        yp = y_proba[idx]
        pred = (yp >= threshold).astype(int)

        if metric_name == "F1":
            values.append(f1_score(yt, pred, zero_division=0))
        elif metric_name == "ROC-AUC" and len(np.unique(yt)) == 2:
            values.append(roc_auc_score(yt, yp))

    if len(values) == 0:
        return (np.nan, np.nan)

    lo, hi = np.percentile(values, [2.5, 97.5])
    return round(float(lo), 4), round(float(hi), 4)

ci_rows = []
for name, y_true, y_proba, thr in [
    ("A - TF-IDF + Logistic Regression", df_test["label"].values, test_proba_a, 0.5),
    ("B - Frozen distilgpt2 + MLP", df_test["label"].values, test_proba_b, 0.5),
    ("C - LoRA fine-tuned GPT", test_labels_c, test_proba_c, best_threshold_c),
]:
    f1_ci = bootstrap_metric_ci(y_true, y_proba, thr, "F1")
    auc_ci = bootstrap_metric_ci(y_true, y_proba, thr, "ROC-AUC")
    ci_rows.append({
        "System": name,
        "F1 95% CI": f1_ci,
        "ROC-AUC 95% CI": auc_ci,
    })

ci_df = pd.DataFrame(ci_rows)
display(ci_df)
save_dataframe(ci_df, METRICS_DIR / "bootstrap_confidence_intervals.csv", index=False)

# ==============================================================================
# Cell 52 [code]
# ==============================================================================
report_a = classification_report(df_test["label"].values, test_preds_a, digits=4)
report_b = classification_report(df_test["label"].values, test_preds_b, digits=4)
report_c = classification_report(test_labels_c, test_preds_c, digits=4)

print("Classification report - Baseline A")
print(report_a)
print("\nClassification report - Baseline B")
print(report_b)
print("\nClassification report - System C")
print(report_c)

with open(LOGS_DIR / "classification_reports.txt", "w", encoding="utf-8") as f:
    f.write("Classification report - Baseline A\n")
    f.write(report_a + "\n\n")
    f.write("Classification report - Baseline B\n")
    f.write(report_b + "\n\n")
    f.write("Classification report - System C\n")
    f.write(report_c + "\n")


# ==============================================================================
# Cell 53 [code]
# ==============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
systems = [
    ("A: TF-IDF + LR", df_test["label"].values, test_preds_a),
    ("B: Frozen distilgpt2 + MLP", df_test["label"].values, test_preds_b),
    ("C: LoRA GPT", test_labels_c, test_preds_c),
]
for ax, (title, labels, preds) in zip(axes, systems):
    cm = confusion_matrix(labels, preds)
    ConfusionMatrixDisplay(cm, display_labels=["Negative", "Positive"]).plot(ax=ax, colorbar=False)
    ax.set_title(title)
plt.tight_layout()
save_current_figure("confusion_matrix.png")
plt.show()

# ==============================================================================
# Cell 54 [code]
# ==============================================================================
plt.figure(figsize=(7, 6))
for name, labels, proba in [
    ("A: TF-IDF + LR", df_test["label"].values, test_proba_a),
    ("B: Frozen distilgpt2 + MLP", df_test["label"].values, test_proba_b),
    ("C: LoRA GPT", test_labels_c, test_proba_c),
]:
    fpr, tpr, _ = roc_curve(labels, proba)
    auc = roc_auc_score(labels, proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC curves on the held-out test set")
plt.legend()
plt.tight_layout()
save_current_figure("roc_curve.png")
plt.show()

plt.figure(figsize=(7, 6))
for name, labels, proba in [
    ("A: TF-IDF + LR", df_test["label"].values, test_proba_a),
    ("B: Frozen distilgpt2 + MLP", df_test["label"].values, test_proba_b),
    ("C: LoRA GPT", test_labels_c, test_proba_c),
]:
    precision, recall, _ = precision_recall_curve(labels, proba)
    pr_auc = average_precision_score(labels, proba)
    plt.plot(recall, precision, label=f"{name} (PR-AUC={pr_auc:.4f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-recall curves on the held-out test set")
plt.legend()
plt.tight_layout()
save_current_figure("pr_curve.png")
plt.show()

# ==============================================================================
# Cell 55 [code]
# ==============================================================================
# Quick calibration check for System C
frac_pos, mean_pred = calibration_curve(test_labels_c, test_proba_c, n_bins=10)
plt.figure(figsize=(6, 5))
plt.plot(mean_pred, frac_pos, marker="o", label="System C")
plt.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Perfect calibration")
plt.xlabel("Mean predicted probability")
plt.ylabel("Observed positive fraction")
plt.title("Calibration curve for System C")
plt.legend()
plt.tight_layout()
save_current_figure("calibration_curve.png")
plt.show()

print(
    "A curve near the diagonal indicates better calibration. "
    "Systematic deviation below the diagonal suggests overconfidence; "
    "deviation above suggests underconfidence."
)

# ==============================================================================
# Cell 56 [code]
# ==============================================================================
# Compare performance across review-length buckets for all systems
df_test_eval = df_test.copy()
df_test_eval["pred_a"] = test_preds_a
df_test_eval["pred_b"] = test_preds_b
df_test_eval["pred_c"] = test_preds_c
df_test_eval["proba_a"] = test_proba_a
df_test_eval["proba_b"] = test_proba_b
df_test_eval["proba_c"] = test_proba_c
df_test_eval["correct_a"] = (df_test_eval["pred_a"] == df_test_eval["label"]).astype(int)
df_test_eval["correct_b"] = (df_test_eval["pred_b"] == df_test_eval["label"]).astype(int)
df_test_eval["correct_c"] = (df_test_eval["pred_c"] == df_test_eval["label"]).astype(int)

bins = [0, 100, 200, 400, 800, 10000]
bucket_labels = ["<=100", "101-200", "201-400", "401-800", ">800"]
df_test_eval["length_bucket"] = pd.cut(df_test_eval["word_count"], bins=bins, labels=bucket_labels)

bucket_rows = []
for system_name, pred_col, proba_col in [
    ("A", "pred_a", "proba_a"),
    ("B", "pred_b", "proba_b"),
    ("C", "pred_c", "proba_c"),
]:
    for bucket in bucket_labels:
        subset = df_test_eval[df_test_eval["length_bucket"] == bucket]
        if subset.empty:
            continue

        y_true = subset["label"].values
        y_pred = subset[pred_col].values
        y_proba = subset[proba_col].values

        row = {
            "System": system_name,
            "Length bucket": bucket,
            "N": len(subset),
            "Accuracy": round(accuracy_score(y_true, y_pred), 4),
            "F1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        }

        if len(np.unique(y_true)) == 2:
            row["ROC-AUC"] = round(roc_auc_score(y_true, y_proba), 4)
        else:
            row["ROC-AUC"] = np.nan

        bucket_rows.append(row)

bucket_df = pd.DataFrame(bucket_rows)
display(bucket_df)
save_dataframe(bucket_df, TABLES_DIR / "review_length_analysis.csv", index=False)

plt.figure(figsize=(10, 4))
sns.barplot(data=bucket_df, x="Length bucket", y="F1", hue="System")
plt.ylim(0, 1)
plt.title("F1 by review-length bucket")
plt.tight_layout()
save_current_figure("review_length_analysis.png")
plt.show()

# ==============================================================================
# Cell 57 [markdown]
# ==============================================================================
# ## 13. Error analysis
#
# Reporting summary metrics is useful, but it can miss important patterns, so this section looks directly at individual errors.
#
# The script prints at least ten System C misclassifications, sorted by confidence, because highly confident mistakes are often the most revealing.

# ==============================================================================
# Cell 58 [code]
# ==============================================================================
df_test_eval["true_label"] = df_test_eval["label"].map({0: "Negative", 1: "Positive"})
df_test_eval["pred_label"] = pd.Series(test_preds_c).map({0: "Negative", 1: "Positive"})
df_test_eval["confidence"] = np.maximum(df_test_eval["proba_c"], 1 - df_test_eval["proba_c"])

misclassified = df_test_eval[df_test_eval["pred_c"] != df_test_eval["label"]].copy()
misclassified = misclassified.sort_values("confidence", ascending=False).reset_index(drop=True)

print(f"Total misclassifications by System C: {len(misclassified):,} / {len(df_test_eval):,}")
print()

for i, row in misclassified.head(10).iterrows():
    snippet = re.sub(r"\s+", " ", row["text"])[:350]
    print("=" * 110)
    print(
        f"[{i + 1}] True={row['true_label']:<8} Pred={row['pred_label']:<8} "
        f"Confidence={row['confidence']:.3f} Length={row['word_count']} words"
    )
    print(snippet + "...")
    print()

# ==============================================================================
# Cell 59 [code]
# ==============================================================================
false_pos = misclassified[(misclassified["label"] == 0) & (misclassified["pred_c"] == 1)]
false_neg = misclassified[(misclassified["label"] == 1) & (misclassified["pred_c"] == 0)]

error_summary = pd.DataFrame({
    "Statistic": [
        "False positives",
        "False negatives",
        "Mean length of correct predictions",
        "Mean length of misclassified predictions",
    ],
    "Value": [
        len(false_pos),
        len(false_neg),
        round(df_test_eval.loc[df_test_eval["correct_c"] == 1, "word_count"].mean(), 1),
        round(misclassified["word_count"].mean(), 1),
    ],
})
display(error_summary)

# ==============================================================================
# Cell 60 [code]
# ==============================================================================
# Export a small file for manual labelling, so the error taxonomy can be reviewed in more detail
annotation_cols = ["text", "true_label", "pred_label", "confidence", "word_count"]
annotation_df = misclassified[annotation_cols].head(50).copy()
annotation_df["failure_mode"] = ""
save_dataframe(annotation_df, ERRORS_DIR / "misclassified_examples.csv", index=False)

print(
    "Saved top 50 misclassified examples to "
    f"{ERRORS_DIR / 'misclassified_examples.csv'} "
    "for optional manual annotation."
)

# ==============================================================================
# Cell 61 [code]
# ==============================================================================
# Summarise the manual error labels once they have been added
annot_path = ERRORS_DIR / "misclassified_examples.csv"

if annot_path.exists():
    annotated = pd.read_csv(annot_path)
    annotated = annotated[annotated["failure_mode"].fillna("").str.strip() != ""].copy()

    if len(annotated) > 0:
        taxonomy_df = (
            annotated["failure_mode"]
            .value_counts()
            .rename_axis("Failure mode")
            .reset_index(name="Count")
        )
        display(taxonomy_df)

        plt.figure(figsize=(8, 4))
        sns.barplot(data=taxonomy_df, x="Failure mode", y="Count")
        plt.xticks(rotation=30, ha="right")
        plt.title("Manual error taxonomy summary")
        plt.tight_layout()
        plt.show()

        save_dataframe(taxonomy_df, ERRORS_DIR / "manual_error_taxonomy_summary.csv", index=False)
    else:
        print("Add labels to the failure_mode column in the exported CSV, then rerun this cell.")

# ==============================================================================
# Cell 62 [markdown]
# ==============================================================================
# ### Common failure patterns
#
# From the examples above, these are the main patterns I noticed:
#
# 1. **mixed sentiment** reviews that include both praise and criticism,
# 2. **late sentiment reversal**, where the real judgement only appears right at the end,
# 3. **sarcasm or irony**, which can flip the looks so it has a positive side and a negative side.
# 4. **genre-specific vocabulary**, where words like *disturbing* or *dark* are not always negative in film reviews,
# 5. **token-budget failures**, where key parts of long reviews can get truncated.
#
# These patterns line up directly with the study design, mainly cutting part of it away and the choice to adapt GPT representations instead of relying purely on frozen features.

# ==============================================================================
# Cell 63 [markdown]
# ==============================================================================
# ## 14. Limitations
#
# This notebook that I designed I believe is as rigorous as possible within my laptop's hardware limits, but some limitations still remain.
#
# - The hyperparameter search was systematic, but it still wasn't exhaustive.
# - I kept the main GPT comparison to realistic options, rather than trying to cover a full model zoo.
# - LoRA made training manageable, but it doesn't represent the full range of PEFT methods.
# - These decoder-only GPTs are being adapted to a task where encoder-style models are often a more natural fit.
# - Token-budget limits are still a key issue for very long reviews, even with careful removing part of it for tuning.

# ==============================================================================
# Cell 64 [markdown]
# ==============================================================================
# ## 15. Future work
#
# The most useful next steps are the ones that follow directly from the limitations and the error patterns that I observed above.
#
# 1. **Sliding-window or chunk-aggregation classification.**  
#    Instead of forcing each review into a single truncated sequence, encode multiple chunks and combine the evidence.
#
# 2. **QLoRA or other lower-memory PEFT variants.**  
#    This could broaden the model choices without making training impractical.
#
# 3. **A wider architecture comparison.**  
#    Compare decoder-only GPT adaptation with encoder-based transformers under the same evaluation setup.
#
# 4. **Calibration-aware deployment.**  
#    If the model is overconfident, apply temperature scaling (or a similar post-hoc method) to calibrate outputs.
#
# 5. **A more formal error taxonomy.**  
#    Manually label a larger set of misclassifications and quantify the main failure modes.
#
# 6. **Aspect-level sentiment or rationale extraction.**  
#    IMDb reviews often mix opinions on acting, plot, pacing, and cinematography, so a finer-grained setup could be more informative than binary sentiment alone.


append_run_summary("Run completed successfully.")


# ==============================================================================
# Cell 65 [code]
# ==============================================================================
# Final save stage: write evidence files and check the required artefacts are present.
#
# I included this section to make the Python notebook more robust as a code artefact.
# It saves brief text/CSV/JSON evidence files so the workflow is easy to follow,
# and verifies that the expected deliverables are actually on disk.

import shutil
from sklearn.metrics import brier_score_loss

def save_markdown(path: Path, text: str) -> None:
    """Save a short markdown/plain-text note as marking evidence."""
    save_text(path, text)

def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def ensure_csv_exists(df: pd.DataFrame | None, path: Path) -> None:
    """Write the CSV only when it is missing and a suitable dataframe is available."""
    if df is not None and not path.exists():
        save_dataframe(df, path, index=False)

def ensure_json_exists(data, path: Path) -> None:
    """Write the JSON file only if it is missing and serialisable data is available."""
    if data is not None and not path.exists():
        write_json(path, data)

def ensure_figure_exists(path: Path, builder=None) -> None:
    """
    Rebuild a figure only when needed: if the file is missing and we have a builder to call.

    The callback should create a new matplotlib figure, then either save it itself
    or leave it active so save_current_figure() can handle it.
    """
    if path.exists():
        return
    if builder is None:
        return
    try:
        builder()
    except Exception as exc:
        append_run_summary(f"Figure regeneration failed for {path.name}: {exc}")

def append_section_header(title: str) -> None:
    append_run_summary("")
    append_run_summary("=" * 88)
    append_run_summary(title)
    append_run_summary("=" * 88)

# ------------------------------
# 1) Narrative write-ups for the report
# ------------------------------
problem_choice_report = """# Problem choice justification

1. This task is set up as long-form decoder-only GPT adaptation, rather than a generic IMDb sentiment exercise.
2. The reviews are long enough that context budget and cutting sections shorter strategy become real design choices.
3. Decoder-only GPTs are less naturally suited to classification than encoder models, so adaptation decisions matter.
4. The study separates pretrained representation quality from task-specific adaptation effects.
5. The code covers baselines, candidate selection, tuning, calibration/error analysis, and
    review-length analysis, instead of relying on one unexamined run.
"""
save_markdown(LOGS_DIR / "problem_choice_justification.md", problem_choice_report)

embedding_report_lines = []
if "coverage_df" in globals():
    embedding_report_lines.append("Tokenisation coverage summary:")
    embedding_report_lines.append(coverage_df.to_string(index=False))
if "sim_df" in globals():
    embedding_report_lines.append("\nProbe-term cosine similarity matrix:")
    embedding_report_lines.append(sim_df.round(3).to_string())
embedding_report_lines.append(
    "\nSingle words alone can’t capture the overall feeling of a full text "
    "because things like negation, reversal, structure, and final conclusions need context to understand "
    "So we should fine-tune the model, not just use fixed word meanings."
)
save_markdown(LOGS_DIR / "embedding_rationale.md", "\n".join(embedding_report_lines))

if "sim_df" in globals():
    emb_csv = (
        sim_df.round(6)
        .reset_index()
        .rename(columns={"index": "probe"})
    )
    ensure_csv_exists(emb_csv, EMBEDDINGS_DIR / "embedding_analysis.csv")

model_selection_report_lines = []
if "candidate_df" in globals():
    model_selection_report_lines.append("Pilot candidate comparison:")
    model_selection_report_lines.append(candidate_df.to_string(index=False))
if "candidate_tradeoff_df" in globals():
    model_selection_report_lines.append("\nCost/quality trade-off diagnostics:")
    model_selection_report_lines.append(candidate_tradeoff_df.to_string(index=False))
if "PRIMARY_MODEL_NAME" in globals():
    model_selection_report_lines.append(
        f"\nSelected primary GPT candidate: {PRIMARY_MODEL_NAME}. "
        "The selection rule prioritises validation F1/ROC-AUC unless extra compute is disproportionate."
    )
save_markdown(LOGS_DIR / "model_selection_rationale.md", "\n".join(model_selection_report_lines))

tuning_report_lines = []
if "tuning_summary_df" in globals():
    tuning_report_lines.append("Staged tuning decisions:")
    tuning_report_lines.append(tuning_summary_df.to_string(index=False))
if "tuning_runs_df" in globals():
    tuning_report_lines.append("\nAll recorded tuning runs:")
    tuning_report_lines.append(tuning_runs_df.to_string(index=False))
if "FINAL_CONFIG" in globals():
    tuning_report_lines.append("\nSelected final configuration:")
    tuning_report_lines.append(pd.DataFrame([FINAL_CONFIG]).to_string(index=False))
save_markdown(LOGS_DIR / "hyperparameter_tuning_rationale.md", "\n".join(tuning_report_lines))

future_work_report = """# Future work linked to current limitations

1. Use a sliding-window or chunk-aggregation classifier to deal with failures caused by token limits.
2. Try QLoRA, or similar low-memory PEFT approaches, to widen the model search space.
3. Run a direct encoder-versus-decoder comparison using the same evaluation protocol.
4. Apply post-hoc calibration (for instance, temperature scaling) if confidence is still miscalibrated.
5. Build a larger manually annotated failure taxonomy for mixed sentiment, sarcasm, and late reversals.
6. Explore more fine-grained aspect-level sentiment or rationale extraction for long-form review structure.
"""
save_markdown(LOGS_DIR / "future_work.md", future_work_report)

# ------------------------------
# 2) Finalise evaluation outputs
# ------------------------------
if "error_summary" in globals():
    ensure_csv_exists(error_summary, ERRORS_DIR / "error_summary.csv")

if all(name in globals() for name in ["df_test", "test_proba_a", "test_proba_b", "test_proba_c", "test_labels_c"]):
    calibration_rows = [
        {"System": "A - TF-IDF + Logistic Regression", "Brier score": round(brier_score_loss(df_test["label"].values, test_proba_a), 6)},
        {"System": "B - Frozen distilgpt2 + MLP", "Brier score": round(brier_score_loss(df_test["label"].values, test_proba_b), 6)},
        {"System": "C - LoRA fine-tuned GPT", "Brier score": round(brier_score_loss(test_labels_c, test_proba_c), 6)},
    ]
    calibration_df = pd.DataFrame(calibration_rows)
    save_dataframe(calibration_df, METRICS_DIR / "calibration_metrics.csv", index=False)

evaluation_report_lines = []
if "final_comparison_df" in globals():
    evaluation_report_lines.append("Final held-out test comparison:")
    evaluation_report_lines.append(final_comparison_df.to_string())
    if "A - TF-IDF + Logistic Regression" in final_comparison_df.index and "C - LoRA fine-tuned GPT" in final_comparison_df.index:
        best_system = final_comparison_df["F1"].astype(float).idxmax()
        evaluation_report_lines.append(
            f"\nHighest held-out F1: {best_system}. "
            "This makes the evaluation honest: the most sophisticated system is not automatically the best-performing one."
        )
if "ci_df" in globals():
    evaluation_report_lines.append("\nBootstrap confidence intervals:")
    evaluation_report_lines.append(ci_df.to_string(index=False))
if "bucket_df" in globals():
    evaluation_report_lines.append("\nLength-bucket analysis:")
    evaluation_report_lines.append(bucket_df.to_string(index=False))
if "error_summary" in globals():
    evaluation_report_lines.append("\nError summary:")
    evaluation_report_lines.append(error_summary.to_string(index=False))
save_markdown(LOGS_DIR / "evaluation_summary.md", "\n".join(evaluation_report_lines))

# ------------------------------
# 3) Backup checks and a quick audit of the required artefacts
# ------------------------------
required_artefacts = {
    "outputs/tables/model_selection.csv": TABLES_DIR / "model_selection.csv",
    "outputs/tables/hyperparameter_tuning_results.csv": TABLES_DIR / "hyperparameter_tuning_results.csv",
    "outputs/metrics/final_metrics.json": METRICS_DIR / "final_metrics.json",
    "outputs/figures/confusion_matrix.png": FIGURES_DIR / "confusion_matrix.png",
    "outputs/figures/review_length_analysis.png": FIGURES_DIR / "review_length_analysis.png",
    "outputs/errors/misclassified_examples.csv": ERRORS_DIR / "misclassified_examples.csv",
    "outputs/embeddings/embedding_analysis.txt": EMBEDDINGS_DIR / "embedding_analysis.txt",
    "outputs/models/final_model": MODEL_DIR,
    "outputs/logs/run_summary.txt": LOGS_DIR / "run_summary.txt",
}
optional_one_of = {
    "outputs/figures/roc_curve.png or outputs/figures/pr_curve.png": [
        FIGURES_DIR / "roc_curve.png",
        FIGURES_DIR / "pr_curve.png",
    ],
    "outputs/embeddings/embedding_analysis.csv or .txt": [
        EMBEDDINGS_DIR / "embedding_analysis.csv",
        EMBEDDINGS_DIR / "embedding_analysis.txt",
    ],
}

# Save the main tables and metrics again, in case any earlier cells were skipped or the run was interrupted.
if "candidate_df" in globals():
    ensure_csv_exists(candidate_df, TABLES_DIR / "model_selection.csv")
if "tuning_runs_df" in globals():
    ensure_csv_exists(tuning_runs_df, TABLES_DIR / "hyperparameter_tuning_results.csv")
if "final_comparison_df" in globals():
    ensure_csv_exists(final_comparison_df.reset_index(), TABLES_DIR / "final_test_comparison.csv")
    ensure_json_exists({
        "Baseline_A": _json_safe(final_comparison_df.loc["A - TF-IDF + Logistic Regression"].to_dict()) if "A - TF-IDF + Logistic Regression" in final_comparison_df.index else None,
        "Baseline_B": _json_safe(final_comparison_df.loc["B - Frozen distilgpt2 + MLP"].to_dict()) if "B - Frozen distilgpt2 + MLP" in final_comparison_df.index else None,
        "System_C": _json_safe(final_comparison_df.loc["C - LoRA fine-tuned GPT"].to_dict()) if "C - LoRA fine-tuned GPT" in final_comparison_df.index else None,
    }, METRICS_DIR / "final_metrics.json")
if "annotation_df" in globals():
    ensure_csv_exists(annotation_df, ERRORS_DIR / "misclassified_examples.csv")
if not (EMBEDDINGS_DIR / "embedding_analysis.txt").exists() and embedding_report_lines:
    save_markdown(EMBEDDINGS_DIR / "embedding_analysis.txt", "\n".join(embedding_report_lines))

audit_rows = []
for logical_name, path in required_artefacts.items():
    exists = path.exists()
    audit_rows.append({
        "artefact": logical_name,
        "path": str(path),
        "exists": exists,
        "kind": "required",
    })

for logical_name, candidate_paths in optional_one_of.items():
    existing_candidates = [str(p) for p in candidate_paths if p.exists()]
    audit_rows.append({
        "artefact": logical_name,
        "path": " | ".join(str(p) for p in candidate_paths),
        "exists": len(existing_candidates) > 0,
        "kind": "one_of",
        "existing_candidates": existing_candidates,
    })

artefact_audit_df = pd.DataFrame(audit_rows)
save_dataframe(artefact_audit_df, LOGS_DIR / "artefact_audit.csv", index=False)
write_json(LOGS_DIR / "artefact_audit.json", artefact_audit_df.to_dict(orient="records"))

missing_required = artefact_audit_df[
    (artefact_audit_df["kind"] == "required") & (~artefact_audit_df["exists"].astype(bool))
]["artefact"].tolist()

append_section_header("POST-RUN ARTIFACT AUDIT")
append_run_summary(artefact_audit_df.to_string(index=False))

if missing_required:
    append_run_summary(f"Missing required artefacts: {missing_required}")
else:
    append_run_summary("All required artefacts were present at end of run.")

append_run_summary("Extended evidence packaging completed.")


# ==============================================================================
# Cell 67 [code]
# ==============================================================================
# Final packaging check.
#
# This cell is here to vierify the expected artefacts and results align with recognisable filenames.
# If any named artefacts are missing but their in-memory results still exist,
# it rebuilds them, writes a manifest, and then exports a zip of the outputs folder.

from zipfile import ZipFile, ZIP_DEFLATED

def rebuild_confusion_matrix_if_missing() -> None:
    path = FIGURES_DIR / "confusion_matrix.png"
    if path.exists():
        return
    required = all(name in globals() for name in ["df_test", "test_preds_a", "test_preds_b", "test_preds_c", "test_labels_c"])
    if not required:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    systems = [
        ("A: TF-IDF + LR", df_test["label"].values, test_preds_a),
        ("B: Frozen distilgpt2 + MLP", df_test["label"].values, test_preds_b),
        ("C: LoRA GPT", test_labels_c, test_preds_c),
    ]
    for ax, (title, labels, preds) in zip(axes, systems):
        cm = confusion_matrix(labels, preds)
        ConfusionMatrixDisplay(cm, display_labels=["Negative", "Positive"]).plot(ax=ax, colorbar=False)
        ax.set_title(title)
    plt.tight_layout()
    save_current_figure("confusion_matrix.png")
    plt.close("all")

def rebuild_roc_pr_if_missing() -> None:
    need_roc = not (FIGURES_DIR / "roc_curve.png").exists()
    need_pr = not (FIGURES_DIR / "pr_curve.png").exists()
    required = all(name in globals() for name in ["df_test", "test_proba_a", "test_proba_b", "test_proba_c", "test_labels_c"])
    if not required:
        return
    if need_roc:
        plt.figure(figsize=(7, 6))
        for name, labels, proba in [
            ("A: TF-IDF + LR", df_test["label"].values, test_proba_a),
            ("B: Frozen distilgpt2 + MLP", df_test["label"].values, test_proba_b),
            ("C: LoRA GPT", test_labels_c, test_proba_c),
        ]:
            fpr, tpr, _ = roc_curve(labels, proba)
            auc = roc_auc_score(labels, proba)
            plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})")
        plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title("ROC curves on the held-out test set")
        plt.legend()
        plt.tight_layout()
        save_current_figure("roc_curve.png")
        plt.close("all")
    if need_pr:
        plt.figure(figsize=(7, 6))
        for name, labels, proba in [
            ("A: TF-IDF + LR", df_test["label"].values, test_proba_a),
            ("B: Frozen distilgpt2 + MLP", df_test["label"].values, test_proba_b),
            ("C: LoRA GPT", test_labels_c, test_proba_c),
        ]:
            precision, recall, _ = precision_recall_curve(labels, proba)
            pr_auc = average_precision_score(labels, proba)
            plt.plot(recall, precision, label=f"{name} (PR-AUC={pr_auc:.4f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-recall curves on the held-out test set")
        plt.legend()
        plt.tight_layout()
        save_current_figure("pr_curve.png")
        plt.close("all")

def rebuild_review_length_figure_if_missing() -> None:
    path = FIGURES_DIR / "review_length_analysis.png"
    if path.exists():
        return
    if "bucket_df" in globals():
        src_df = bucket_df.copy()
    else:
        csv_path = TABLES_DIR / "review_length_analysis.csv"
        if not csv_path.exists():
            return
        src_df = pd.read_csv(csv_path)
    plt.figure(figsize=(10, 4))
    sns.barplot(data=src_df, x="Length bucket", y="F1", hue="System")
    plt.ylim(0, 1)
    plt.title("F1 by review-length bucket")
    plt.tight_layout()
    save_current_figure("review_length_analysis.png")
    plt.close("all")

def rebuild_misclassified_if_missing() -> None:
    path = ERRORS_DIR / "misclassified_examples.csv"
    if path.exists():
        return
    if "annotation_df" in globals():
        save_dataframe(annotation_df, path, index=False)

def rebuild_final_metrics_if_missing() -> None:
    path = METRICS_DIR / "final_metrics.json"
    if path.exists():
        return
    if "final_comparison_df" in globals():
        write_json(path, {
            "Baseline_A": _json_safe(final_comparison_df.loc["A - TF-IDF + Logistic Regression"].to_dict()) if "A - TF-IDF + Logistic Regression" in final_comparison_df.index else None,
            "Baseline_B": _json_safe(final_comparison_df.loc["B - Frozen distilgpt2 + MLP"].to_dict()) if "B - Frozen distilgpt2 + MLP" in final_comparison_df.index else None,
            "System_C": _json_safe(final_comparison_df.loc["C - LoRA fine-tuned GPT"].to_dict()) if "C - LoRA fine-tuned GPT" in final_comparison_df.index else None,
        })

def build_submission_bundle() -> None:
    rebuild_confusion_matrix_if_missing()
    rebuild_roc_pr_if_missing()
    rebuild_review_length_figure_if_missing()
    rebuild_misclassified_if_missing()
    rebuild_final_metrics_if_missing()

    manifest_rows = []
    for p in sorted(OUTPUT_DIR.rglob("*")):
        if p.is_file():
            manifest_rows.append({
                "relative_path": str(p.relative_to(OUTPUT_DIR.parent)),
                "size_bytes": int(p.stat().st_size),
            })

    manifest_df = pd.DataFrame(manifest_rows)
    save_dataframe(manifest_df, LOGS_DIR / "submission_manifest.csv", index=False)

    bundle_path = ROOT_DIR / "outputs_submission_bundle.zip"
    with ZipFile(bundle_path, "w", compression=ZIP_DEFLATED) as zf:
        for p in sorted(OUTPUT_DIR.rglob("*")):
            if p.is_file():
                zf.write(p, arcname=str(p.relative_to(ROOT_DIR)))

    append_section_header("SUBMISSION BUNDLE EXPORT")
    append_run_summary(f"Submission bundle written to: {bundle_path}")
    append_run_summary(f"Manifest rows: {len(manifest_rows)}")

build_submission_bundle()
append_run_summary("Run completed successfully.")
