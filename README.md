# Parameter-Efficient Fine-Tuning of Decoder-Only GPT Models for Long-Form Sentiment Classification on IMDb

Does LoRA fine-tuning of a small decoder-only GPT beat simpler baselines on
long-form sentiment classification? In this study, no. LoRA fine-tuning lifted
`distilgpt2` from an F1 of 0.832 (frozen features with an MLP) to 0.903, but a
TF-IDF and logistic regression baseline matched it.

This was an individual assignment (CA1) for the MSc module Artificial
Intelligence for Computer Vision & NLP at Atlantic Technological University
(ATU) Donegal, April 2026. `gpt_sentiment.py` runs the whole study end to end,
and `outputs/` holds the recorded results of the final run. The run used one
NVIDIA RTX 3070 Ti (8 GB) with Python 3.11.9, PyTorch 2.5.1 (CUDA 12.1),
Transformers 5.5.4 and PEFT 0.19.1.

## Results

Held-out IMDb test set: 25,000 reviews, balanced 12,500 / 12,500.

| System | Accuracy | Balanced acc. | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A: TF-IDF + logistic regression | 0.9026 | 0.9026 | 0.9021 | 0.9032 | 0.9027 | 0.9654 | 0.9636 | 0.0760 |
| B: frozen `distilgpt2` + MLP | 0.8280 | 0.8280 | 0.8129 | 0.8522 | 0.8321 | 0.9115 | 0.9114 | 0.1205 |
| C: LoRA fine-tuned `distilgpt2` | 0.9026 | 0.9026 | 0.8964 | 0.9105 | 0.9034 | 0.9660 | 0.9650 | 0.0759 |

Bootstrap 95% confidence intervals (500 resamples of the test set):

| System | F1 | ROC-AUC |
| --- | --- | --- |
| A | 0.8989–0.9067 | 0.9636–0.9674 |
| B | 0.8270–0.8371 | 0.9079–0.9151 |
| C | 0.8997–0.9077 | 0.9644–0.9679 |

- **Fine-tuning versus frozen features:** LoRA fine-tuning adds a large gain over
  frozen features from the same model (F1 0.8321 to 0.9034, ROC-AUC 0.9115 to
  0.9660).
- **Fine-tuning versus TF-IDF:** the two are indistinguishable. Their confidence
  intervals overlap on F1 and ROC-AUC, and their Brier scores are almost equal.
- **Decision thresholds:** System C's threshold (0.35) was chosen on the
  validation split; the baselines use 0.5. ROC-AUC and PR-AUC do not depend on
  a threshold and show the same tie.
- **Run-to-run variation:** seeds are fixed, but GPU kernels are not
  deterministic. An earlier run with the identical configuration, also logged
  in `outputs/logs/run_summary.txt`, gave System C an accuracy of 0.8990 and an
  F1 of 0.9021. The identical 90.26% accuracies in the final run fall within
  that variation.

Error analysis: System C made 1,315 false positives and 1,119 false negatives.
Misclassified reviews were longer on average (260.9 words against 225.0 for
correct predictions). The 50 most confident misclassifications are exported to
`outputs/errors/misclassified_examples.csv` for review against expected failure
modes: mixed sentiment, late reversals, sarcasm and truncation.

## Model selection and tuning

**Pilot.** Short LoRA runs (one epoch, 2,000 training and 500 validation
reviews) compared the two candidates:

| Model | Trainable params | Train time | Validation F1 | Validation ROC-AUC |
| --- | --- | --- | --- | --- |
| `distilgpt2` | 407,040 | 38 s | 0.852 | 0.917 |
| `gpt2` | 812,544 | 71 s | 0.832 | 0.904 |

`distilgpt2` scored higher in half the time, so it became the primary model.

**Staged coordinate search.** On 4,000 training and 1,000 validation reviews,
one setting was tuned at a time, keeping the best value before moving on:

| Stage | Setting | Values tried | Selected |
| --- | --- | --- | --- |
| Design | maximum sequence length | 128, 256 | 128 |
| Design | pooling | last token, mean | last token |
| Design | truncation | keep start, keep start and end | keep start and end |
| Optimisation | learning rate | 1e-4, 3e-4, 5e-4 | 5e-4 |
| Optimisation | weight decay | 0.0, 0.01 | 0.01 |
| Optimisation | LoRA rank | 4, 8, 16 | 4 |
| Optimisation | LoRA dropout | 0.0, 0.05 | 0.05 |
| Confirmation | epochs | 2, 3 | 3 |

The final model was then trained on the full 22,500-review training split and
validated on the 2,500-review validation split, with LoRA alpha 16, batch size
8, gradient accumulation 2 and warm-up ratio 0.06. Adapters sit on the GPT-2
`c_attn` and `c_proj` modules, and the classification head is trained in full.
The selected learning rate is the largest value searched, so a higher one was
not tested. The complete configuration is in `outputs/metrics/final_config.json`
and every tuning run is in `outputs/tables/hyperparameter_tuning_results.csv`.

## Data

The script uses the Large Movie Review Dataset (Maas et al., 2011), from the
local archive `aclImdb_v1.tar.gz` if present, otherwise through the Hugging
Face `imdb` dataset. The original train and test splits stay separate. A
stratified 10% of the training split (seed 42) becomes the validation split,
giving 22,500 training, 2,500 validation and 25,000 test reviews.

Training reviews have a median length of 174 words and a 95th percentile of
596. In GPT-2 BPE tokens the median is 216 and the 95th percentile 770, and
87.2% of reviews fit in 512 tokens, so truncation is a real design choice. The
`head_tail` strategy keeps both the opening and the end of a review, where the
overall judgement often appears.

Dataset source: https://ai.stanford.edu/~amaas/data/sentiment/

> Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng and
> Christopher Potts (2011). Learning Word Vectors for Sentiment Analysis.
> *Proceedings of the 49th Annual Meeting of the Association for Computational
> Linguistics*, pp. 142–150.

## Running the study

Install a CUDA build of PyTorch, then the remaining requirements:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

Run from the repository root:

```bash
python gpt_sentiment.py
```

The script recreates `outputs/` and `artefacts/` and finishes by writing
`outputs_submission_bundle.zip`. Set `DEBUG_MODE = True` near the top of the
script for a quick check on 4,000 training, 1,000 validation and 2,000 test
reviews. The script started as a notebook, so it is organised in numbered
cells and prints tables as it goes.

## Recorded outputs

| Path | Contents |
| --- | --- |
| `outputs/tables/final_test_comparison.csv` | Test metrics for all three systems. |
| `outputs/metrics/` | Final configuration, test metrics, bootstrap intervals, Brier scores, threshold search. |
| `outputs/tables/` | Class balance, review lengths, pilot comparison, tuning runs, training history, results by review length. |
| `outputs/figures/` | Exploratory, tuning and evaluation figures. The numbered `figure_NNN.png` files are automatic copies of the named figures. |
| `outputs/embeddings/` | Token-length analysis, embedding nearest neighbours and similarity, contextual similarity before and after fine-tuning. |
| `outputs/errors/` | Error counts and the 50 most confident misclassifications. |
| `outputs/logs/` | Run summary, classification reports, written rationales, artefact audit and submission manifest. |
| `outputs/models/final_model/` | The trained LoRA adapter, its config and the tokenizer. |

The data splits (`outputs/splits/`) are not included because they copy the
IMDb text; the script regenerates them from the dataset with seed 42.

## Limitations

- Each system was trained once, so run-to-run variation is described by one
  earlier run rather than measured over several seeds.
- Tuning used subsets and a one-setting-at-a-time search, which is systematic
  but not exhaustive.
- Only LoRA was tested among parameter-efficient methods, and only two small
  GPT-2 models among decoder-only models.
- Encoder models are a more natural fit for classification and were not
  compared.
- Reviews longer than 128 tokens are truncated, even with the start-and-end
  strategy.
