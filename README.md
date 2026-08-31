# TrustRAG

**Cheap claim-level hallucination detection for Retrieval-Augmented Generation.**

Retrieval-Augmented Generation is supposed to stop a language model inventing facts:
give it the source documents and ask it to answer from them. It does not fully work.
Human annotators find unsupported text in a large share of RAG answers even when the
correct evidence was supplied.

The strongest detectors available today are themselves large generative models used as
judges, which costs seconds of latency and real money per answer. This project asks
whether that expense is necessary.

TrustRAG splits an answer into sentence-level claims, scores every claim against every
evidence sentence with a **0.4B-parameter natural language inference encoder**, fuses
six cheap signals into a calibrated probability, and turns that probability into an
action: show the claim, flag it, or hold it back.

Nothing is fine-tuned. The verifier is used off the shelf.

---

## Authors

| | |
|---|---|
| Adhithyan V |
| Aditya Raj |

Course project for **NLP with LLMs**. B.Tech, group of 2.

---

## The idea in one paragraph

Natural language inference models are trained on exactly one question: *does this
hypothesis follow from this premise?* That is the same question as *is this claim
supported by this evidence?* So an NLI model needs no task-specific training to do
this job. Our checkpoint was trained on five NLI corpora including FEVER, which is
literally claims paired with retrieved Wikipedia evidence, and that is why it
transfers.

---

## Repository layout

```
trustrag/                   the package
  data.py                   RAGTruth loading, span-to-claim labelling, leak-free splits
  features.py               the six features, NLI and embedding scoring
scripts/
  download_data.py          fetches all four datasets from Hugging Face
  check_env.py              verifies packages, GPU, data, and the NLI model
notebooks/
  TrustRAG_Colab_Setup.ipynb   Colab fallback runtime
docs/
  proposal/                 the submitted proposal and slides
  build/                    scripts that generate those documents
data/                       datasets (not tracked, see data/README.md)
results/                    experiment output
```

---

## Setup

Requires Python 3.9 or newer. A CUDA GPU is strongly recommended but not required.

```bash
git clone <repo-url>
cd trustrag

# on a machine with an NVIDIA GPU, install the CUDA build of torch FIRST,
# otherwise pip gives you the CPU-only wheel
pip install torch --index-url https://download.pytorch.org/whl/cu126

pip install -r requirements.txt
python -m spacy download en_core_web_sm

python scripts/download_data.py     # ~58 MB, about a minute
python scripts/check_env.py         # add --models to smoke-test the verifier
```

`check_env.py` prints a green/warn line for every dependency, the GPU it found, and
the row count of every data file. It should end with `Core stack is ready.`

---

## Quickstart

```python
from trustrag.data import load_claims, make_splits, describe

df = load_claims(task_type="QA")     # one row per claim
train, val = make_splits(df)         # grouped by source passage, stratified
describe(train, "train")
```

The first call runs spaCy over every response and takes a few minutes, then caches to
parquet. Later calls return in under a second.

```python
from trustrag.features import cached_extract, FEATURE_NAMES

X, y = cached_extract(train.sample(2000), tag="demo")   # X is [n, 6]
```

---

## Datasets

| Dataset | Size | Role |
|---|---|---|
| **RAGTruth** | 15,090 train / 2,700 test | Primary. Character-level hallucination spans with four type labels |
| **HaluEval** | 10,000 QA rows | Transfer test. Never trained on. Perfectly balanced |
| **HotpotQA** | 7,405 validation questions | Live demo corpus, 2 gold + 8 distractor paragraphs each |
| **FEVER** | — | Used indirectly, through the NLI checkpoint's training mix |

All are public and pulled from Hugging Face by `scripts/download_data.py`.
See [`data/README.md`](data/README.md) for schemas and a few loading traps.

---

## Where things stand

Measured on our own data, not quoted from papers.

**Claim-level dataset built.** 110,640 claims from the RAGTruth training split,
11.1% of them unsupported, across 2,514 source passages. Splits are grouped by source
passage and verified to share zero passages, so evidence cannot leak between them.

**Labelling rule corrected.** A claim counts as unsupported if *any* of it falls
inside an annotated span. A stricter 50% overlap rule was tested first and discarded:
the median flagged span is 34 characters against a median sentence of 104, so a 50%
threshold silently discards more than half of all genuine hallucinations.

**Evidence pooling validated.** Scoring a claim against each evidence sentence and
keeping the best beats scoring against the whole passage as one premise:

| Scoring method | AUROC | AUPRC |
|---|---|---|
| Whole passage as one premise | 0.626 | 0.570 |
| Max over evidence sentences | **0.705** | **0.635** |

**Known weakness, confirmed.** Running the verifier on real examples of each RAGTruth
error type: both *baseless information* categories are caught cleanly, with
`P(entailment)` at 0.000 and 0.001. *Subtle conflict* is missed entirely, scoring
`P(entailment) = 0.987` on a near-paraphrase that weakens the source's meaning. It is
also the rarest category, 185 spans out of 12,756.

Raw NLI at 0.705 AUROC is the zero-training baseline the feature fusion and
calibration have to improve on.

---

## Method

1. **Split.** Answer and evidence both into sentences with spaCy.
2. **Check.** One NLI pass per (claim, evidence sentence) pair, giving
   P(entailment), P(neutral), P(contradiction).
3. **Score.** Per claim, take the best evidence sentence:
   `support = max over evidence of [P(entail) − P(contradict)]`.
4. **Combine.** Six features into a logistic regression, then isotonic calibration.
5. **Decide.** Two thresholds map the probability to emit / flag / withhold.

The six features: support score, peak contradiction, mean top-3 entailment, best
embedding cosine, claim specificity (entities and numerals), claim length.

Logistic regression is deliberate rather than a neural head. With six features the
weights are readable, so we can say which signal carries the detection.

## Models

| Component | Model | Params |
|---|---|---|
| Verifier | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` | 435M |
| Verifier (ablation) | `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` | 184M |
| Embeddings | `BAAI/bge-small-en-v1.5` | 33M |
| Sentence splitting | spaCy `en_core_web_sm` | — |

Everything in the deployed path is open-weight and encoder-scale. A 70B model appears
only as a comparison baseline, never as a component.

---

## Related work

The core claim, that a small encoder can approach a large judge, is established. We
build on it rather than claiming it.

- **MiniCheck** (Tang et al., EMNLP 2024) — a 770M model reaching GPT-4 accuracy at
  400x lower cost.
- **Luna** (Belyi et al., 2024) — DeBERTa-large token-level RAG hallucination
  detection, 97% cost and 91% latency reduction against GPT-3.5.
- **LettuceDetect** (Kovács et al., 2025) — ModernBERT trained on RAGTruth, 79.22%
  example-level F1.

All three **fine-tune** on RAGTruth. We do not train anything, which is the difference
this project is testing. All three also stop at a detection score; none produce a
calibrated probability or act on it, which is where the calibration and selective
prediction work sits.

## References

1. Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS 33.
2. Niu et al. (2024). RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models. ACL 2024.
3. Li et al. (2023). HaluEval: A Large-Scale Hallucination Evaluation Benchmark for LLMs. EMNLP 2023.
4. Laban et al. (2022). SummaC: Re-Visiting NLI-based Models for Inconsistency Detection. TACL 10.
5. Zha et al. (2023). AlignScore: Evaluating Factual Consistency with a Unified Alignment Function. ACL 2023.
6. Geifman and El-Yaniv (2017). Selective Classification for Deep Neural Networks. NeurIPS 30.
7. He et al. (2023). DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training. ICLR 2023.
8. Thorne et al. (2018). FEVER: a Large-scale Dataset for Fact Extraction and VERification. NAACL-HLT 2018.

The full list of 16 is in [`docs/proposal/TrustRAG_Proposal.pdf`](docs/proposal/TrustRAG_Proposal.pdf).
