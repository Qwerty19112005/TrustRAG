# TrustRAG datasets

Downloaded and verified by `scripts/download_data.py`.
Total on disk: 57.5 MB. Re-run that script anywhere (including Colab) to rebuild this folder.

```
data/
  ragtruth/train.parquet                    22.3 MB   15,090 rows
  ragtruth/test.parquet                      3.9 MB    2,700 rows
  halueval/qa.parquet                        3.8 MB   10,000 rows
  hotpotqa/distractor_validation.parquet    27.5 MB    7,405 rows
```

Read any of them with pyarrow (installed locally, version 25.0.1):

```python
import pyarrow.parquet as pq
t = pq.ParquetFile("data/ragtruth/train.parquet").read().to_pydict()
```

Or with the datasets library:

```python
from datasets import load_dataset
ds = load_dataset("parquet", data_files={"train": "data/ragtruth/train.parquet",
                                         "test":  "data/ragtruth/test.parquet"})
```

---

## 1. RAGTruth (`wandb/RAGTruth-processed`)

Primary dataset. Trains and evaluates the detector.

**Columns:** `id`, `query`, `context`, `output`, `task_type`, `quality`, `model`,
`temperature`, `hallucination_labels`, `hallucination_labels_processed`, `input_str`

**Read this before Day 1.** `hallucination_labels` is a **JSON-encoded string**, not a
struct. It needs `json.loads()` before use. An empty response means no hallucination.

```python
import json
spans = json.loads(row["hallucination_labels"]) if row["hallucination_labels"] else []
# each span: {"start": int, "end": int, "text": str, "meta": str,
#             "label_type": str, "implicit_true": bool, "due_to_null": bool}
```

`start` and `end` are character offsets into `output`. This is exactly what the
span-to-claim projection consumes. A claim is labelled unsupported if ANY of it
falls inside a span; see `trustrag/data.py`.

The separate `hallucination_labels_processed` column is only
`struct<evident_conflict: int32, baseless_info: int32>`, a two-way count. The full
four-way type split lives in the raw JSON, so parse that instead.

**Verified counts on the train split:**

| Property | Value |
|---|---|
| Responses with at least one hallucination span | 6,721 of 15,090 (**44.5%**) |
| Total annotated spans | 12,756 |

| Label type | Spans |
|---|---|
| Evident Baseless Info | 5,499 |
| Evident Conflict | 4,705 |
| Subtle Baseless Info | 2,367 |
| Subtle Conflict | 185 |

| Task type | Rows |
|---|---|
| Data2txt | 5,298 |
| QA | 5,034 |
| Summary | 4,758 |

Models, 2,515 responses each: `gpt-4-0613`, `gpt-3.5-turbo-0613`,
`mistral-7B-instruct`, `llama-2-7b-chat`, `llama-2-13b-chat`, `llama-2-70b-chat`.

**Two planning notes.**

1. Filtering to `task_type == "QA"` leaves **5,034 training rows**, not 15,090. That is
   still plenty, but size the run accordingly.
2. **Subtle Conflict has only 185 spans**, about 1.4% of the total. Per-type results for
   that category will be noisy, so report a confidence interval or pool it with Subtle
   Baseless Info rather than quoting a bare number.

## 2. HaluEval (`pminervini/HaluEval`, config `qa`)

Transfer test. Never trained on.

**Columns:** `knowledge`, `question`, `right_answer`, `hallucinated_answer`

Each row yields two evaluation examples, one supported and one not, so the task is
perfectly balanced and a random detector scores exactly 0.50 AUROC. 10,000 rows gives
20,000 evaluation examples.

Six other configs exist upstream and were not downloaded: `dialogue`,
`dialogue_samples`, `general`, `qa_samples`, `summarization`,
`summarization_samples`. Add them to `WANT` in `download_data.py` if needed.

## 3. HotpotQA (`hotpotqa/hotpot_qa`, config `distractor`, split `validation`)

Live demo corpus only.

**Columns:** `id`, `question`, `answer`, `type`, `level`, `supporting_facts`, `context`

`context` is `struct<title: list<string>, sentences: list<list<string>>>`, 10 paragraphs
per question, 2 gold and 8 distractors. `supporting_facts` gives the gold titles and
sentence indices, which is free grounding ground truth.

All 7,405 rows are `level == "hard"`; `type` is 5,918 bridge and 1,487 comparison.
Across the split there are **66,581 unique paragraph titles**, far more than the ~500
documents the demo needs to index.

---

## Deliberately not downloaded

**HotpotQA train (332 MB, two shards).** The project only indexes about 500 documents
for the demo. The validation split above already carries 66,581 unique paragraphs, so
the train split adds size and no capability.

**FEVER.** The proposal uses it indirectly. Our NLI checkpoint
(`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`) was already trained on
FEVER among five NLI corpora, so we consume it through the model weights rather than as
data. Nothing to download.

## Models still to fetch

These are pulled at runtime by `transformers`, not stored here:

- `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` (0.4B, the verifier)
- `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` (0.18B, the size ablation)
- `BAAI/bge-small-en-v1.5` (33M, the embedding feature and demo retriever)
- `spacy/en_core_web_sm` (sentence splitting and entity counts)
