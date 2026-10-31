# TrustRAG

Course project for NLP with LLMs, by Adhithyan (230121003) and Aditya Raj (230121004).

RAG systems are supposed to answer from the documents they retrieve, but they still write sentences those documents don't support. TrustRAG checks an answer one sentence at a time. It runs a small NLI model (DeBERTa-v3, 0.4B parameters, not fine-tuned) against the passages, adds some simple word-matching features, and gives each sentence a probability of being unsupported. That probability decides whether the sentence gets shown, flagged or held back.


## Results

On the RAGTruth QA test set (5,790 sentences, 6.2% unsupported):

| Method | AUROC | AUPRC |
|---|---|---|
| Word overlap | 0.757 | 0.120 |
| NLI alone, best sentence | 0.725 | 0.119 |
| The 6 features from our proposal | 0.760 | 0.156 |
| All 12 features, logistic regression | 0.833 | 0.320 |
| All 12 features, boosted trees | 0.883 | 0.423 |

Other things we found:

- Word matching carries most of the signal on RAGTruth. Removing the six added features costs 0.073 AUROC, and removing the three NLI features costs 0.011.
- Calibration brings the error (ECE) down from 0.287 to 0.033. Flagging about a sixth of sentences takes the unsupported rate among the rest from 6.2% to 3.0%.
- On 1,008 sentences, the gpt-oss-120b judge scores 0.877 and our boosted trees score 0.873. That gap is within noise, and ours needs no API.
- On HaluEval, which we never trained on, the NLI score is what carries over (0.757 on rows where both answers have similar length).
- On news summaries, whole-passage NLI drops from 0.850 to 0.761 once an article gets cut off at 512 tokens. Our detector stays near 0.79.
- Contradictions are the weak spot (0.623), because they reuse the passage's words.
- For a small label check we had an LLM relabel 108 test sentences and used that as a baseline. RAGTruth agrees with it only moderately (kappa 0.46). It hasn't been checked against people, so treat this one as a rough signal.

## What's in the repo

```
trustrag/                 the code everything else uses
  data.py                 loads RAGTruth, turns annotated spans into sentence labels
  features.py             NLI and embedding scoring, the first six features
  lexical.py              the six added features, word overlap baseline
  detector.py             the classifier, and check() which scores an answer
  rag.py                  retrieval and the generator for the demo
  evaluate.py             metrics
scripts/
  setup/                  download_data.py, check_env.py
  experiments/            run_experiment.py, bench_verifiers.py, length_split.py
  llm_judges/             run_judges.py, judge_ci.py
  label_check/            make_label_sample.py, agreement.py
  demo/                   run_demo.py, demo_app.py
notebooks/
  TrustRAG_Results_and_Demo.ipynb    results and a live demo, outputs saved
  TrustRAG_Colab_Setup.ipynb         sets up the data and model on Colab if you have no GPU
results/                  our saved results, one folder per experiment
data/                     empty until you run download_data.py
```

## Setup

We ran everything on Windows 11 with Python 3.14, an RTX 3060 laptop GPU (6 GB) and 16 GB of RAM. You don't need a GPU to look at the results. The demo also runs on CPU, it's just slow.

```bash
git clone https://github.com/Qwerty19112005/TrustRAG.git
cd TrustRAG

# only with an NVIDIA GPU, otherwise pip gives you the CPU build of torch
pip install torch --index-url https://download.pytorch.org/whl/cu126

pip install -r requirements.txt
python -m spacy download en_core_web_sm

python scripts/setup/download_data.py    # about 58 MB
python scripts/setup/check_env.py        # should end with "All good."
```

The models download by themselves the first time they're needed, about 5 GB in total.

## See it working

If you only have a few minutes, start here.

**The notebook.** `notebooks/TrustRAG_Results_and_Demo.ipynb` has its outputs saved, so you can read it without running anything. To run it yourself:

```bash
cd notebooks
jupyter notebook TrustRAG_Results_and_Demo.ipynb
```

Part 1 reads the saved results and takes seconds. Part 2 loads the models and runs the checker live, which takes about a minute once they're downloaded.

**The demo page.**

```bash
python scripts/demo/demo_app.py
```

Wait for `Ready`, then open http://localhost:8000. The first start embeds all the HotpotQA paragraphs, about 12 minutes on our GPU. After that it starts in about 30 seconds. You can ask about the HotpotQA paragraphs or paste your own text, and the two buttons at the top fill in examples.

Keep in mind that the generator is small (Qwen2.5-1.5B) and gets things wrong, sometimes confidently. SHOW only means a sentence looks grounded in the passages, not that it's true. A false claim written in the passage's own words can get through. Also, a tiny change to a question, even an accent mark, can change the answer.

## Running the experiments

You don't have to, because the results are already in `results/`. If you want to check them, here's what makes what, with times from our laptop:

| Command | Writes to | Time |
|---|---|---|
| `python scripts/experiments/run_experiment.py --smoke` | `results/smoke/` | about 3 min, good first test |
| `python scripts/experiments/run_experiment.py` | `results/qa/` | about 105 min |
| `python scripts/experiments/run_experiment.py --verifier base` | `results/qa_base/` | about 35 min |
| `python scripts/experiments/run_experiment.py --task Summary` | `results/summary/` | about 3 hours |
| `python scripts/experiments/length_split.py` | `results/summary/by_length.json` | under a minute |
| `python scripts/experiments/bench_verifiers.py large` (then `base`) | timing inside `metrics.json` | a few minutes |
| `python scripts/llm_judges/judge_ci.py` | `results/judges/bootstrap.json` | about a minute |
| `python scripts/label_check/agreement.py` | `results/label_check/agreement.json` | about a minute |
| `python scripts/demo/run_demo.py` | `results/demo/demo.json` | about 20 min the first time |

Features get saved to `data/cache/` as they're computed, so if a run dies it picks up where it stopped, and a second run only takes a few minutes. Everything is seeded. When we reran all of this from the saved features the numbers came out identical, and recomputing one chunk of features from scratch gave the same values too. Timings will be different on your machine, and the demo's generated answers could change with a different version of transformers.

**LLM judges.** These need a free key from console.groq.com. Put it in an environment variable called `GROQ_API_KEY`, then run:

```bash
python scripts/llm_judges/run_judges.py
```

The free tier gives each model 200,000 tokens a day, so both judges took us a night. Our answers are saved in `results/judges/cache/` and the script skips anything already there, so you only need a key if you delete them. If Groq has retired a model by the time you read this, `--list-models` shows what's available, and the names are at the top of `run_judges.py`.

**Label check.** `results/label_check/llm_labels.json` holds the LLM's labels. They didn't come from a script here, so `agreement.py` just reads them. `make_label_sample.py` shows how the 108 sentences were picked.

## File names in the report

We tidied the code into folders after submitting the report and contribution statement, so they mention some old paths:

| Old path | Now |
|---|---|
| `scripts/run_mid.py` | `scripts/experiments/run_experiment.py` |
| `scripts/bench_verifiers.py` | `scripts/experiments/bench_verifiers.py` |
| `scripts/length_split.py` | `scripts/experiments/length_split.py` |
| `scripts/run_judges.py`, `scripts/judge_ci.py` | `scripts/llm_judges/` |
| `scripts/make_label_sample.py`, `scripts/agreement.py` | `scripts/label_check/` |
| `scripts/demo.py` | `scripts/demo/run_demo.py` |
| `scripts/download_data.py` | `scripts/setup/download_data.py` |

## Problems you might hit

- **Everything runs on CPU even though you have a GPU.** You probably got the CPU build of torch. Install it with the command under Setup. `check_env.py` tells you which GPU it sees.
- **"The paging file is too small" on Windows.** This happened to us while starting the demo with a lot of other apps open. It means Windows ran out of virtual memory while loading a model. Close some programs and try again.
- **CUDA out of memory.** The demo holds the generator and the verifier at once, about 5.2 GB. The experiments alone peak around 2.6 GB. Close anything else using the GPU.

## Data and models

| | Used for |
|---|---|
| [RAGTruth](https://aclanthology.org/2024.acl-long.585/) | training and testing |
| [HaluEval](https://aclanthology.org/2023.emnlp-main.397/) | testing on data we never trained on |
| [HotpotQA](https://aclanthology.org/D18-1259/) | the demo's document collection |
| `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` | the verifier (435M parameters) |
| `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` | the smaller verifier (184M) |
| `BAAI/bge-small-en-v1.5` | embeddings and retrieval |
| `Qwen/Qwen2.5-1.5B-Instruct` | the demo's generator |
| `openai/gpt-oss-120b`, `openai/gpt-oss-20b` | LLM judges, through Groq |

More on the data files is in `data/README.md`. The full reference list is at the end of the report.
