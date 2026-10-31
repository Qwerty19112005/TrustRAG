"""A small RAG pipeline for the demo: bge-small retrieval and a local Qwen generator."""

import gc

import numpy as np
import torch

from trustrag import features as F

GENERATOR = "Qwen/Qwen2.5-1.5B-Instruct"
TOP_K = 3
CHUNK = 600            # characters per chunk, about one HotpotQA paragraph
MAX_SOURCE = 50000     # characters of pasted text we read

# bge-small expects this prefix on queries only
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
PROMPT = ("Answer the following question in two to four sentences:\n{question}\n"
          "Bear in mind that your response should be strictly based on the following {k} passages:\n"
          "{passages}\n"
          "In case the passages do not contain the necessary information to answer the question, "
          "please reply with: \"Unable to answer based on given passages.\"\noutput:")


def retrieve(questions, V, k=TOP_K):
    Q = F.embed([QUERY_PREFIX + q for q in questions])
    return np.argsort(-(Q @ V.T), axis=1)[:, :k]


def prompt_for(question, passages):
    return PROMPT.format(question=question, k=len(passages),
                         passages="\n".join("passage %d: %s" % (i + 1, p) for i, p in enumerate(passages)))


def chunks_of(text):
    """Group whole sentences into chunks so a long document can be searched."""
    out, current = [], ""
    for sent in F.split_sentences(text, max_chars=MAX_SOURCE, min_len=1, cap=10 ** 6):
        if current and len(current) + len(sent) > CHUNK:
            out.append(current)
            current = sent
        else:
            current = (current + " " + sent).strip()
    return out + [current] if current else out


class Generator:
    def __init__(self, name=GENERATOR):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name)
        dtype = torch.float16 if F._device() == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(F._device()).eval()

    def __call__(self, prompt, max_new_tokens=200):
        text = self.tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                            add_generation_prompt=True)
        x = self.tok(text, return_tensors="pt").to(F._device())
        with torch.no_grad():
            out = self.model.generate(**x, max_new_tokens=max_new_tokens, do_sample=False,
                                      temperature=None, top_p=None, top_k=None)
        return self.tok.decode(out[0, x["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def close(self):
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
