# 01 — Tiny Word Embeddings (no PyTorch)

The smallest possible working example of the full learning pipeline:

```text
text
 ↓
tokenizer          "cat" → 1          (lookup table — NEVER changes)
 ↓
token ID
 ↓
embedding matrix   E[1]               (learnable vector at that address)
 ↓
prediction         softmax(E[1] @ W + b)
 ↓
loss               cross-entropy vs. correct next token
 ↓
backpropagation    gradients
 ↓
update             E, W, b change
```

## Run it

```bash
uv sync              # creates .venv and installs numpy (from pyproject.toml)
uv run tiny_embeddings.py
```

Requires [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`).

The lockfile (`uv.lock`) pins the exact numpy version, so every machine gets
the same environment.

## What to look at

- **Tokenizer** — a plain dictionary. `"cat" → 1` never changes during training.
- **Embedding matrix** `E` — one row per token. `E[1]` is the vector for `cat`.
  Backpropagation updates only the row of the input token (`E[input_id] -= lr * d_embedding`).
- **Prediction matrix** `W` — maps an embedding to scores over all 6 tokens.
- **Loss curve** — starts ~7.16, drops to ~1.46. A perfect model would reach
  ~1.386 (ln 4): the model must distribute probability mass between
  `running` and `sleeping` after `is`.

## Key insight

The tokenizer creates the **address**. The embedding matrix stores the
**learnable vector at that address**. Training changes the vector, never the
address.

| | before | after |
| --- | --- | --- |
| `"cat" → 1` | `E[1] = random` | `E[1] = learned numbers` |
| `"dog" → 2` | `E[2] = random` | `E[2] = learned numbers` |

## Files

- `pyproject.toml` — project metadata + dependencies (the uv source of truth)
- `uv.lock` — pinned dependency versions (commit this)
- `tiny_embeddings.py` — the model
- `.venv/` — local environment, created by uv, git-ignored

## Limitation vs. a real LLM

This toy model predicts from **one token only**. A real Transformer uses
attention across many tokens, so the representation of `cat` changes depending
on its context (e.g. "the cat **is** running" vs. "the cat **was** sleeping").
