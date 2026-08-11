"""
Tiny word embeddings — the full pipeline in ~100 lines, no PyTorch.

    text
     ↓
    tokenizer        "cat" → 1        (a lookup table, never changes)
     ↓
    token ID
     ↓
    embedding matrix  E[1]            (learnable vector at that address)
     ↓
    prediction        softmax(E[1] @ W + b)
     ↓
    loss              cross-entropy vs. correct next token
     ↓
    backpropagation   gradients
     ↓
    update            E, W, b change
"""

import numpy as np

# --------------------------------
# 1. TOKENIZER
# --------------------------------
# Mapping word -> ID. This is just a dictionary; nothing is learned here.

vocab = {
    "<pad>": 0,
    "cat": 1,
    "dog": 2,
    "is": 3,
    "running": 4,
    "sleeping": 5,
}

id_to_token = {v: k for k, v in vocab.items()}

# Training examples: input -> expected next token
training_data = [
    ("cat", "is"),
    ("dog", "is"),
    ("is", "running"),
    ("is", "sleeping"),
]

# --------------------------------
# 2. MODEL PARAMETERS
# --------------------------------

vocab_size = len(vocab)  # 6
embedding_dim = 3

np.random.seed(42)

# The embedding matrix: one row per token, each row a learnable vector.
E = np.random.randn(vocab_size, embedding_dim) * 0.1

# Snapshot BEFORE training, so we can show the contrast later.
E_before = E.copy()

# Prediction matrix: maps an embedding vector to scores over all tokens.
W = np.random.randn(embedding_dim, vocab_size) * 0.1

# Bias
b = np.zeros(vocab_size)

# --------------------------------
# 3. SOFTMAX
# --------------------------------


def softmax(x):
    x = x - np.max(x)  # numerical stability
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


# --------------------------------
# 4. TRAIN
# --------------------------------

learning_rate = 0.1

for epoch in range(5000):
    total_loss = 0

    for input_word, target_word in training_data:
        input_id = vocab[input_word]
        target_id = vocab[target_word]

        # -------------------------
        # Embedding lookup
        # -------------------------

        embedding = E[input_id]

        # -------------------------
        # Prediction
        # -------------------------

        logits = embedding @ W + b

        probabilities = softmax(logits)

        # -------------------------
        # Cross entropy loss
        # -------------------------

        loss = -np.log(probabilities[target_id] + 1e-9)
        total_loss += loss

        # -------------------------
        # Backpropagation
        # -------------------------

        d_logits = probabilities.copy()
        d_logits[target_id] -= 1

        dW = np.outer(embedding, d_logits)
        db = d_logits

        d_embedding = d_logits @ W.T

        # -------------------------
        # Update parameters
        # -------------------------

        W -= learning_rate * dW
        b -= learning_rate * db
        E[input_id] -= learning_rate * d_embedding

    if epoch % 500 == 0:
        print("epoch:", epoch, "loss:", total_loss)

# --------------------------------
# 5. LOOK AT LEARNED EMBEDDINGS
# --------------------------------

print("\nLearned embeddings:")

for token, token_id in vocab.items():
    print(token, "→", E[token_id])

# --------------------------------
# 6. TEST
# --------------------------------


def predict(word):
    """Return the most likely next word after `word`."""
    token_id = vocab[word]

    embedding = E[token_id]

    logits = embedding @ W + b

    probabilities = softmax(logits)

    predicted_id = np.argmax(probabilities)

    return id_to_token[predicted_id]


print("\nPredictions:")

for word in ["cat", "dog", "is"]:
    print(word, "→", predict(word))

# --------------------------------
# 7. KEY INSIGHT
# --------------------------------

print("\n--- Key insight ---")
print("Tokenizer (never changes):")
print("    cat ->", vocab["cat"], "   dog ->", vocab["dog"], "   is ->", vocab["is"])
print("\nEmbedding row for 'cat':")
print("    BEFORE training:", E_before[vocab["cat"]])
print("    AFTER  training:", E[vocab["cat"]])
print("\nThe tokenizer still maps 'cat' -> 1. Only the vector at E[1] changed.")
