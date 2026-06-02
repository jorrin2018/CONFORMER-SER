"""Create a minimal local tokenizer for the lightweight experiment.

Since HuggingFace Hub is unreachable, we create a simple local tokenizer
that mimics DistilBERT's behavior (lowercase, WordPiece-style vocab).
Our SimpleTransformerEncoder only needs integer token IDs, so a basic
whitespace tokenizer with a small vocab is sufficient.
"""

import os
import json

out_dir = os.path.join(os.path.dirname(__file__), 'local_tokenizer')
os.makedirs(out_dir, exist_ok=True)

# Build a simple vocab: special tokens + lowercase letters + common words
vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
# lowercase letters and digits
for c in 'abcdefghijklmnopqrstuvwxyz0123456789':
    vocab.append(c)
# common punctuation
for c in '.,!?;:\'"-()[]{}':
    vocab.append(c)
# some common words
common_words = [
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
    'my', 'your', 'his', 'its', 'our', 'their', 'this', 'that', 'these', 'those',
    'and', 'or', 'but', 'if', 'then', 'else', 'when', 'where', 'why', 'how',
    'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such',
    'no', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
    'can', 'may', 'must', 'shall', 'need', 'dare', 'ought', 'used', 'to', 'of',
    'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'between', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'up', 'down', 'out', 'off', 'over'
]
vocab.extend(common_words)

# Fill up to a reasonable vocab size (not 30522, just enough for basic text)
# We pad to 1000 entries so the embedding matrix isn't too sparse
while len(vocab) < 1000:
    vocab.append(f'[EXTRA_{len(vocab)}]')

# Save vocab.txt (one token per line)
with open(os.path.join(out_dir, 'vocab.txt'), 'w', encoding='utf-8') as f:
    for token in vocab:
        f.write(token + '\n')

# Save tokenizer_config.json
config = {
    "do_lower_case": True,
    "model_max_length": 128,
    "pad_token": "[PAD]",
    "unk_token": "[UNK]",
    "cls_token": "[CLS]",
    "sep_token": "[SEP]",
    "mask_token": "[MASK]",
    "vocab_size": len(vocab)
}
with open(os.path.join(out_dir, 'tokenizer_config.json'), 'w', encoding='utf-8') as f:
    json.dump(config, f, indent=2)

print(f"Local tokenizer created at: {out_dir}")
print(f"Vocab size: {len(vocab)}")
