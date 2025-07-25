# BPE Tokenizer Implementations

Here, the implementation of two versions of a BPE (Byte-Pair Encoding) tokenizer is described.

1. **Simple Version**
   
   A straightforward version that directly applies the BPE algorithm to the corpus.  
   **Disadvantage:**  
   Tokens like `"dog"`, `"dog?"`, and `"dog."` are treated as completely different tokens.

2. **Regex-based Version**
   
   An improved version that splits the corpus based on regex patterns, effectively separating punctuation and special characters.  
   This version also provides the option to register special tokens, such as `<|endoftext|>`.

---

## Methods Overview

Each implementation above contains three main methods:

- **`train`**  
  Used for training the tokenizer (note: this is different from LLM training, and is part of pre-processing that happens even before LLMs are trained).

- **`encode`**  
  Encodes input text to token IDs.

- **`decode`**  
  Decodes token IDs back to text.

> The encoding method operates on bytes, which are converted from Unicode code points using UTF-8 encoding.  
> In Python, text is read as Unicode code points and then converted to bytes via UTF-8 encoding, after which it is encoded into token IDs.

---

## Supporting Functions

There are two major supporting functions used in the implementation:

- **`merge`**  
  Merges the most frequent pair (only one pair at a time).

- **`get_stats`**  
  Calculates the frequency of each pair of bytes.