"""Tokenize text data"""

import regex as re

def create_vocab(text):
    split_text = re.split(r'([,.:;"!()\']|--|[\s])', text)
    split_text = [t for t in split_text if t.split()]
    split_text.extend(["<unk>", "<endoftext>"])
    unique_words = sorted(set(split_text))
    vocab = {t:i for i, t in enumerate(unique_words)}
    return vocab

class SimpleTokenizer:
    """Requires vocabulary"""
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:t for t,i in vocab.items()}

    def encode(self, text):
        split_text = re.split(r'([,.:;"!()\']|--|[\s])', text)
        split_text = [t for t in split_text if t.split()]
        unk_token_id = self.str_to_int['<unk>']
        token_id = [self.str_to_int.get(t, unk_token_id) for t in split_text]
        return token_id

    def decode(self, ids):
        converted_text = [self.int_to_str[id] for id in ids]
        joined_text = " ".join(converted_text)
        final_text = re.sub(r'\s+([,.:;"!()\'])', r'\1', joined_text)
        return final_text

if __name__ == "__main__":
    with open("test_text.txt", "r") as f:
        corpus = f.read()
    vocab = create_vocab(corpus)
    tokenizer = SimpleTokenizer(vocab)
    text = "Hello, world! This is the process of tokenization."
    encoded = tokenizer.encode(text)
    print(f"Encoded: {encoded}")
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded}")