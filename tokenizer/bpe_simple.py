import pickle

def get_stats(ids, counts=None):
    """
    Counts occurrences of consecutive token id pairs in a list.
    """
    counts = {} if counts is None else counts

    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    """
    Merges a given pair in ids list and replace with new integer token idx.
    """
    new_ids = []
    i = 0
    n = len(ids)
    while i < n-1:
        if pair[0] == ids[i] and pair[1] == ids[i+1]:
            new_ids.append(idx)
            i += 2   #Escape the pair in ids
        else:
            new_ids.append(ids[i])
            i += 1

    return new_ids

class BpeSimple:
    """Simpler Version of BPE implementation"""
    def __init__(self):
        self.vocab = {}  #used during decoding
        self.merges = {}  #used during encoding

    def encode(self, text):
        """
        Converts the input text into a list of bytes and finds the most occurring byte pairs.

        Then, performs a lookup on self.merges to merge suitable candidate pairs.
        """
        ids = list(text.encode('utf-8'))

        while True:
            stats = get_stats(ids)
            
            mergable_pairs = [pair for pair in stats if pair in self.merges]

            if not mergable_pairs:
                break

            best_pair = min(mergable_pairs, key= lambda p: self.merges[p])  #Find the pair with lowest index. We merge only one pair at a time.
            new_token = self.merges[best_pair]
            ids = merge(ids, best_pair, new_token)
        return ids
    
    def decode(self, ids):
        """
        Converts a list of token ids into a string using the vocab.
        """
        token_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = token_bytes.decode('utf-8', errors='replace')
        return text
    
    def train(self, vocab_size, corpus, verbose=False):
        """
        Trains the BPE tokenizer on the given corpus. 
        
        Uses stats function to get most occuring pair and merges it till vocab size is reached.
        """
        if vocab_size < 256:
            raise ValueError("Vocab size must be greater than 256")
        
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        num_merges = vocab_size - 256
        ids = list(corpus.encode('utf-8'))

        for i in range(num_merges):
            stats = get_stats(ids)
            pair = max(stats, key=stats.get)
            idx = 256+i
            ids = merge(ids, pair, idx)
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({self.vocab[idx]}) had {stats[pair]} occurrences")

    def save(self, vocab_name="vocab", merges_name="merges"):
        """
        Saves vocab and merges as pickle files after training is done. 
        """
        with open(f"{merges_name}.pkl", "wb") as f:
            pickle.dump(self.merges, f)

        with open(f"{vocab_name}.pkl", "wb") as f:
            pickle.dump(self.vocab, f)

    def load(self, *, vocab_name, merges_name):
        """
        Loads vocab and merges from pickle files.
        """
        if not (vocab_name.endswith(".pkl") and merges_name.endswith(".pkl")):
            raise ValueError("Filename must end with '.pkl'")

        with open(vocab_name, 'rb') as f:
            self.vocab = pickle.load(f)

        with open(merges_name, 'rb') as f:
            self.merges = pickle.load(f)       

if __name__ == "__main__":
    with open("test_text.txt", "r") as f:
        corpus = f.read()

    tokenizer = BpeSimple()
    tokenizer.train(301, corpus, verbose=True)
    tokenizer.save()
    tokenizer.load(vocab_name="vocab.pkl", merges_name="merges.pkl")

    text = "Hello, world! This is the process of tokenization."
    encoded = tokenizer.encode(text)
    print(f"Encoded: {encoded}")
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded}")