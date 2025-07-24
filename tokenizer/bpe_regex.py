import regex as re

GPT_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

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
    while i < n:
        if i < n-1 and pair[0] == ids[i] and pair[1] == ids[i+1]:
            new_ids.append(idx)
            i += 2   #Escape the pair in ids
        else:
            new_ids.append(ids[i])
            i += 1

    return new_ids

class RegexTokenizer:
    def __init__(self, pattern=None):
        self.pattern = pattern if pattern is not None else GPT_SPLIT_PATTERN
        self.compiled_pattern = re.compile(self.pattern)
        self.vocab = {} #used in decode
        self.merges = {} #used in encode
        self.special_tokens = {}
        self.inverse_special_tokens = {}

    def train(self, vocab_size, text, verbose=False):
        """
        Trains the BPE tokenizer on the given corpus by first chunking based on regex pattern. 
        
        Uses stats function to get most occuring pair in the chunks and merges it till vocab size is reached.
        """
        if vocab_size < 256:
            raise ValueError("Vocab size must be greater than 256")
        
        num_merges = vocab_size - 256    
        text_chunks = re.findall(self.compiled_pattern, text)
        ids = [list(ch.encode('utf-8')) for ch in text_chunks]
        self.vocab = {idx: bytes([idx]) for idx in range(256)}

        for i in range(num_merges):
            stats = {}
            for chunk_ids in ids:
                stats = get_stats(chunk_ids, stats)
            
            if not stats:
                # If no pairs are found, it means the list contains only one id,
                # which must already exist in self.vocab (indices 0–255), so continue.
                continue

            pair = max(stats, key=stats.get)
            idx = 256+i

            ids = [merge(chunck_ids, pair, idx) for chunck_ids in ids]
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({self.vocab[idx]}) had {stats[pair]} occurrences")
        

    def register_special_tokens(self, special_tokens: list):
        """
        Registers special tokens in the vocab.
        """
        vocab_len = len(self.vocab)
        if vocab_len == 0:
            raise RuntimeError("Tokenizer not trained. Please train the tokenizer before adding special tokens in the vocab.")

        special_tokens = list(set(special_tokens)) #Ensure only unique special tokens
        self.special_tokens = {special_token: (vocab_len + i) for i,special_token in enumerate(special_tokens)} #i starts with 0 and last idx in self.vocab is len(self.vocab)-1
        self.inverse_special_tokens = {v:k for k,v in self.special_tokens.items()}
    
    def decode(self, ids):
        """
        Converts a list of token ids into a string using the vocab.
        """
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode('utf-8'))
            else:
                raise ValueError(f"Invalid token id: {idx}")

        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode('utf-8', errors='replace')
        return text

    def _encode_chunk(self, chunk):
        chunk_ids = list(chunk.encode('utf-8'))
        
        while True:
            stats = get_stats(chunk_ids)
            mergable_pairs = [pair for pair in stats if pair in self.merges]

            if not mergable_pairs:
                break

            best_pair = min(mergable_pairs, key= lambda p: self.merges[p]) #Find minimum idx, so that merges are performed from lowest index to the next one
            new_token = self.merges[best_pair]
            chunk_ids = merge(chunk_ids, best_pair, new_token)

        return chunk_ids
        
    
    def encode_ordinary(self, text):
        """
        Converts ordinary text without special tokens into a list of token ids.
        """
        text_chunks = re.findall(self.compiled_pattern, text)
        ids = []

        for chunk in text_chunks:
            chunk_ids = self._encode_chunk(chunk)
            ids.extend(chunk_ids)
        return ids

    def encode(self, text, allowed_special="none_raise"):
        """
        Unlike encode_ordinary which handles only texts, this function handles special tokens.
        allowed_special: can be "all" | "none" | "none_raise" or a custom set of special tokens 
        """
        special = None

        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":  
            #if none_raise, then an error is raised if any special token is encountered in text
            special = {}
            if any(token in text for token in self.special_tokens):
                raise AssertionError("Text contains a special token")
        elif isinstance(allowed_special, set):
            if len(self.special_tokens) == 0:
                raise ValueError(f"Special tokens are not registered in the vocab. Register using method {self.register_special_tokens.__name__}")
            special = {k:v for k,v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")

        if not special:
            #No special tokens are there
            return self.encode_ordinary(text)
        
        #If there are specail tokens, we form a regex pattern to split text such that special tokens seperated from ordinary text
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"   #Example (<\|endoftext\|>|<\|startoftext\|>)
        text_split = re.split(special_pattern, text)

        ids = []
        for part in text_split:
            if part in special:
                # If the split part is a special token then directly assing idx from special
                ids.append(special[part])
            else:
                # If the split part is ordinary text
                ids.extend(self.encode_ordinary(part))
        return ids
        
if __name__ == "__main__":
    with open("test_text.txt", "r") as f:
        text = f.read()
    
    tokenizer = RegexTokenizer()
    tokenizer.train(1000, text, verbose=True)
    tokenizer.register_special_tokens(["<|startoftext|>", "<|endoftext|>"])

    my_text = my_txt = "Hi, How are you doing?<|endoftext|>"
    encoded = tokenizer.encode(my_text, allowed_special="all")
    print(f"Encoded: {encoded}")
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded}")
