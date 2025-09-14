import torch

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())

def generate_token_ids(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    # idx is (batch, n_tokens) array of indices in the current context
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:] # Crop current context if it exceeds the supported context size

        #get predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values and replace the rest with -inf which will evnentually be zero after softmax
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            replacing_val = torch.tensor(float("-inf")).to(logits.device)
            logits = torch.where(logits < min_val, replacing_val, logits)

        # Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature
            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
        else:
            # Apply softmax to get probabilities
            probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

            # Get the idx of the vocab entry with the highest probability value
            idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)

        if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break
        
        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)
    return idx

def generate_text(model, tokenizer, device, start_context, context_size, max_new_tokens=128):
    model.eval()

    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    
    with torch.no_grad():
        token_ids = generate_token_ids(model=model, idx=encoded, max_new_tokens=max_new_tokens, context_size=context_size)
    
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    return decoded_text