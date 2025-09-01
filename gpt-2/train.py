import torch

def train_model(model, train_loader, val_loader, num_epochs, optimizer, device):
    model.to(device)
    train_curve, val_curve = [], []

    for epoch in range(1, num_epochs + 1):
        # ---- train ----
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device).long()

            optimizer.zero_grad()
            logits = model(x)  # [B, N, vocab_size]
            # Sum loss over all tokens in this batch
            loss = torch.nn.functional.cross_entropy(logits.flatten(0,1), y.flatten()) # logits.flatten(0, 1): [B, N, vocab_size] -> [B*N, vocab_size], merges dim 0 and dim 1 only
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader) #Average loss
        train_curve.append(train_loss)

        # ---- validate ----
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device).long()
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(logits.flatten(0,1), y.flatten())
                running_loss += loss.item()

        val_loss = running_loss / len(val_loader)
        val_curve.append(val_loss)

        print(f"Epoch {epoch}/{num_epochs} | train={train_loss:.4f} | val={val_loss:.4f}")

    return train_curve, val_curve