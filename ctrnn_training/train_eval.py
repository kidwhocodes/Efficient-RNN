import torch
import torch.nn as nn

def train_epoch(model, data, device, opt, criterion, steps=50, last_only=True, clip=1.0):
    model.train()
    total_loss, total_count = 0.0, 0
    for _ in range(steps):
        x, y = data.sample_batch()
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        if last_only:
            loss = criterion(logits[-1], y[-1])
            N = y[-1].numel()
        else:
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            N = y.numel()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        total_loss += float(loss) * N
        total_count += N
    return total_loss / max(1, total_count)

@torch.no_grad()
def evaluate(model, data, device, criterion, steps=20, last_only=True):
    model.eval()
    total_loss, total_correct, total_count = 0.0, 0, 0
    for _ in range(steps):
        x, y = data.sample_batch()
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        if last_only:
            loss = criterion(logits[-1], y[-1])
            pred = logits[-1].argmax(-1)
            N = y[-1].numel()
            correct = (pred == y[-1]).sum().item()
        else:
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            pred = logits.argmax(-1)
            N = y.numel()
            correct = (pred == y).sum().item()
        total_loss += float(loss) * N
        total_correct += correct
        total_count += N
    return total_loss / max(1, total_count), total_correct / max(1, total_count)
