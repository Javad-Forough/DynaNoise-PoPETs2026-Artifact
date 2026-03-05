import torch
import torch.nn.functional as F

def train_one_epoch(model, train_loader, optimizer, device: str = "cuda"):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device).long()

        optimizer.zero_grad()
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * labels.size(0)
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_samples += int(labels.size(0))

    return total_loss / max(1, total_samples), total_correct / max(1, total_samples)

def eval_model(model, test_loader, device: str = "cuda"):
    model.eval()
    total_correct, total_samples = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device).long()
            logits = model(images)
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total_samples += int(labels.size(0))
    return total_correct / max(1, total_samples)

def eval_model_with_noise(model, test_loader, dyna_noise, device: str = "cuda"):
    model.eval()
    total_correct, total_samples = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device).long()
            probs = dyna_noise.forward(model(images))
            total_correct += int((probs.argmax(dim=1) == labels).sum().item())
            total_samples += int(labels.size(0))
    return total_correct / max(1, total_samples)
