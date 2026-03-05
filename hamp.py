from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F


TensorOrTuple = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


def _entropy_from_probs(probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Per-sample entropy H(p) for probs [B,K]."""
    p = probs.clamp_min(eps)
    return -(p * p.log()).sum(dim=1)


def _soft_label_entropy(p_true: float, k: int, eps: float = 1e-12) -> float:
    """Entropy of the HAMP soft-label template (Eq. 5) for given p_true."""
    if k <= 1:
        return 0.0
    q = (1.0 - p_true) / (k - 1)
    p_true = max(min(p_true, 1.0 - eps), eps)
    q = max(min(q, 1.0 - eps), eps)
    # H = -p log p - (k-1) q log q
    return float(
        -(p_true * torch.log(torch.tensor(p_true))).item()
        - (k - 1) * (q * torch.log(torch.tensor(q))).item()
    )


def _solve_p_for_gamma(gamma: float, k: int) -> float:
    gamma = float(max(min(gamma, 1.0), 0.0))
    if k <= 1:
        return 1.0
    target = gamma * float(torch.log(torch.tensor(float(k))).item())
    # If gamma==0 => allow very low entropy => p can be 1.0
    if target <= 0:
        return 1.0
    # If gamma==1 => uniform => p=1/k
    if abs(target - float(torch.log(torch.tensor(float(k))).item())) < 1e-12:
        return 1.0 / k

    lo = 1.0 / k
    hi = 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        h = _soft_label_entropy(mid, k)
        if h >= target:
            # entropy still high enough; we can increase p (reduce entropy)
            lo = mid
        else:
            hi = mid
    return float(lo)


def _make_soft_labels(hard_labels: torch.Tensor, k: int, p_true: float) -> torch.Tensor:
    """Create HAMP high-entropy soft labels (Eq. 5) for a batch."""
    device = hard_labels.device
    b = hard_labels.numel()
    q = (1.0 - p_true) / (k - 1) if k > 1 else 0.0
    y = torch.full((b, k), fill_value=q, device=device, dtype=torch.float32)
    y.scatter_(1, hard_labels.view(-1, 1), float(p_true))
    return y


def _infer_vocab_size(model) -> int:
    vs = getattr(getattr(model, "config", None), "vocab_size", None)
    if isinstance(vs, int) and vs > 0:
        return vs
    return 30522


@dataclass
class HAMP:
    """HAMP defense: training-time + testing-time output modification."""

    gamma: float = 0.95
    alpha: float = 0.001
    num_classes: int = 10
    # When applying output modification, we generate x_rand per batch.
    cache_random_inputs: bool = False
    _cached_xrand: Optional[TensorOrTuple] = None

    def __post_init__(self):
        self.gamma = float(self.gamma)
        self.alpha = float(self.alpha)
        self.num_classes = int(self.num_classes)
        self._p_true = _solve_p_for_gamma(self.gamma, self.num_classes)

    @property
    def p_true(self) -> float:
        return self._p_true

    @torch.no_grad()
    def _generate_random_inputs(self, model, x: TensorOrTuple) -> TensorOrTuple:
        if self.cache_random_inputs and (self._cached_xrand is not None):
            return self._cached_xrand

        if isinstance(x, tuple):
            input_ids, attention_mask = x
            b, seqlen = input_ids.shape
            vocab = _infer_vocab_size(model)

            # DistilBERT-style special tokens.
            CLS, SEP = 101, 102
            rand_ids = torch.randint(low=0, high=vocab, size=(b, seqlen), device=input_ids.device)
            if seqlen >= 1:
                rand_ids[:, 0] = CLS
            if seqlen >= 2:
                rand_ids[:, -1] = SEP
            rand_mask = torch.ones_like(attention_mask)
            xrand = (rand_ids, rand_mask)
        else:
            # Vision
            x_t = x
            mn = float(x_t.min().item())
            mx = float(x_t.max().item())
            if mn == mx:
                mn, mx = 0.0, 1.0
            xrand = torch.empty_like(x_t).uniform_(mn, mx)

        if self.cache_random_inputs:
            self._cached_xrand = xrand
        return xrand

    @torch.no_grad()
    def modify_output_probs(self, model, x: TensorOrTuple) -> torch.Tensor:
        """Testing-time output modification.
        """
        model.eval()

        # 1) Compute F(x)
        if isinstance(x, tuple):
            input_ids, attention_mask = x
            logits = model(input_ids, attention_mask=attention_mask)
        else:
            logits = model(x)
        probs = F.softmax(logits, dim=1)

        # 2) Compute F(x_rand)
        xrand = self._generate_random_inputs(model, x)
        if isinstance(xrand, tuple):
            rid, rmask = xrand
            logits_r = model(rid, attention_mask=rmask)
        else:
            logits_r = model(xrand)
        probs_r = F.softmax(logits_r, dim=1)

        order = torch.argsort(probs, dim=1, descending=True)
        rand_sorted, _ = torch.sort(probs_r, dim=1, descending=True)
        new_probs = torch.empty_like(probs)
        new_probs.scatter_(1, order, rand_sorted)
        return new_probs


def train_one_epoch_hamp(
    model,
    loader,
    optimizer,
    hamp: HAMP,
    device: str = "cuda",
):

    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for batch in loader:
        optimizer.zero_grad(set_to_none=True)

        if isinstance(batch, dict):
            input_ids = batch["input_ids"].to(device).long()
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device).long()
            logits = model(input_ids, attention_mask=attention_mask)
        else:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device).long()
            logits = model(images)

        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        soft_y = _make_soft_labels(labels, hamp.num_classes, hamp.p_true)


        loss_kl = F.kl_div(log_probs, soft_y, reduction="batchmean")
        ent = _entropy_from_probs(probs).mean()
        loss = loss_kl - hamp.alpha * ent  

        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * labels.size(0)
        preds = logits.argmax(dim=1)
        total_correct += int((preds == labels).sum().item())
        total += int(labels.size(0))

    return (total_loss / max(total, 1)), (total_correct / max(total, 1))


def train_or_load_hamp_model(
    *,
    model,
    train_loader,
    epochs: int,
    device: str,
    hamp: HAMP,
    lr: float = 0.01,
    momentum: float = 0.9,
    save_dir: str = "saved_models",
    ckpt_tag: str = "",
):
    
    os.makedirs(save_dir, exist_ok=True)
    tag = ckpt_tag.strip() or f"gamma{hamp.gamma}_alpha{hamp.alpha}"
    ckpt_path = os.path.join(save_dir, f"hamp_{tag}.pt")

    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"[HAMP] Loaded HAMP-trained model from {ckpt_path}")
        return model

    print(
        f"[HAMP] Training HAMP model for {epochs} epochs "
        f"(gamma={hamp.gamma}, alpha={hamp.alpha}, p_true={hamp.p_true:.4f})"
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    for ep in range(epochs):
        avg_loss, avg_acc = train_one_epoch_hamp(
            model, train_loader, optimizer, hamp=hamp, device=device
        )
        if (ep + 1) % 2 == 0 or ep == 0:
            print(f"[HAMP][EPOCH {ep+1:02d}] loss={avg_loss:.4f}, acc={avg_acc:.4f}")

    torch.save(model.state_dict(), ckpt_path)
    print(f"[HAMP] Saved HAMP-trained model to {ckpt_path}")
    return model
