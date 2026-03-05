from __future__ import annotations

import os
import json
import hashlib
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from models import get_model
from train_eval import train_one_epoch


# =============================================================================
# Metrics
# =============================================================================

def compute_metrics(preds, labels) -> Dict[str, float]:
    preds = np.asarray(preds).astype(int)
    labels = np.asarray(labels).astype(int)

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)  # == TPR
    f1 = f1_score(labels, preds, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    tpr = tp / max(1, (tp + fn))
    tnr = tn / max(1, (tn + fp))
    bal_acc = 0.5 * (tpr + tnr)

    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "precision": float(prec),
        "recall": float(rec),
        "tpr": float(tpr),
        "tnr": float(tnr),
        "f1": float(f1),
        "attack_success_rate": float(acc),  # backward compat
    }


# =============================================================================
# Helpers: model forward and defense application
# =============================================================================

def _forward_logits_only(model, batch, device: str = "cuda"):
    model.eval()
    with torch.no_grad():
        if isinstance(batch, dict):
            input_ids = batch["input_ids"].to(device).long()
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids, attention_mask=attention_mask)
        else:
            images, _ = batch
            images = images.to(device)
            logits = model(images)
    return logits


def _forward_logits_and_labels(model, batch, device: str = "cuda"):
    model.eval()
    with torch.no_grad():
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
    return logits, labels


def _batch_size_of(batch) -> int:
    if isinstance(batch, dict):
        return int(batch["input_ids"].size(0))
    _, labs = batch
    return int(labs.size(0))


def _extract_model_inputs_for_hamp(batch, device: str = "cuda"):
    if isinstance(batch, dict):
        input_ids = batch["input_ids"].to(device).long()
        attention_mask = batch["attention_mask"].to(device)
        return (input_ids, attention_mask)
    images, _ = batch
    return images.to(device)


def _safe_probs(probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Clamp + renormalize + remove NaN/Inf."""
    probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
    probs = probs.clamp(min=eps, max=1.0 - eps)
    probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(eps)
    return probs


def _sanitize_features_np(X: np.ndarray) -> np.ndarray:
    """Make sklearn-safe: no NaN/Inf."""
    return np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32, copy=False)


def _apply_defense_or_softmax(
    logits,
    dyna_noise=None,
    hamp=None,
    model=None,
    batch=None,
    device: str = "cuda",
):
    """Returns probabilities AFTER defense (if any), with numeric safety."""
    eps = 1e-12

    if hamp is not None:
        if model is None or batch is None:
            raise ValueError("HAMP defense requires `model` and `batch`.")
        x = _extract_model_inputs_for_hamp(batch, device=device)
        probs = hamp.modify_output_probs(model, x)
    elif dyna_noise is not None:
        probs = dyna_noise.forward(logits)
    else:
        probs = F.softmax(logits, dim=1)

    return _safe_probs(probs, eps=eps)


# =============================================================================
# 1) CONFIDENCE THRESHOLD ATTACK
# =============================================================================

def attack_confidence_threshold(probs, threshold: float = 0.9):
    max_probs, _ = torch.max(probs, dim=1)
    return (max_probs > threshold).long()


def compute_confidence_attack_metrics(
    model,
    in_loader,
    out_loader,
    threshold: float = 0.9,
    dyna_noise=None,
    hamp=None,
    device: str = "cuda",
):
    model.eval()
    all_preds, all_labels = [], []

    for batch in in_loader:
        logits = _forward_logits_only(model, batch, device=device)
        probs = _apply_defense_or_softmax(logits, dyna_noise=dyna_noise, hamp=hamp, model=model, batch=batch, device=device)
        preds_batch = attack_confidence_threshold(probs, threshold)
        all_preds.extend(preds_batch.detach().cpu().numpy())
        all_labels.extend([1] * _batch_size_of(batch))

    for batch in out_loader:
        logits = _forward_logits_only(model, batch, device=device)
        probs = _apply_defense_or_softmax(logits, dyna_noise=dyna_noise, hamp=hamp, model=model, batch=batch, device=device)
        preds_batch = attack_confidence_threshold(probs, threshold)
        all_preds.extend(preds_batch.detach().cpu().numpy())
        all_labels.extend([0] * _batch_size_of(batch))

    return compute_metrics(np.array(all_preds), np.array(all_labels))


# =============================================================================
# 2) LOSS THRESHOLD ATTACK
# =============================================================================

def attack_loss_threshold(probs, true_labels, threshold: float = 0.5):
    eps = 1e-12
    p_true = torch.gather(probs, 1, true_labels.unsqueeze(1)).squeeze(1)
    p_true = torch.clamp(p_true, min=eps, max=1.0)
    xent = -torch.log(p_true)
    return (xent < threshold).long()


def compute_loss_attack_metrics(
    model,
    in_loader,
    out_loader,
    threshold: float = 0.5,
    dyna_noise=None,
    hamp=None,
    device: str = "cuda",
):
    model.eval()
    all_preds, all_labels = [], []

    for batch in in_loader:
        logits, labels = _forward_logits_and_labels(model, batch, device=device)
        probs = _apply_defense_or_softmax(logits, dyna_noise=dyna_noise, hamp=hamp, model=model, batch=batch, device=device)
        preds_batch = attack_loss_threshold(probs, labels, threshold)
        all_preds.extend(preds_batch.detach().cpu().numpy())
        all_labels.extend([1] * labels.size(0))

    for batch in out_loader:
        logits, labels = _forward_logits_and_labels(model, batch, device=device)
        probs = _apply_defense_or_softmax(logits, dyna_noise=dyna_noise, hamp=hamp, model=model, batch=batch, device=device)
        preds_batch = attack_loss_threshold(probs, labels, threshold)
        all_preds.extend(preds_batch.detach().cpu().numpy())
        all_labels.extend([0] * labels.size(0))

    return compute_metrics(np.array(all_preds), np.array(all_labels))


# =============================================================================
# 3) (Legacy) Feature extraction helper (still used elsewhere)
# =============================================================================

def _extract_features_from_logits_and_labels(
    logits,
    labels,
    dyna_noise=None,
    hamp=None,
    model=None,
    batch=None,
    device: str = "cuda",
):
    eps = 1e-12
    probs = _apply_defense_or_softmax(
        logits,
        dyna_noise=dyna_noise,
        hamp=hamp,
        model=model,
        batch=batch,
        device=device,
    )

    max_conf, _ = torch.max(probs, dim=1)
    p_true = torch.gather(probs, 1, labels.unsqueeze(1)).squeeze(1).clamp_min(eps)
    xent = -torch.log(p_true)

    sorted_probs, _ = torch.sort(probs, descending=True)
    margin = sorted_probs[:, 0] - sorted_probs[:, 1]

    entropy = -torch.sum(probs * torch.log(probs + eps), dim=1)
    feats = torch.stack([max_conf, xent, margin, entropy], dim=1)
    return feats.detach().cpu().numpy().tolist()


def extract_features(model, inputs, labels, device: str = "cuda", dyna_noise=None, hamp=None, batch=None):
    model.eval()
    with torch.no_grad():
        if isinstance(inputs, dict):
            logits = model(
                inputs["input_ids"].to(device).long(),
                attention_mask=inputs["attention_mask"].to(device),
            )
            labels = labels.to(device).long()
        else:
            logits = model(inputs.to(device))
            labels = labels.to(device).long()

    return _extract_features_from_logits_and_labels(
        logits,
        labels,
        dyna_noise=dyna_noise,
        hamp=hamp,
        model=model,
        batch=batch,
        device=device,
    )


# =============================================================================
# 4) SHADOW MODEL ATTACK
# =============================================================================

def _entropy_from_probs(probs: torch.Tensor) -> torch.Tensor:
    eps = 1e-12
    return -(probs * torch.log(probs + eps)).sum(dim=1)


def _mentropy_from_probs(probs: torch.Tensor, true_labels: torch.Tensor) -> torch.Tensor:
    eps = 1e-12
    probs = probs.clamp(min=eps, max=1.0 - eps)

    p_y = torch.gather(probs, 1, true_labels.unsqueeze(1)).squeeze(1)
    term1 = -(1.0 - p_y) * torch.log(p_y)

    log1m = torch.log(1.0 - probs)
    term_all = probs * log1m
    term_y = torch.gather(term_all, 1, true_labels.unsqueeze(1)).squeeze(1)
    term2 = -(term_all.sum(dim=1) - term_y)

    out = term1 + term2
    return torch.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)


def train_shadow_model(
    loader_in,
    loader_out,  # kept for API compat
    num_classes: int = 10,
    epochs: int = 5,
    device: str = "cuda",
    model_name: str = "alexnet",
    dataset_name: str = "cifar10",
    seed: int = 42,
    *,
    shadow_models_dir: str = "saved_models_shadow",
    shadow_model_path: Optional[str] = None,
    load_only: bool = False,
):

    import torch.optim as optim

    os.makedirs(shadow_models_dir, exist_ok=True)

    if shadow_model_path is None:
        filename = f"{model_name}_{dataset_name}_shadow_seed{seed}_{epochs}epochs.pt"
        shadow_model_path = os.path.join(shadow_models_dir, filename)

    shadow_model = get_model(model_name, num_classes).to(device)

    if os.path.exists(shadow_model_path):
        shadow_model.load_state_dict(torch.load(shadow_model_path, map_location=device))
        print(f"[INFO] Loaded shadow model from {shadow_model_path}")
        return shadow_model.eval()

    if load_only:
        raise FileNotFoundError(f"shadow model not found: {shadow_model_path}")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    optimizer = optim.SGD(shadow_model.parameters(), lr=0.01, momentum=0.9)
    shadow_model.train()
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(shadow_model, loader_in, optimizer, device=device)
        print(f"[SHADOW TRAIN EPOCH {epoch+1}] train_loss={train_loss:.4f}, train_acc={train_acc:.4f}")

    torch.save(shadow_model.state_dict(), shadow_model_path)
    print(f"[INFO] Shadow model saved to {shadow_model_path}")
    return shadow_model.eval()


def _extract_shadow_attack_features_from_loader(
    model,
    loader,
    dyna_noise=None,
    hamp=None,
    device: str = "cuda",
    mc_samples: int = 1,
) -> np.ndarray:
   
    model.eval()
    M = max(1, int(mc_samples))
    feats = []
    eps = 1e-12

    for batch in loader:
        with torch.no_grad():
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

        probs_accum = None
        for _ in range(M):
            probs = _apply_defense_or_softmax(
                logits,
                dyna_noise=dyna_noise,
                hamp=hamp,
                model=model,
                batch=batch,
                device=device,
            )
            probs_accum = probs if probs_accum is None else (probs_accum + probs)

        probs = probs_accum / float(M)
        probs = _safe_probs(probs, eps=eps)

        conf, _ = probs.max(dim=1)
        ent = _entropy_from_probs(probs)
        ment = _mentropy_from_probs(probs, labels)
        p_true = torch.gather(probs, 1, labels.unsqueeze(1)).squeeze(1).clamp(min=eps, max=1.0 - eps)
        loss = -torch.log(p_true)

        Xb = torch.stack([conf, ent, ment, p_true, loss], dim=1)
        Xb = torch.nan_to_num(Xb, nan=0.0, posinf=1e6, neginf=-1e6)
        feats.append(Xb.detach().cpu().numpy())

    X = np.concatenate(feats, axis=0) if feats else np.zeros((0, 5), dtype=np.float32)
    return _sanitize_features_np(X)


def _gather_shadow_features(
    shadow_model,
    shadow_in_loader,
    shadow_out_loader,
    dyna_noise=None,
    hamp=None,
    device: str = "cuda",
    mc_samples: int = 1,
):
    X_in = _extract_shadow_attack_features_from_loader(
        shadow_model, shadow_in_loader, dyna_noise=dyna_noise, hamp=hamp, device=device, mc_samples=mc_samples
    )
    X_out = _extract_shadow_attack_features_from_loader(
        shadow_model, shadow_out_loader, dyna_noise=dyna_noise, hamp=hamp, device=device, mc_samples=mc_samples
    )
    X = np.vstack([X_in, X_out])
    Y = np.concatenate([np.ones(len(X_in), dtype=int), np.zeros(len(X_out), dtype=int)])
    return X, Y


def _gather_target_features_for_shadow_attack(
    target_model,
    target_in_loader,
    target_out_loader,
    dyna_noise=None,
    hamp=None,
    device: str = "cuda",
    mc_samples: int = 1,
):
    X_in = _extract_shadow_attack_features_from_loader(
        target_model, target_in_loader, dyna_noise=dyna_noise, hamp=hamp, device=device, mc_samples=mc_samples
    )
    X_out = _extract_shadow_attack_features_from_loader(
        target_model, target_out_loader, dyna_noise=dyna_noise, hamp=hamp, device=device, mc_samples=mc_samples
    )
    X = np.vstack([X_in, X_out])
    Y = np.concatenate([np.ones(len(X_in), dtype=int), np.zeros(len(X_out), dtype=int)])
    return X, Y


def train_attack_model(X, Y):
    from sklearn.linear_model import LogisticRegression

    X = _sanitize_features_np(X)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, Y)
    return clf


def compute_shadow_attack_metrics(
    model,
    target_in_loader,
    target_out_loader,
    shadow_in_loader,
    shadow_out_loader,
    device: str = "cuda",
    dyna_noise=None,
    hamp=None,
    epochs: int = 5,
    model_name: str = "alexnet",
    dataset_name: str = "cifar10",
    seed: int = 42,
    *,
    shadow_models_dir: str = "saved_models_shadow",
    shadow_model_path: Optional[str] = None,
    load_only: bool = False,
):
    
    num_classes = 10

    shadow_model = train_shadow_model(
        loader_in=shadow_in_loader,
        loader_out=shadow_out_loader,
        num_classes=num_classes,
        epochs=epochs,
        device=device,
        model_name=model_name,
        dataset_name=dataset_name,
        seed=seed,
        shadow_models_dir=shadow_models_dir,
        shadow_model_path=shadow_model_path,
        load_only=load_only,
    )

    mc = 1  # keep as in your original file

    X_shadow, Y_shadow = _gather_shadow_features(
        shadow_model=shadow_model,
        shadow_in_loader=shadow_in_loader,
        shadow_out_loader=shadow_out_loader,
        dyna_noise=None,
        hamp=None,
        device=device,
        mc_samples=mc,
    )
    attack_clf = train_attack_model(X_shadow, Y_shadow)

    X_target, Y_target = _gather_target_features_for_shadow_attack(
        target_model=model,
        target_in_loader=target_in_loader,
        target_out_loader=target_out_loader,
        dyna_noise=dyna_noise,
        hamp=hamp,
        device=device,
        mc_samples=mc,
    )
    y_pred = attack_clf.predict(_sanitize_features_np(X_target))
    return compute_metrics(y_pred, Y_target)


# =============================================================================
# 5) LiRA
# =============================================================================

def _compute_per_sample_losses_and_labels(
    model,
    loader,
    device: str = "cuda",
    dyna_noise=None,
    hamp=None,
    mc_samples: int = 1,
):
    
    model.eval()
    losses = []
    ytrue = []
    M = max(1, int(mc_samples))
    eps = 1e-12

    for batch in loader:
        with torch.no_grad():
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

        if dyna_noise is None and hamp is None:
            probs = _safe_probs(torch.softmax(logits, dim=1), eps=eps)
            p_true = torch.gather(probs, 1, labels.unsqueeze(1)).squeeze(1).clamp_min(eps)
            xent = -torch.log(p_true)
            losses.extend(xent.detach().cpu().tolist())
            ytrue.extend(labels.detach().cpu().tolist())
            continue

        xent_accum = None
        for _ in range(M):
            probs = _apply_defense_or_softmax(
                logits, dyna_noise=dyna_noise, hamp=hamp, model=model, batch=batch, device=device
            )
            p_true = torch.gather(probs, 1, labels.unsqueeze(1)).squeeze(1).clamp_min(eps)
            xent = -torch.log(p_true)
            xent_accum = xent if xent_accum is None else (xent_accum + xent)

        xent_mean = xent_accum / float(M)
        xent_mean = torch.nan_to_num(xent_mean, nan=0.0, posinf=1e6, neginf=-1e6)

        losses.extend(xent_mean.detach().cpu().tolist())
        ytrue.extend(labels.detach().cpu().tolist())

    losses = np.asarray(losses, dtype=np.float64)
    labels = np.asarray(ytrue, dtype=np.int64)
    losses = np.nan_to_num(losses, nan=0.0, posinf=1e6, neginf=-1e6)
    return losses, labels


def _gaussian_fit(arr: np.ndarray):
    arr = np.asarray(arr, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
    if arr.size == 0:
        return 0.0, 1e-6
    mu = float(np.mean(arr))
    sigma = float(np.std(arr, ddof=1)) if arr.size > 1 else 1e-6
    return mu, max(sigma, 1e-6)


def _gaussian_logpdf(x, mu, sigma):
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
    var = float(sigma) * float(sigma)
    return -0.5 * np.log(2 * np.pi * var) - (x - mu) ** 2 / (2 * var)


def _lira_cache_tag(model_name, dataset_name, seed, epochs, num_shadows, split_hash=None):
    payload = {
        "model": model_name,
        "dataset": dataset_name.lower(),
        "seed": int(seed),
        "epochs": int(epochs),
        "num_shadows": int(num_shadows),
        "split_hash": str(split_hash) if split_hash is not None else "",
    }
    h = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    return f"{model_name}_{dataset_name}_lira_seed{seed}_E{epochs}_S{num_shadows}_{h}"


def _train_or_load_shadow_models_for_lira_cached(
    loader_in,
    num_classes: int,
    num_shadows: int = 3,
    epochs: int = 5,
    device: str = "cuda",
    model_name: str = "alexnet",
    dataset_name: str = "cifar10",
    seed: int = 42,
    split_hash: str = None,
):
   
    import torch.optim as optim

    shadow_models = []
    shadow_folder = "saved_models_shadow"
    os.makedirs(shadow_folder, exist_ok=True)

    tag = _lira_cache_tag(model_name, dataset_name, seed, epochs, num_shadows, split_hash=split_hash)

    for i in range(num_shadows):
        path = os.path.join(shadow_folder, f"{tag}_shadow{i}.pt")

        m = get_model(model_name, num_classes).to(device)
        if os.path.exists(path):
            m.load_state_dict(torch.load(path, map_location=device))
            print(f"[INFO] Loaded LiRA shadow model #{i} from {path}")
            shadow_models.append(m.eval())
            continue

        torch.manual_seed(seed + 1000 + i)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + 1000 + i)

        print(f"[LiRA] Training shadow model #{i} for {epochs} epochs (seed={seed+1000+i}) ...")
        opt = optim.SGD(m.parameters(), lr=0.01, momentum=0.9)
        m.train()
        for ep in range(epochs):
            tr_loss, tr_acc = train_one_epoch(m, loader_in, opt, device=device)
            if (ep + 1) % 2 == 0 or (ep == 0) or (ep + 1 == epochs):
                print(f"[LiRA][Shadow {i}][Epoch {ep+1:02d}] loss={tr_loss:.4f} acc={tr_acc:.4f}")

        torch.save(m.state_dict(), path)
        print(f"[INFO] Saved LiRA shadow model #{i} to {path}")
        shadow_models.append(m.eval())

    return shadow_models


def _load_lira_shadows_fixed_filenames(
    *,
    num_classes: int,
    num_shadows: int,
    device: str,
    model_name: str,
    lira_shadow_models_dir: str,
    lira_filename_template: str,
):
    models = []
    for i in range(int(num_shadows)):
        fname = lira_filename_template.format(i=i)
        path = os.path.join(lira_shadow_models_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"[ARTIFACT] LiRA shadow model missing: {path}")
        m = get_model(model_name, num_classes).to(device)
        m.load_state_dict(torch.load(path, map_location=device), strict=True)
        m.eval()
        print(f"[INFO] Loaded LiRA shadow model #{i} from {path}")
        models.append(m)
    return models


def compute_lira_attack_metrics(
    target_model,
    target_in_loader,
    target_out_loader,
    shadow_in_loader,
    shadow_out_loader,  
    device: str = "cuda",
    dyna_noise=None,
    hamp=None,
    epochs: int = 5,
    num_shadows: int = 3,
    model_name: str = "alexnet",
    dataset_name: str = "cifar10",
    seed: int = 42,
    split_hash: str = None,
    *,
    lira_shadow_models_dir: str = "saved_models_shadow",
    lira_filename_template: str = "lira_shadow{i}.pt",
    load_only: bool = False,
):
    
    num_classes = 10
    mc = 1  

    if load_only:
        shadow_models = _load_lira_shadows_fixed_filenames(
            num_classes=num_classes,
            num_shadows=num_shadows,
            device=device,
            model_name=model_name,
            lira_shadow_models_dir=lira_shadow_models_dir,
            lira_filename_template=lira_filename_template,
        )
    else:
        shadow_models = _train_or_load_shadow_models_for_lira_cached(
            loader_in=shadow_in_loader,
            num_classes=num_classes,
            num_shadows=num_shadows,
            epochs=epochs,
            device=device,
            model_name=model_name,
            dataset_name=dataset_name,
            seed=seed,
            split_hash=split_hash,
        )

    
    all_in_losses, all_in_y = [], []
    all_out_losses, all_out_y = [], []

    for sm in shadow_models:
        li, yi = _compute_per_sample_losses_and_labels(sm, shadow_in_loader, device=device, dyna_noise=None, hamp=None, mc_samples=mc)
        lo, yo = _compute_per_sample_losses_and_labels(sm, shadow_out_loader, device=device, dyna_noise=None, hamp=None, mc_samples=mc)
        all_in_losses.append(li); all_in_y.append(yi)
        all_out_losses.append(lo); all_out_y.append(yo)

    in_losses = np.concatenate(all_in_losses) if all_in_losses else np.array([], dtype=np.float64)
    in_y = np.concatenate(all_in_y) if all_in_y else np.array([], dtype=np.int64)
    out_losses = np.concatenate(all_out_losses) if all_out_losses else np.array([], dtype=np.float64)
    out_y = np.concatenate(all_out_y) if all_out_y else np.array([], dtype=np.int64)

    mu_in_g, sigma_in_g = _gaussian_fit(in_losses)
    mu_out_g, sigma_out_g = _gaussian_fit(out_losses)

    mu_in, sg_in, mu_out, sg_out = {}, {}, {}, {}
    for c in range(num_classes):
        li_c = in_losses[in_y == c]
        lo_c = out_losses[out_y == c]
        mu_in[c], sg_in[c] = _gaussian_fit(li_c) if li_c.size > 1 else (mu_in_g, sigma_in_g)
        mu_out[c], sg_out[c] = _gaussian_fit(lo_c) if lo_c.size > 1 else (mu_out_g, sigma_out_g)

    
    llr_in_shadow = _gaussian_logpdf(in_losses, mu_in_g, sigma_in_g) - _gaussian_logpdf(in_losses, mu_out_g, sigma_out_g)
    llr_out_shadow = _gaussian_logpdf(out_losses, mu_in_g, sigma_in_g) - _gaussian_logpdf(out_losses, mu_out_g, sigma_out_g)

    labels_shadow = np.concatenate([np.ones_like(llr_in_shadow, dtype=int), np.zeros_like(llr_out_shadow, dtype=int)])
    preds_pos = np.concatenate([llr_in_shadow > 0.0, llr_out_shadow > 0.0]).astype(int)
    preds_neg = np.concatenate([llr_in_shadow < 0.0, llr_out_shadow < 0.0]).astype(int)
    invert_rule = (accuracy_score(labels_shadow, preds_neg) > accuracy_score(labels_shadow, preds_pos))

    all_preds, all_labels = [], []

    def _score_loader(loader, true_mem_label: int):
        losses, y = _compute_per_sample_losses_and_labels(target_model, loader, device=device, dyna_noise=dyna_noise, hamp=hamp, mc_samples=mc)
        llr = np.zeros_like(losses, dtype=np.float64)
        for i in range(losses.shape[0]):
            c = int(y[i])
            llr[i] = _gaussian_logpdf(losses[i], mu_in[c], sg_in[c]) - _gaussian_logpdf(losses[i], mu_out[c], sg_out[c])

        preds = (llr < 0.0).astype(int) if invert_rule else (llr > 0.0).astype(int)
        labels = np.full_like(preds, fill_value=true_mem_label)
        all_preds.extend(list(preds))
        all_labels.extend(list(labels))

    _score_loader(target_in_loader, 1)
    _score_loader(target_out_loader, 0)

    return compute_metrics(np.array(all_preds), np.array(all_labels))


# =============================================================================
# 6) SM21 Metric-based attacks
# =============================================================================

def _learn_class_thresholds(metric_in, metric_out, class_in, class_out, num_classes: int, direction: str = "leq"):
    def best_tau(vals, labs):
        uniq = np.unique(vals)
        if uniq.size == 0:
            return 0.0, 0.5
        best_acc, best_t = -1.0, float(uniq[0])
        for t in uniq:
            pred = (vals <= t).astype(int) if direction == "leq" else (vals >= t).astype(int)
            acc = (pred == labs).mean()
            if acc > best_acc:
                best_acc, best_t = acc, float(t)
        return best_t, best_acc

    vals_all = np.concatenate([metric_in, metric_out])
    labs_all = np.concatenate([np.ones_like(metric_in, dtype=int), np.zeros_like(metric_out, dtype=int)])
    global_tau, _ = best_tau(vals_all, labs_all)

    taus = {}
    for c in range(num_classes):
        in_mask = (class_in == c)
        out_mask = (class_out == c)
        vals_c = np.concatenate([metric_in[in_mask], metric_out[out_mask]])
        labs_c = np.concatenate([np.ones(in_mask.sum(), dtype=int), np.zeros(out_mask.sum(), dtype=int)])
        if vals_c.size == 0:
            continue
        tau_c, _ = best_tau(vals_c, labs_c)
        taus[c] = tau_c

    return taus, global_tau


def compute_entropy_attack_metrics_sm21(
    model,
    target_in_loader,
    target_out_loader,
    shadow_in_loader,
    shadow_out_loader,
    device: str = "cuda",
    dyna_noise=None,
    hamp=None,
    shadow_epochs: int = 5,
    model_name: str = "alexnet",
    dataset_name: str = "cifar10",
    seed: int = 42,
    split_hash: str = None,  # accepted for compatibility
    *,
    shadow_models_dir: str = "saved_models_shadow",
    shadow_model_path: Optional[str] = None,
    load_only: bool = False,
):
    num_classes = 10

    shadow_model = train_shadow_model(
        shadow_in_loader,
        shadow_out_loader,
        num_classes=num_classes,
        epochs=shadow_epochs,
        device=device,
        model_name=model_name,
        dataset_name=dataset_name,
        seed=seed,
        shadow_models_dir=shadow_models_dir,
        shadow_model_path=shadow_model_path,
        load_only=load_only,
    )

    def collect_shadow(loader):
        ent_list = []
        cls_list = []
        for batch in loader:
            logits, _labels = _forward_logits_and_labels(shadow_model, batch, device=device)
            probs = _apply_defense_or_softmax(logits, dyna_noise=None, hamp=None, model=shadow_model, batch=batch, device=device)
            ent = _entropy_from_probs(probs).detach().cpu().numpy()
            yhat = probs.argmax(dim=1).detach().cpu().numpy()
            ent_list.append(ent)
            cls_list.append(yhat)
        return np.concatenate(ent_list), np.concatenate(cls_list)

    ent_in, cls_in = collect_shadow(shadow_in_loader)
    ent_out, cls_out = collect_shadow(shadow_out_loader)

    taus, global_tau = _learn_class_thresholds(ent_in, ent_out, cls_in, cls_out, num_classes=num_classes, direction="leq")

    all_preds, all_labels = [], []

    def score_target(loader, true_mem_label: int):
        for batch in loader:
            logits, _labels = _forward_logits_and_labels(model, batch, device=device)
            probs = _apply_defense_or_softmax(logits, dyna_noise=dyna_noise, hamp=hamp, model=model, batch=batch, device=device)

            ent = _entropy_from_probs(probs)
            yhat = probs.argmax(dim=1)

            ent_np = np.nan_to_num(ent.detach().cpu().numpy(), nan=0.0, posinf=1e6, neginf=-1e6)
            yhat_np = yhat.detach().cpu().numpy()

            pred = np.zeros_like(ent_np, dtype=int)
            for i in range(ent_np.shape[0]):
                tau = taus.get(int(yhat_np[i]), global_tau)
                pred[i] = 1 if ent_np[i] <= tau else 0

            all_preds.extend(pred.tolist())
            all_labels.extend([true_mem_label] * len(pred))

    score_target(target_in_loader, 1)
    score_target(target_out_loader, 0)

    return compute_metrics(np.array(all_preds), np.array(all_labels))


def compute_mentropy_attack_metrics_sm21(
    model,
    target_in_loader,
    target_out_loader,
    shadow_in_loader,
    shadow_out_loader,
    device: str = "cuda",
    dyna_noise=None,
    hamp=None,
    shadow_epochs: int = 5,
    model_name: str = "alexnet",
    dataset_name: str = "cifar10",
    seed: int = 42,
    split_hash: str = None,  # accepted for compatibility
    *,
    shadow_models_dir: str = "saved_models_shadow",
    shadow_model_path: Optional[str] = None,
    load_only: bool = False,
):
    num_classes = 10

    shadow_model = train_shadow_model(
        shadow_in_loader,
        shadow_out_loader,
        num_classes=num_classes,
        epochs=shadow_epochs,
        device=device,
        model_name=model_name,
        dataset_name=dataset_name,
        seed=seed,
        shadow_models_dir=shadow_models_dir,
        shadow_model_path=shadow_model_path,
        load_only=load_only,
    )

    def collect_shadow(loader):
        m_list = []
        y_list = []
        for batch in loader:
            logits, labels = _forward_logits_and_labels(shadow_model, batch, device=device)
            probs = _apply_defense_or_softmax(logits, dyna_noise=None, hamp=None, model=shadow_model, batch=batch, device=device)
            ment = _mentropy_from_probs(probs, labels).detach().cpu().numpy()
            ytrue = labels.detach().cpu().numpy()
            m_list.append(ment)
            y_list.append(ytrue)
        return np.concatenate(m_list), np.concatenate(y_list)

    m_in, y_in = collect_shadow(shadow_in_loader)
    m_out, y_out = collect_shadow(shadow_out_loader)

    taus, global_tau = _learn_class_thresholds(m_in, m_out, y_in, y_out, num_classes=num_classes, direction="leq")

    all_preds, all_labels = [], []

    def score_target(loader, true_mem_label: int):
        for batch in loader:
            logits, labels = _forward_logits_and_labels(model, batch, device=device)
            probs = _apply_defense_or_softmax(logits, dyna_noise=dyna_noise, hamp=hamp, model=model, batch=batch, device=device)

            ment = _mentropy_from_probs(probs, labels)
            ment_np = np.nan_to_num(ment.detach().cpu().numpy(), nan=0.0, posinf=1e6, neginf=-1e6)
            ytrue_np = labels.detach().cpu().numpy()

            pred = np.zeros_like(ment_np, dtype=int)
            for i in range(ment_np.shape[0]):
                tau = taus.get(int(ytrue_np[i]), global_tau)
                pred[i] = 1 if ment_np[i] <= tau else 0

            all_preds.extend(pred.tolist())
            all_labels.extend([true_mem_label] * len(pred))

    score_target(target_in_loader, 1)
    score_target(target_out_loader, 0)

    return compute_metrics(np.array(all_preds), np.array(all_labels))
