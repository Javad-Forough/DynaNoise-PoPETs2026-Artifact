from __future__ import annotations

import os
import csv
import json
import time
import hashlib
import random
import argparse
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Subset

import torchvision
import torchvision.transforms as transforms

from models import get_model
from train_eval import train_one_epoch, eval_model

from membership_inference import (
    compute_confidence_attack_metrics,
    compute_loss_attack_metrics,
    compute_shadow_attack_metrics,
    compute_lira_attack_metrics,
    compute_entropy_attack_metrics_sm21,
    compute_mentropy_attack_metrics_sm21,
    train_shadow_model,  
)

from dyna_noise import DynaNoise
from hamp import HAMP


def set_all_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_num_classes(_: str = "cifar10") -> int:
    return 10



def get_fixed_cifar_loader(
    model_name: str,
    batch_size: int = 64,
    train: bool = True,
    seed: int = 42,
    num_workers: int = 2,
    data_dir: str = "data",
    shuffle: bool = True,
) -> DataLoader:
    tfms = []
    if model_name.lower() in ["alexnet", "resnet18", "vgg16_bn"]:
        tfms.append(transforms.Resize(224))
    tfms += [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ]
    ds = torchvision.datasets.CIFAR10(
        root=data_dir, train=train, download=True, transform=transforms.Compose(tfms)
    )
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, generator=g)


def get_train_test_loaders_once(
    model_name: str,
    batch_size: int,
    seed: int,
    data_dir: str,
) -> Tuple[DataLoader, DataLoader]:
    train_loader = get_fixed_cifar_loader(model_name, batch_size=batch_size, train=True, seed=seed, data_dir=data_dir, shuffle=True)
    test_loader = get_fixed_cifar_loader(model_name, batch_size=batch_size, train=False, seed=seed, data_dir=data_dir, shuffle=False)
    return train_loader, test_loader


def _subset_indices(subset_obj) -> List[int]:
    if isinstance(subset_obj, Subset):
        return list(map(int, subset_obj.indices))
    return list(range(len(subset_obj)))


def _hash_splits(*subsets) -> str:
    h = hashlib.sha1()
    for s in subsets:
        idx = _subset_indices(s)
        h.update(np.asarray(idx, dtype=np.int64).tobytes())
    return h.hexdigest()[:12]


def build_membership_splits(
    train_dataset,
    batch_size: int,
    seed: int,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader, Dict[str, int], str]:
    g_main = torch.Generator().manual_seed(seed)
    n_full = len(train_dataset)
    target_sz = int(0.7 * n_full)
    shadow_sz = n_full - target_sz
    target_ds, shadow_ds = random_split(train_dataset, [target_sz, shadow_sz], generator=g_main)

    g_t = torch.Generator().manual_seed(seed + 1)
    nt = len(target_ds)
    tin_sz = int(0.8 * nt)
    tout_sz = nt - tin_sz
    tin_ds, tout_ds = random_split(target_ds, [tin_sz, tout_sz], generator=g_t)

    g_s = torch.Generator().manual_seed(seed + 2)
    ns = len(shadow_ds)
    sin_sz = ns // 2
    sout_sz = ns - sin_sz
    sin_ds, sout_ds = random_split(shadow_ds, [sin_sz, sout_sz], generator=g_s)

    split_hash = _hash_splits(tin_ds, tout_ds, sin_ds, sout_ds)

    def _dl(subset, shuffle=True):
        return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    sizes = {
        "full_size": n_full,
        "tin_size": len(tin_ds),
        "tout_size": len(tout_ds),
        "sin_size": len(sin_ds),
        "sout_size": len(sout_ds),
    }
    return _dl(tin_ds, True), _dl(tout_ds, True), _dl(sin_ds, True), _dl(sout_ds, True), sizes, split_hash


def _clone_loader(base: DataLoader, *, dataset, shuffle: bool = False) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=base.batch_size,
        shuffle=shuffle,
        num_workers=base.num_workers,
    )


def make_balanced_attack_eval_loaders(
    in_loader: DataLoader,
    out_loader: DataLoader,
    *,
    seed: int,
    split_hash: str,
) -> Tuple[DataLoader, DataLoader]:
    n_in = len(in_loader.dataset)
    n_out = len(out_loader.dataset)
    if n_in == 0 or n_out == 0:
        return in_loader, out_loader
    n = min(n_in, n_out)
    sub_seed = int(hashlib.sha256(f"{seed}|{split_hash}|attack_eval_balance".encode()).hexdigest()[:8], 16)
    g = torch.Generator().manual_seed(sub_seed)

    def _subsample_dataset(ds, n_keep: int):
        if len(ds) <= n_keep:
            return ds
        idx = torch.randperm(len(ds), generator=g)[:n_keep].tolist()
        return Subset(ds, idx)

    in_ds_bal = _subsample_dataset(in_loader.dataset, n)
    out_ds_bal = _subsample_dataset(out_loader.dataset, n)
    return (
        _clone_loader(in_loader, dataset=in_ds_bal, shuffle=False),
        _clone_loader(out_loader, dataset=out_ds_bal, shuffle=False),
    )



def maybe_load_or_train_target(
    *,
    model_name: str,
    epochs: int,
    device: str,
    train_loader: DataLoader,
    save_dir: str,
    split_hash: str,
    explicit_ckpt: Optional[str],
) -> Tuple[torch.nn.Module, str, bool]:
    num_classes = get_num_classes()
    os.makedirs(save_dir, exist_ok=True)

    candidates: List[str] = []
    if explicit_ckpt:
        candidates.append(explicit_ckpt)
    candidates.append(os.path.join(save_dir, f"{model_name}_cifar10_target.pt"))
    candidates.append(os.path.join(save_dir, f"{model_name}_cifar10_{epochs}epochs_tin_{split_hash}.pt"))

    ckpt = None
    for p in candidates:
        if p and os.path.exists(p):
            ckpt = p
            break

    model = get_model(model_name, num_classes).to(device)
    if ckpt is not None:
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"[TARGET] Loaded: {ckpt}", flush=True)
        return model.eval(), ckpt, True

    ckpt = os.path.join(save_dir, f"{model_name}_cifar10_{epochs}epochs_tin_{split_hash}.pt")
    print(f"[TARGET] Training target on tin for {epochs} epochs (fallback) ...", flush=True)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    model.train()
    for ep in range(1, epochs + 1):
        loss, acc = train_one_epoch(model, train_loader, opt, device=device)
        if (ep % 2 == 0) or (ep == 1) or (ep == epochs):
            print(f"[TARGET][ep {ep:03d}] loss={loss:.4f} acc={acc:.4f}", flush=True)
    torch.save(model.state_dict(), ckpt)
    print(f"[TARGET] Saved: {ckpt}", flush=True)
    return model.eval(), ckpt, False


def load_required_ckpt_model(
    *,
    model_name: str,
    ckpt_path: str,
    device: str,
) -> torch.nn.Module:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"[HAMP] Missing required checkpoint: {ckpt_path}")
    m = get_model(model_name, get_num_classes()).to(device)
    m.load_state_dict(torch.load(ckpt_path, map_location=device), strict=True)
    m.eval()
    print(f"[HAMP] Loaded: {ckpt_path}", flush=True)
    return m


@dataclass
class DefenseArtifacts:
    model: torch.nn.Module
    dyna_like: Optional[object]
    hamp_like: Optional[object]
    defense_name: str
    extra: Dict[str, Any]
    test_acc_after: float


def eval_model_with_dyna_like_probs(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: str,
    dyna_like,
) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device).long()
            logits = model(x)
            probs = dyna_like.forward(logits)
            pred = probs.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.size(0))
    return float(correct) / max(1, total)


def eval_model_with_hamp_probs(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: str,
    hamp_def: HAMP,
) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device).long()
            probs = hamp_def.modify_output_probs(model, x)
            pred = probs.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.size(0))
    return float(correct) / max(1, total)


def run_defense_none(args, device, baseline_model, test_loader) -> DefenseArtifacts:
    test_acc = float(eval_model(baseline_model, test_loader, device=device))
    return DefenseArtifacts(baseline_model.eval(), None, None, "none", {}, test_acc)


def run_defense_dyna(args, device, baseline_model, test_loader) -> DefenseArtifacts:
    dyna = DynaNoise(
        base_variance=args.bv,
        lambda_scale=args.ls,
        temperature=args.t,
        ensemble_size=args.ensemble_size,
    )
    test_acc = eval_model_with_dyna_like_probs(baseline_model, test_loader, device, dyna)
    return DefenseArtifacts(
        baseline_model.eval(),
        dyna,
        None,
        "dyna",
        {"base_variance": args.bv, "lambda_scale": args.ls, "temperature": args.t, "ensemble_size": args.ensemble_size},
        float(test_acc),
    )


def run_defense_hamp(args, device, test_loader) -> DefenseArtifacts:
    # load-only hamp target model + defense object
    hamp_ckpt = args.hamp_ckpt or os.path.join(args.saved_models_dir, f"hamp_{args.model}_cifar10.pt")
    hamp_target = load_required_ckpt_model(model_name=args.model, ckpt_path=hamp_ckpt, device=device)

    hamp_def = HAMP(
        gamma=args.hamp_gamma,
        alpha=args.hamp_alpha,
        num_classes=get_num_classes(),
        cache_random_inputs=False,
    )
    test_acc = eval_model_with_hamp_probs(hamp_target, test_loader, device, hamp_def)
    return DefenseArtifacts(
        hamp_target,
        None,
        hamp_def,
        "hamp",
        {"hamp_ckpt": hamp_ckpt, "hamp_gamma": args.hamp_gamma, "hamp_alpha": args.hamp_alpha},
        float(test_acc),
    )



ATTACK_KEYS = ["conf", "loss", "shadow", "lira", "entropy", "mentropy"]


def _call_with_optional_kwargs(fn, kwargs: Dict[str, Any]):
    try:
        return fn(**kwargs)
    except TypeError:
        for drop in ["split_hash", "seed", "shadow_model_path", "shadow_models_dir", "load_only",
                     "lira_shadow_models_dir", "lira_filename_template"]:
            if drop in kwargs:
                k2 = dict(kwargs)
                k2.pop(drop, None)
                try:
                    return fn(**k2)
                except TypeError:
                    kwargs = k2
        return fn(**kwargs)


def run_selected_attacks(
    *,
    attack: str,
    model: torch.nn.Module,
    in_loader: DataLoader,
    out_loader: DataLoader,
    sin_loader: DataLoader,
    sout_loader: DataLoader,
    device: str,
    model_name: str,
    seed: int,
    split_hash: str,
    dyna_like=None,
    hamp_like=None,
    shadow_epochs: int = 30,
    lira_shadow_epochs: int = 30,
    lira_num_shadows: int = 3,
    sm21_shadow_epochs: int = 30,
    calibrate_thresholds: bool = False,
) -> Dict[str, Optional[Dict[str, Any]]]:
    attack = attack.lower()
    want_all = (attack == "all")
    results: Dict[str, Optional[Dict[str, Any]]] = {k: None for k in ATTACK_KEYS}

    in_eval_loader, out_eval_loader = make_balanced_attack_eval_loaders(in_loader, out_loader, seed=seed, split_hash=split_hash)

    conf_tau = 0.9
    loss_tau = 0.5

    if calibrate_thresholds and (want_all or attack in ["conf", "loss"]):
        _ = train_shadow_model(
            loader_in=sin_loader,
            loader_out=sout_loader,
            num_classes=get_num_classes(),
            epochs=max(1, int(shadow_epochs)),
            device=device,
            model_name=model_name,
            dataset_name="cifar10",
            seed=seed + 777,
        )
        print("[ATTACK][CALIB] Threshold calibration is disabled in artifact runner; using defaults.", flush=True)

    if want_all or attack == "conf":
        results["conf"] = compute_confidence_attack_metrics(
            model, in_eval_loader, out_eval_loader,
            threshold=conf_tau,
            dyna_noise=dyna_like,
            hamp=hamp_like,
            device=device,
        )

    if want_all or attack == "loss":
        results["loss"] = compute_loss_attack_metrics(
            model, in_eval_loader, out_eval_loader,
            threshold=loss_tau,
            dyna_noise=dyna_like,
            hamp=hamp_like,
            device=device,
        )

    if want_all or attack == "shadow":
        results["shadow"] = _call_with_optional_kwargs(compute_shadow_attack_metrics, dict(
            model=model,
            target_in_loader=in_eval_loader,
            target_out_loader=out_eval_loader,
            shadow_in_loader=sin_loader,
            shadow_out_loader=sout_loader,
            device=device,
            dyna_noise=dyna_like,
            hamp=hamp_like,
            epochs=shadow_epochs,
            model_name=model_name,
            dataset_name="cifar10",
            seed=seed,
            shadow_model_path=os.path.join("saved_models_shadow", "shadow_model.pt"),
            load_only=True,
        ))

    if want_all or attack == "lira":
        set_all_seeds(seed)
        results["lira"] = _call_with_optional_kwargs(compute_lira_attack_metrics, dict(
            target_model=model,
            target_in_loader=in_eval_loader,
            target_out_loader=out_eval_loader,
            shadow_in_loader=sin_loader,
            shadow_out_loader=sout_loader,
            device=device,
            dyna_noise=dyna_like,
            hamp=hamp_like,
            epochs=lira_shadow_epochs,
            num_shadows=lira_num_shadows,
            model_name=model_name,
            dataset_name="cifar10",
            seed=seed,
            split_hash=split_hash,
            lira_shadow_models_dir="saved_models_shadow",
            lira_filename_template="lira_shadow{i}.pt",
            load_only=True,
        ))

    if want_all or attack == "entropy":
        results["entropy"] = _call_with_optional_kwargs(compute_entropy_attack_metrics_sm21, dict(
            model=model,
            target_in_loader=in_eval_loader,
            target_out_loader=out_eval_loader,
            shadow_in_loader=sin_loader,
            shadow_out_loader=sout_loader,
            device=device,
            dyna_noise=dyna_like,
            hamp=hamp_like,
            shadow_epochs=sm21_shadow_epochs,
            model_name=model_name,
            dataset_name="cifar10",
            seed=seed,
            split_hash=split_hash,
            shadow_model_path=os.path.join("saved_models_shadow", "shadow_model.pt"),
            load_only=True,
        ))

    if want_all or attack == "mentropy":
        results["mentropy"] = _call_with_optional_kwargs(compute_mentropy_attack_metrics_sm21, dict(
            model=model,
            target_in_loader=in_eval_loader,
            target_out_loader=out_eval_loader,
            shadow_in_loader=sin_loader,
            shadow_out_loader=sout_loader,
            device=device,
            dyna_noise=dyna_like,
            hamp=hamp_like,
            shadow_epochs=sm21_shadow_epochs,
            model_name=model_name,
            dataset_name="cifar10",
            seed=seed,
            split_hash=split_hash,
            shadow_model_path=os.path.join("saved_models_shadow", "shadow_model.pt"),
            load_only=True,
        ))

    return results


def attack_acc(metrics: Optional[Dict[str, Any]]) -> Optional[float]:
    if metrics is None:
        return None
    return metrics.get("balanced_accuracy", metrics.get("accuracy", None))


def flatten_attack_before_after(
    before: Dict[str, Optional[Dict[str, Any]]],
    after: Dict[str, Optional[Dict[str, Any]]],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in ATTACK_KEYS:
        b = before.get(k) or {}
        a = after.get(k) or {}
        out[f"{k}_acc_before"] = b.get("accuracy", None)
        out[f"{k}_acc_after"] = a.get("accuracy", None)
        out[f"{k}_bacc_before"] = b.get("balanced_accuracy", None)
        out[f"{k}_bacc_after"] = a.get("balanced_accuracy", None)
    return out


def compute_midput(
    test_acc_before: float,
    test_acc_after: float,
    attacks_before: Dict[str, Optional[Dict[str, Any]]],
    attacks_after: Dict[str, Optional[Dict[str, Any]]],
    attack_scope: str,
) -> Dict[str, Any]:
    acc_drop = float(test_acc_before) - float(test_acc_after)

    if attack_scope.lower() == "all":
        keys = [k for k in ATTACK_KEYS
                if attack_acc(attacks_before.get(k)) is not None and attack_acc(attacks_after.get(k)) is not None]
    else:
        k = attack_scope.lower()
        keys = [k] if k in ATTACK_KEYS and attack_acc(attacks_before.get(k)) is not None and attack_acc(attacks_after.get(k)) is not None else []

    per_attack: Dict[str, Any] = {}
    imps: List[float] = []
    for k in keys:
        b = float(attack_acc(attacks_before[k]))
        a = float(attack_acc(attacks_after[k]))
        imp = b - a
        put = imp - acc_drop
        per_attack[f"{k}_imp"] = imp
        per_attack[f"{k}_put"] = put
        imps.append(imp)

    avg_imp = float(np.mean(imps)) if len(imps) > 0 else 0.0
    midput = avg_imp - acc_drop

    return {
        "acc_drop": acc_drop,
        "avg_attack_imp": avg_imp,
        "midput": midput,
        "midput_num_attacks": len(imps),
        **per_attack,
    }

def make_unique_run_name(args_dict: Dict[str, Any]) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = json.dumps(args_dict, sort_keys=True).encode("utf-8")
    h = hashlib.sha1(payload).hexdigest()[:10]
    return f"{ts}_{h}"


def write_single_row_csv(path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(row.keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerow(row)
    print(f"[CSV] Wrote unique run CSV: {path}", flush=True)


def append_master_csv(path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader()
            w.writerow(row)
        print(f"[CSV] Created master CSV: {path}", flush=True)
        return

    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        old_fields = list(r.fieldnames or [])
        old_rows = list(r)

    new_fields = list(dict.fromkeys(old_fields + list(row.keys())))
    needs_rewrite = (new_fields != old_fields)

    if needs_rewrite:
        for rr in old_rows:
            for k in new_fields:
                rr.setdefault(k, "")
        row_filled = {k: row.get(k, "") for k in new_fields}
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=new_fields)
            w.writeheader()
            w.writerows(old_rows)
            w.writerow(row_filled)
        print(f"[CSV] Rewrote master CSV with expanded columns + appended: {path}", flush=True)
    else:
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=old_fields)
            row_filled = {k: row.get(k, "") for k in old_fields}
            w.writerow(row_filled)
        print(f"[CSV] Appended to master CSV: {path}", flush=True)


def fmt(x: Optional[float], nd: int = 4) -> str:
    if x is None:
        return "NA"
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def print_before_after_block(title: str, before: Optional[float], after: Optional[float], nd: int = 4) -> None:
    print(f"{title:<28}  before={fmt(before, nd)}   after={fmt(after, nd)}", flush=True)



def main():
    p = argparse.ArgumentParser("Artifact runner (CIFAR-10 only; defenses: none/dyna/hamp).")

    p.add_argument("--dataset", default="cifar10", choices=["cifar10"])
    p.add_argument("--model", default="alexnet", choices=["alexnet", "resnet18", "vgg16_bn"])
    p.add_argument("--defense", required=True, choices=["none", "dyna", "hamp"])
    p.add_argument("--attack", required=True, choices=["conf", "loss", "shadow", "lira", "entropy", "mentropy", "all"])

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--data-dir", type=str, default="data")

    p.add_argument("--saved-models-dir", type=str, default="saved_models")
    p.add_argument("--target-ckpt", type=str, default=None)

    # Dyna params
    p.add_argument("--bv", type=float, default=0.3)
    p.add_argument("--ls", type=float, default=2.0)
    p.add_argument("--t", type=float, default=10.0)
    p.add_argument("--ensemble-size", type=int, default=1)

    # HAMP params (load-only)
    p.add_argument("--hamp-ckpt", type=str, default=None)
    p.add_argument("--hamp-gamma", type=float, default=0.95)
    p.add_argument("--hamp-alpha", type=float, default=0.001)

    # Attacks
    p.add_argument("--shadow-epochs", type=int, default=30)
    p.add_argument("--lira-shadow-epochs", type=int, default=30)
    p.add_argument("--lira-num-shadows", type=int, default=3)
    p.add_argument("--sm21-shadow-epochs", type=int, default=30)
    p.add_argument("--calibrate-thresholds", dest="calibrate_thresholds", action="store_true")

    # Output
    p.add_argument("--results-dir", type=str, default="results_artifact")
    p.add_argument("--master-name", type=str, default="master.csv")

    args = p.parse_args()

    set_all_seeds(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[RUN] device={device} seed={args.seed}", flush=True)

    os.makedirs(args.results_dir, exist_ok=True)

    train_loader, test_loader = get_train_test_loaders_once(args.model, args.batch_size, args.seed, args.data_dir)
    print("[DATA] CIFAR-10 train/test loaders ready.", flush=True)

    in_loader, out_loader, sin_loader, sout_loader, sizes, split_hash = build_membership_splits(
        train_dataset=train_loader.dataset,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=2,
    )
    print(f"[SPLIT] {sizes} split_hash={split_hash}", flush=True)

    # Baseline target
    baseline_model, target_ckpt_used, target_loaded = maybe_load_or_train_target(
        model_name=args.model,
        epochs=args.epochs,
        device=device,
        train_loader=in_loader,
        save_dir=args.saved_models_dir,
        split_hash=split_hash,
        explicit_ckpt=args.target_ckpt,
    )

    test_acc_before = float(eval_model(baseline_model, test_loader, device=device))
    print(f"[BASELINE] test_acc_before={test_acc_before:.4f}", flush=True)

    attacks_before = run_selected_attacks(
        attack=args.attack,
        model=baseline_model,
        in_loader=in_loader,
        out_loader=out_loader,
        sin_loader=sin_loader,
        sout_loader=sout_loader,
        device=device,
        model_name=args.model,
        seed=args.seed,
        split_hash=split_hash,
        dyna_like=None,
        hamp_like=None,
        shadow_epochs=args.shadow_epochs,
        lira_shadow_epochs=args.lira_shadow_epochs,
        lira_num_shadows=args.lira_num_shadows,
        sm21_shadow_epochs=args.sm21_shadow_epochs,
        calibrate_thresholds=args.calibrate_thresholds,
    )

    # Defense
    if args.defense == "none":
        defense_art = run_defense_none(args, device, baseline_model, test_loader)
    elif args.defense == "dyna":
        defense_art = run_defense_dyna(args, device, baseline_model, test_loader)
    elif args.defense == "hamp":
        defense_art = run_defense_hamp(args, device, test_loader)
    else:
        raise ValueError(f"Unknown defense: {args.defense}")

    test_acc_after = float(defense_art.test_acc_after)
    print(f"[DEFENSE] {defense_art.defense_name} test_acc_after={test_acc_after:.4f}", flush=True)

    attacks_after = run_selected_attacks(
        attack=args.attack,
        model=defense_art.model,
        in_loader=in_loader,
        out_loader=out_loader,
        sin_loader=sin_loader,
        sout_loader=sout_loader,
        device=device,
        model_name=args.model,
        seed=args.seed,
        split_hash=split_hash,
        dyna_like=defense_art.dyna_like,
        hamp_like=defense_art.hamp_like,
        shadow_epochs=args.shadow_epochs,
        lira_shadow_epochs=args.lira_shadow_epochs,
        lira_num_shadows=args.lira_num_shadows,
        sm21_shadow_epochs=args.sm21_shadow_epochs,
        calibrate_thresholds=args.calibrate_thresholds,
    )

    midput_info = compute_midput(
        test_acc_before=test_acc_before,
        test_acc_after=test_acc_after,
        attacks_before=attacks_before,
        attacks_after=attacks_after,
        attack_scope=args.attack,
    )

    # Report
    print("\n==================== REPORT (BEFORE vs AFTER) ====================", flush=True)
    print_before_after_block("Target test accuracy", test_acc_before, test_acc_after)

    for k in ATTACK_KEYS:
        mb = attacks_before.get(k)
        ma = attacks_after.get(k)
        if mb is None and ma is None:
            continue
        print_before_after_block(f"Attack acc ({k})", mb.get("accuracy") if mb else None, ma.get("accuracy") if ma else None)
        # print_before_after_block(f"Attack bAcc ({k})", mb.get("balanced_accuracy") if mb else None, ma.get("balanced_accuracy") if ma else None)

    print("\n---------------------- MIDPUT / PUT ----------------------", flush=True)
    print(f"acc_drop: {fmt(midput_info.get('acc_drop'))}", flush=True)
    print(f"avg_attack_imp: {fmt(midput_info.get('avg_attack_imp'))}  (over {midput_info.get('midput_num_attacks', 0)} attacks)", flush=True)
    print(f"MIDPUT: {fmt(midput_info.get('midput'))}", flush=True)
    for k in ATTACK_KEYS:
        if f"{k}_put" in midput_info:
            print(f"PUT[{k}]: {fmt(midput_info.get(f'{k}_put'))}   (imp={fmt(midput_info.get(f'{k}_imp'))})", flush=True)

    # CSV
    args_dict = vars(args).copy()
    run_id = make_unique_run_name(args_dict)
    flat_attacks = flatten_attack_before_after(attacks_before, attacks_after)

    row: Dict[str, Any] = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset": "cifar10",
        "model": args.model,
        "defense": defense_art.defense_name,
        "attack_scope": args.attack,
        "seed": args.seed,
        "split_hash": split_hash,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        **sizes,
        "test_acc_before": test_acc_before,
        "test_acc_after": test_acc_after,
        **flat_attacks,
        **midput_info,
        **defense_art.extra,
        "target_ckpt_used": target_ckpt_used,
        "target_loaded": bool(target_loaded),
    }

    unique_name = f"{run_id}__cifar10__{args.model}__{defense_art.defense_name}__{args.attack}.csv"
    unique_path = os.path.join(args.results_dir, unique_name)
    write_single_row_csv(unique_path, row)

    master_path = os.path.join(args.results_dir, args.master_name)
    append_master_csv(master_path, row)


if __name__ == "__main__":
    main()
