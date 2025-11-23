# src/training/train.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import f1_score, accuracy_score

from src.data.datasets import build_dataloaders, EMOTION_LABELS
from src.models.main_cnn import MainCNN
from src.training.losses import build_loss
from src.training.schedulers import build_scheduler

logger = logging.getLogger("train")


def _count_class_frequencies(dataset) -> List[int]:
    """
    Count label frequencies from a RAFDBDataset instance.
    Assumes each sample is (img, label_idx).
    """
    counts = np.zeros(len(EMOTION_LABELS), dtype=np.int64)
    for _, label in dataset:
        counts[int(label)] += 1
    return counts.tolist()


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []

    for step, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = logits.argmax(dim=1).detach().cpu().numpy()
        targets = labels.detach().cpu().numpy()
        all_preds.extend(preds.tolist())
        all_targets.extend(targets.tolist())

    epoch_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro")

    return {
        "loss": float(epoch_loss),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
    }


def evaluate(
    model: nn.Module,
    loader,
    criterion: Optional[nn.Module],
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    running_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            if criterion is not None:
                loss = criterion(logits, labels)
                running_loss += loss.item() * images.size(0)

            preds = logits.argmax(dim=1).detach().cpu().numpy()
            targets = labels.detach().cpu().numpy()
            all_preds.extend(preds.tolist())
            all_targets.extend(targets.tolist())

    if len(all_targets) == 0:
        return {"loss": 0.0, "accuracy": 0.0, "macro_f1": 0.0}

    avg_loss = (
        running_loss / len(loader.dataset) if criterion is not None else 0.0
    )
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro")

    # per-class F1 (useful later)
    per_class_f1 = f1_score(
        all_targets,
        all_preds,
        average=None,
        labels=list(range(len(EMOTION_LABELS))),
        zero_division=0,
    ).tolist()

    return {
        "loss": float(avg_loss),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "per_class_f1": per_class_f1,
    }


def train_model(cfg: Dict[str, Any]) -> None:
    """
    High-level training function used by main_train.py.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ---------- Data ----------
    loaders = build_dataloaders(cfg)
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    # Count class frequencies for weighted loss
    class_counts = _count_class_frequencies(train_loader.dataset)
    logger.info(f"Class counts: {class_counts}")

    num_classes = len(EMOTION_LABELS)

    # ---------- Model ----------
    in_channels = int(cfg["dataset"].get("channels", 1))
    model = MainCNN(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=cfg.get("model", {}).get("base_channels", 32),
        num_blocks=cfg.get("model", {}).get("num_blocks", 4),
        use_se=cfg.get("model", {}).get("use_se", True),
        activation=cfg.get("model", {}).get("activation", "elu"),
        dropout_p=cfg.get("model", {}).get("dropout_p", 0.25),
    )
    model.to(device)

    # ---------- Loss ----------
    loss_cfg = cfg.get("loss", None)
    criterion = build_loss(
        num_classes=num_classes,
        loss_cfg=loss_cfg,
        class_counts=class_counts,
    )

    # ---------- Optimizer ----------
    opt_cfg = cfg.get("optimizer", {})
    opt_name = opt_cfg.get("name", "adamw").lower()
    lr = float(opt_cfg.get("lr", 1e-3))
    weight_decay = float(opt_cfg.get("weight_decay", 1e-4))
    momentum = float(opt_cfg.get("momentum", 0.9))

    if opt_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    else:  # "adamw" or default
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    num_epochs = int(cfg["training"]["num_epochs"])
    steps_per_epoch = max(1, len(train_loader))
    scheduler_cfg = cfg.get("scheduler", None)
    scheduler = build_scheduler(optimizer, scheduler_cfg, num_epochs, steps_per_epoch)

    # ---------- Logging ----------
    log_cfg = cfg.get("logging", {})
    log_dir = Path(log_cfg.get("log_dir", "experiments/logs"))
    ckpt_dir = Path(log_cfg.get("checkpoint_dir", "experiments/checkpoints"))
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    use_tb = bool(log_cfg.get("tensorboard", True))
    writer = None
    if use_tb:
        writer = SummaryWriter(log_dir=str(log_dir))

    early_stopping_patience = int(
        cfg["training"].get("early_stopping_patience", 15)
    )
    best_metric_name = log_cfg.get("save_best_metric", "val_macro_f1")

    best_val_metric = -float("inf")
    best_epoch = -1
    epochs_no_improve = 0

    # ---------- Training loop ----------
    for epoch in range(1, num_epochs + 1):
        logger.info(f"Epoch {epoch}/{num_epochs}")

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
        )
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        logger.info(
            f"Train - loss: {train_metrics['loss']:.4f}, "
            f"acc: {train_metrics['accuracy']:.4f}, "
            f"macro_f1: {train_metrics['macro_f1']:.4f}"
        )
        logger.info(
            f"Val   - loss: {val_metrics['loss']:.4f}, "
            f"acc: {val_metrics['accuracy']:.4f}, "
            f"macro_f1: {val_metrics['macro_f1']:.4f}"
        )

        # TensorBoard
        if writer is not None:
            writer.add_scalar("train/loss", train_metrics["loss"], epoch)
            writer.add_scalar("train/accuracy", train_metrics["accuracy"], epoch)
            writer.add_scalar("train/macro_f1", train_metrics["macro_f1"], epoch)
            writer.add_scalar("val/loss", val_metrics["loss"], epoch)
            writer.add_scalar("val/accuracy", val_metrics["accuracy"], epoch)
            writer.add_scalar("val/macro_f1", val_metrics["macro_f1"], epoch)

        # Scheduler step
        if scheduler is not None:
            from torch.optim.lr_scheduler import ReduceLROnPlateau

            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_metrics["macro_f1"])
            else:
                scheduler.step()

        # Early stopping logic
        current_metric = val_metrics.get("macro_f1", 0.0)
        if best_metric_name == "val_accuracy":
            current_metric = val_metrics.get("accuracy", 0.0)

        if current_metric > best_val_metric:
            best_val_metric = current_metric
            best_epoch = epoch
            epochs_no_improve = 0

            ckpt_path = ckpt_dir / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_metric": best_val_metric,
                    "config": cfg,
                },
                ckpt_path,
            )
            logger.info(
                f"New best model (metric={best_val_metric:.4f}) saved to {ckpt_path}"
            )
        else:
            epochs_no_improve += 1
            logger.info(
                f"No improvement for {epochs_no_improve} epoch(s) "
                f"(best={best_val_metric:.4f} at epoch {best_epoch})"
            )

        if epochs_no_improve >= early_stopping_patience:
            logger.info(
                f"Early stopping triggered after {epoch} epochs "
                f"(best epoch = {best_epoch})"
            )
            break

    if writer is not None:
        writer.close()

    logger.info(
        f"Training finished. Best {best_metric_name}={best_val_metric:.4f} at epoch {best_epoch}"
    )
