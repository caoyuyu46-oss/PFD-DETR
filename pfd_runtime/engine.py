# Paper optimizer grouping adapted from Ultralytics 8.0.201, AGPL-3.0.
# Modified for the PFD-DETR distribution; see NOTICE and LICENSE.
from __future__ import annotations

import copy
import json
import math
import platform
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import digest, save_json
from .data import FruitDataset, collate
from .metrics import coco_metrics, operating_metrics, select_thresholds
from .modeling import criterion, initialize, set_seed


class EMA:
    def __init__(self, model, decay=0.9999):
        self.model = copy.deepcopy(model).eval()
        self.model.requires_grad_(False)
        self.updates, self.decay = 0, decay

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        decay = self.decay * (1 - math.exp(-self.updates / 2000))
        state = model.state_dict()
        for name, value in self.model.state_dict().items():
            if value.is_floating_point():
                value.mul_(decay).add_(state[name].detach(), alpha=1 - decay)
            else:
                value.copy_(state[name])


def loader(dataset, batch_size, workers, shuffle=False, generator=None):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, collate_fn=collate,
                      generator=generator, pin_memory=torch.cuda.is_available(), persistent_workers=False)


def build_optimizer(model, training, architecture="legacy_official"):
    parameters = model.parameters()
    if architecture == "paper_v716":
        normalizations = tuple(value for name, value in vars(torch.nn).items()
                               if "Norm" in name and isinstance(value, type))
        biases, weights, normalized = [], [], []
        for module_name, module in model.named_modules():
            for parameter_name, parameter in module.named_parameters(recurse=False):
                if not parameter.requires_grad:
                    continue
                name = f"{module_name}.{parameter_name}"
                if "bias" in name:
                    biases.append(parameter)
                elif isinstance(module, normalizations):
                    normalized.append(parameter)
                else:
                    weights.append(parameter)
        parameters = [{"params": biases, "weight_decay": 0., "group_name": "bias"},
                      {"params": weights, "weight_decay": training["weight_decay"], "group_name": "weight"},
                      {"params": normalized, "weight_decay": 0., "group_name": "normalization"}]
    elif architecture != "legacy_official":
        raise ValueError("Unsupported optimizer architecture")
    return torch.optim.AdamW(parameters, lr=training["lr"], betas=tuple(training["betas"]),
                             weight_decay=training["weight_decay"])


@torch.inference_mode()
def predict(model, dataset, device, batch_size=1, workers=0, limit=None):
    was_training = model.training
    model.eval()
    predictions = []
    for batch_index, (images, targets) in enumerate(loader(dataset, batch_size, workers)):
        if limit is not None and batch_index >= limit:
            break
        output = model(images.to(device))
        scores = output["pred_logits"].sigmoid().squeeze(-1)
        boxes = output["pred_boxes"]
        for target, image_scores, image_boxes in zip(targets, scores, boxes):
            width, height = target["orig_size"].tolist()
            indices = image_scores.argsort(descending=True)[:100]
            for index in indices:
                score = float(image_scores[index])
                if score < 0.001:
                    continue
                cx, cy, w, h = image_boxes[index].tolist()
                predictions.append({"image_id": int(target["image_id"]), "category_id": dataset.category_id,
                                    "bbox": [(cx - w / 2) * width, (cy - h / 2) * height, w * width, h * height], "score": score})
    model.train(was_training)
    return predictions


def evaluate_predictions(dataset, predictions, thresholds=None):
    result = coco_metrics(dataset.coco, predictions)
    result["APS640"] = coco_metrics(dataset.coco, predictions, canvas=True)["APS"]
    result["evaluation_dataset_sha256"] = digest(dataset.coco)
    if thresholds:
        for purpose, threshold in thresholds.items():
            result[purpose] = operating_metrics(dataset.coco, predictions, threshold) if threshold is not None else None
    return result


def train(config, output, seed=0, device="cuda", workers=4, max_steps=None, resume=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "last.pth").exists() and resume is None:
        raise FileExistsError(f"Existing training output requires --resume or a new directory: {output}")
    training = config["training"]
    if training.get("warmup_epochs", 0) != 0:
        raise ValueError("This release recipe supports warmup_epochs=0 only")
    size = config["model"].get("image_size", 640)
    train_data = FruitDataset(config["data"], "train", size, config["augmentation"], seed)
    val_data = FruitDataset(config["data"], "val", size)
    if not train_data.images or not val_data.images:
        raise ValueError("Training and validation splits must be nonempty")
    init = config["initialization"]
    pretrained = init.get("checkpoint") if init["type"] == "coco" else None
    if init["type"] == "coco" and not pretrained:
        raise ValueError("COCO initialization requires a checkpoint path")
    model, provenance = initialize(config["model"], seed, pretrained)
    model.to(device)
    loss_function = criterion(config["model"]).to(device)
    optimizer = build_optimizer(model, training, config["model"].get("architecture", "legacy_official"))
    ema = EMA(model, training.get("ema_decay", 0.9999))
    first_epoch, best_ap, steps = 0, -1.0, 0
    if resume:
        previous = torch.load(resume, map_location="cpu")
        if previous.get("smoke_steps") is not None and max_steps is None:
            raise ValueError("Smoke-test checkpoints cannot resume as formal experiment runs")
        if digest(previous["config"]) != digest(config) or previous["seed"] != seed:
            raise ValueError("Resume requires the same resolved configuration and seed")
        model.load_state_dict(previous["raw_model"])
        ema.model.load_state_dict(previous["model"])
        ema.updates = previous["ema_updates"]
        optimizer.load_state_dict(previous["optimizer"])
        first_epoch, best_ap, steps = previous["epoch"] + 1, previous["best_ap"], previous["steps"]
        provenance = previous["initialization"]
        torch.set_rng_state(previous["torch_rng"])
        if device.startswith("cuda") and previous.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(previous["cuda_rng"])
    save_json(output / "config.json", config)
    save_json(output / "initialization.json", provenance)
    save_json(output / "environment.json", {"python": platform.python_version(), "torch": torch.__version__, "device": device,
                                           "cuda": torch.version.cuda, "amp": False, "tf32": False, "smoke_steps": max_steps})
    generator = torch.Generator()
    train_loader = loader(train_data, training["batch_size"], workers, True, generator)
    start = time.perf_counter()
    for epoch in range(first_epoch, training["epochs"]):
        train_data.set_epoch(epoch)
        generator.manual_seed(seed + epoch * 100003)
        fraction = epoch / max(training["epochs"] - 1, 1)
        factor = training["final_lr_ratio"] + (1 - training["final_lr_ratio"]) * (1 + math.cos(math.pi * fraction)) / 2
        for group in optimizer.param_groups:
            group["lr"] = training["lr"] * factor
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulated, losses, batches = 0, {}, 0
        for batch_index, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
            outputs = model(images, targets)
            loss_dict = loss_function(outputs, targets)
            total = sum(loss_dict.values())
            if not torch.isfinite(total):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}, batch {batch_index}")
            (total * len(images) / training["nominal_batch_size"]).backward()
            accumulated += len(images)
            batches += 1
            for key, value in loss_dict.items():
                losses[key] = losses.get(key, 0.0) + float(value.detach())
            last = batch_index + 1 == len(train_loader)
            if accumulated >= training["nominal_batch_size"] or last or max_steps is not None:
                scale = training["nominal_batch_size"] / accumulated
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(scale)
                torch.nn.utils.clip_grad_norm_(model.parameters(), training["clip_norm"])
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)
                accumulated, steps = 0, steps + 1
                if max_steps is not None and steps >= max_steps:
                    break
        predictions = predict(ema.model, val_data, device, training.get("eval_batch_size", 1), workers)
        metrics = evaluate_predictions(val_data, predictions)
        ap = metrics["AP"]
        is_best = ap is not None and ap >= best_ap
        best_ap = max(best_ap, ap if ap is not None else -1)
        state = {"config": config, "seed": seed, "epoch": epoch, "steps": steps, "best_ap": best_ap,
                 "model": ema.model.state_dict(), "raw_model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "ema_updates": ema.updates, "initialization": provenance, "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if device.startswith("cuda") else None,
                 "validation_metrics": metrics, "smoke_steps": max_steps}
        torch.save(state, output / "last.pth")
        if is_best:
            torch.save(state, output / "best.pth")
            save_json(output / "validation_predictions.json", predictions)
            save_json(output / "thresholds.json", select_thresholds(val_data.coco, predictions))
        record = {"epoch": epoch, "steps": steps, "learning_rate": optimizer.param_groups[0]["lr"],
                  "losses": {key: value / batches for key, value in losses.items()}, "validation": metrics,
                  "elapsed_seconds": time.perf_counter() - start}
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps({"epoch": epoch, "validation_AP": ap, "best_AP": best_ap, "steps": steps}), flush=True)
        if max_steps is not None and steps >= max_steps:
            break
    return output / "best.pth"
