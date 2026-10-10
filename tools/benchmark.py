import argparse
import platform
import time

import cv2
import numpy as np
import torch

from _bootstrap import ROOT
from pfd_runtime.config import resolve, save_json
from pfd_runtime.modeling import load_trained


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--scope", choices=("forward", "image-to-boxes"), default="forward")
    parser.add_argument("--images", nargs="+")
    parser.add_argument("--threshold", type=float, default=0.001)
    args = parser.parse_args()
    if min(args.iterations, args.rounds) < 1 or args.warmup < 0:
        raise ValueError("Invalid benchmark repetition counts")
    if not 0 <= args.threshold <= 1:
        raise ValueError("Invalid confidence threshold")
    if args.scope == "image-to-boxes" and not args.images:
        raise ValueError("Image-to-boxes timing requires --images")
    model, config, _ = load_trained(resolve(args.checkpoint), args.device)
    model.eval()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    size = config["model"].get("image_size", 640)
    images = torch.zeros(1, 3, size, size, device=args.device)
    decoded = []
    for path in args.images or []:
        image = cv2.imdecode(np.fromfile(resolve(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot decode image: {path}")
        decoded.append(image)

    def run(index):
        if args.scope == "forward":
            return model(images)
        image = decoded[index % len(decoded)]
        height, width = image.shape[:2]
        rgb = cv2.cvtColor(cv2.resize(image, (size, size)), cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float().div_(255).unsqueeze(0).to(args.device)
        output = model(tensor)
        scores, labels = output["pred_logits"][0].sigmoid().max(-1)
        indices = scores.argsort(descending=True)[:100]
        indices = indices[scores[indices] >= args.threshold]
        boxes = output["pred_boxes"][0, indices]
        corners = torch.cat((boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2), -1)
        corners = corners * corners.new_tensor([width, height, width, height])
        return torch.cat((corners, scores[indices, None], labels[indices, None]), -1).cpu().numpy()
    sync = lambda: torch.cuda.synchronize(args.device) if args.device.startswith("cuda") else None
    measurements = []
    peak_allocations = []
    with torch.inference_mode():
        for _ in range(args.rounds):
            for index in range(args.warmup):
                run(index)
            sync()
            if args.device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(args.device)
            values = []
            for index in range(args.iterations):
                sync()
                start = time.perf_counter()
                run(index)
                sync()
                values.append((time.perf_counter() - start) * 1000)
            measurements.append(values)
            if args.device.startswith("cuda"):
                peak_allocations.append(torch.cuda.max_memory_allocated(args.device))
    values = np.array(measurements)
    save_json(resolve(args.output), {"scope": args.scope, "device": args.device,
              "decoded_images": len(decoded), "disk_decode_included": False,
              "threshold": args.threshold if args.scope == "image-to-boxes" else None,
              "hardware": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else platform.processor(),
              "precision": "FP32", "batch_size": 1, "input_size": size, "warmup": args.warmup,
              "iterations": args.iterations, "rounds": args.rounds, "mean_ms": float(values.mean()),
              "p50_ms": float(np.median(values)), "p95_ms": float(np.quantile(values, 0.95)),
              "sequential_fps": float(1000 / values.mean()), "round_means_ms": values.mean(1).tolist(),
              "peak_allocated_bytes": max(peak_allocations) if peak_allocations else None,
              "peak_allocated_MiB": max(peak_allocations) / 1024 ** 2 if peak_allocations else None,
              "round_peak_allocated_bytes": peak_allocations if peak_allocations else None,
              "memory_scope": "CUDA peak allocated bytes after warmup; includes resident model, inputs, and measured-forward allocations; excludes reserved allocator memory"})


if __name__ == "__main__":
    main()
