import argparse

import cv2
import numpy as np
import torch

from _bootstrap import ROOT
from pfd_runtime.config import resolve, save_json
from pfd_runtime.modeling import load_trained


def normalize(cam):
    cam = cam - cam.min()
    return cam / (cam.max() + 1e-7)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--target-layer", default="encoder.cam_fusion")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--query-ratio", type=float, default=0.02)
    args = parser.parse_args()
    if not 0 < args.query_ratio <= 1 or not 0 <= args.score_threshold <= 1:
        raise ValueError("Invalid query ratio or score threshold")
    model, config, checkpoint = load_trained(resolve(args.checkpoint), args.device)
    model.eval().requires_grad_(True)
    image = cv2.imdecode(np.fromfile(resolve(args.image), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Cannot decode the input image")
    height, width = image.shape[:2]
    size = config["model"].get("image_size", 640)
    resized = cv2.cvtColor(cv2.resize(image, (size, size)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    tensor = torch.from_numpy(resized.transpose(2, 0, 1).copy()).unsqueeze(0).to(args.device)
    saved = {}

    def capture(module, inputs, output):
        saved["activation"] = output
        output.retain_grad()

    hook = model.get_submodule(args.target_layer).register_forward_hook(capture)
    try:
        output = model(tensor)
        scores = output["pred_logits"][0].sigmoid().amax(-1)
        selected = scores.argsort(descending=True)[:int(len(scores) * args.query_ratio)]
        selected = selected[scores[selected] >= args.score_threshold]
        if len(selected):
            scores[selected].sum().backward()
            activation = saved["activation"].detach()
            gradient = saved["activation"].grad.detach()
            squared, cubed = gradient.square(), gradient.pow(3)
            denominator = 2 * squared + activation.sum((2, 3), keepdim=True) * cubed + 1e-6
            alpha = torch.where(gradient != 0, squared / denominator, torch.zeros_like(gradient))
            weights = (alpha * gradient.relu()).sum((2, 3), keepdim=True)
            cam = (weights * activation).sum(1).relu()[0].cpu().numpy()
            cam = cv2.resize(normalize(cam), (size, size))
        else:
            cam = np.zeros((size, size), dtype=np.float32)
        boxes = output["pred_boxes"][0].detach().cpu().numpy()
        scores_array = scores.detach().cpu().numpy()
    finally:
        hook.remove()
    bounded = np.zeros_like(cam)
    visible = []
    for box, score in zip(boxes, scores_array):
        if score <= args.score_threshold:
            continue
        cx, cy, bw, bh = box * size
        x1, y1, x2, y2 = np.array([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], dtype=int)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(size - 1, x2), min(size - 1, y2)
        if x2 > x1 and y2 > y1:
            bounded[y1:y2, x1:x2] = normalize(cam[y1:y2, x1:x2].copy())
            visible.append((x1, y1, x2, y2, float(score)))
    bounded = normalize(bounded)
    colored = cv2.cvtColor(cv2.applyColorMap((bounded * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB) / 255
    overlay = (colored + resized) / 2
    overlay = np.uint8(255 * overlay / max(overlay.max(), 1e-7))
    for x1, y1, x2, y2, score in visible:
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 180, 0), 2)
        cv2.putText(overlay, f"fruit {score:.2f}", (x1, max(y1 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 180, 0), 1)
    destination = resolve(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = cv2.imencode(".png", cv2.cvtColor(cv2.resize(overlay, (width, height)), cv2.COLOR_RGB2BGR))[1]
    encoded.tofile(destination)
    np.savez_compressed(destination.with_suffix(".npz"), raw_cam=cam, box_normalized_cam=bounded)
    save_json(destination.with_suffix(".json"), {"method": "GradCAMPlusPlus", "target_layer": args.target_layer,
              "target": "sum_of_selected_class_scores", "query_ratio": args.query_ratio, "score_threshold": args.score_threshold,
              "selected_queries": len(selected), "normalization": "per_box_then_per_image", "seed": checkpoint["seed"],
              "input_size": size, "checkpoint": resolve(args.checkpoint).name})


if __name__ == "__main__":
    main()
