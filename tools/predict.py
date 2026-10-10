import argparse

import cv2
import numpy as np
import torch

from _bootstrap import ROOT
from pfd_runtime.config import resolve, save_json
from pfd_runtime.modeling import load_trained


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        raise ValueError("Threshold must lie in [0, 1]")
    model, config, checkpoint = load_trained(resolve(args.checkpoint), args.device)
    image = cv2.imdecode(np.fromfile(resolve(args.image), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Cannot decode image")
    height, width = image.shape[:2]
    size = config["model"].get("image_size", 640)
    rgb = cv2.cvtColor(cv2.resize(image, (size, size)), cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float().div(255).unsqueeze(0).to(args.device)
    with torch.inference_mode():
        result = model.eval()(tensor)
    scores = result["pred_logits"][0].sigmoid().amax(-1)
    rows = []
    for index in scores.argsort(descending=True)[:100]:
        score = float(scores[index])
        if score < args.threshold:
            continue
        cx, cy, bw, bh = result["pred_boxes"][0, index].cpu().tolist()
        box = [(cx - bw / 2) * width, (cy - bh / 2) * height, bw * width, bh * height]
        rows.append({"category_id": 1, "score": score, "bbox": box})
        x, y, w, h = box
        cv2.rectangle(image, (round(x), round(y)), (round(x + w), round(y + h)), (0, 180, 255), 2)
        cv2.putText(image, f"fruit {score:.2f}", (round(x), max(round(y) - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 180, 255), 1)
    destination = resolve(args.output)
    if destination.suffix.lower() != ".png":
        raise ValueError("Output image must use the .png extension")
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", image)[1].tofile(destination)
    save_json(destination.with_suffix(".json"), {"image": resolve(args.image).name, "width": width, "height": height,
              "threshold": args.threshold, "detections": rows, "visible_count": len(rows),
              "checkpoint": resolve(args.checkpoint).name, "seed": checkpoint["seed"]})


if __name__ == "__main__":
    main()
