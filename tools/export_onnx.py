import argparse

import onnx
import torch

from _bootstrap import ROOT
from pfd_runtime.config import resolve
from pfd_runtime.modeling import load_trained


class ExportModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        output = self.model(images)
        return output["pred_logits"], output["pred_boxes"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    model, config, _ = load_trained(resolve(args.checkpoint))
    model.eval()
    size = config["model"].get("image_size", 640)
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(ExportModel(model), torch.zeros(1, 3, size, size), str(output), opset_version=17,
                      input_names=["images"], output_names=["pred_logits", "pred_boxes"], do_constant_folding=True)
    onnx.checker.check_model(str(output), full_check=True)
    print(output)


if __name__ == "__main__":
    main()
