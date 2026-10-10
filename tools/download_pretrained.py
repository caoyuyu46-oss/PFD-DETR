import argparse
import hashlib
import urllib.request

from _bootstrap import ROOT
from pfd_runtime.config import resolve, save_json

URL = "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth"
SHA256 = "3ba8b5c909c9a1c4f21e96d0a7251ab1a485093955ca327d0061fef8d33c66f0"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="weights/rtdetr_r18vd_6x_coco.pth")
    args = parser.parse_args()
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    temporary = output.with_suffix(".download")
    urllib.request.urlretrieve(URL, temporary)
    actual = hashlib.sha256(temporary.read_bytes()).hexdigest()
    if actual != SHA256:
        raise ValueError(f"Checkpoint SHA-256 mismatch: {actual}")
    temporary.replace(output)
    save_json(output.with_suffix(".json"), {"url": URL, "sha256": actual, "filename": output.name})
    print(output)


if __name__ == "__main__":
    main()
