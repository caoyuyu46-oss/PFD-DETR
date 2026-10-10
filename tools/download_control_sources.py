import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

from _bootstrap import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", choices=("WTConv", "FreqFusion", "all"), default="all")
    args = parser.parse_args()
    locks = json.loads((ROOT / "configs/control_sources.json").read_text(encoding="utf-8"))
    for lock in locks:
        if args.control not in ("all", lock["name"]):
            continue
        for item in lock["files"]:
            destination = ROOT / "external_sources" / lock["name"] / item["path"]
            if destination.is_file() and hashlib.sha256(destination.read_bytes()).hexdigest() == item["sha256"]:
                print(f"Verified {destination.relative_to(ROOT)}")
                continue
            url = lock["repository"].replace("https://github.com/", "https://raw.githubusercontent.com/")
            url += f"/{lock['commit']}/{item['path']}"
            request = urllib.request.Request(url, headers={"User-Agent": "PFD-DETR-source-fetcher"})
            with urllib.request.urlopen(request, timeout=60) as response:
                content = response.read()
            if hashlib.sha256(content).hexdigest() != item["sha256"]:
                raise ValueError(f"Downloaded source hash mismatch: {url}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
            print(f"Downloaded {destination.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
