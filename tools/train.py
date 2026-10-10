import argparse

from _bootstrap import ROOT
from pfd_runtime.config import load_config, resolve
from pfd_runtime.engine import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume")
    args = parser.parse_args()
    config = load_config(resolve(args.config))
    train(config, resolve(args.output), args.seed, args.device, args.workers, args.max_steps, args.resume)


if __name__ == "__main__":
    main()
