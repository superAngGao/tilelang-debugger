"""Resolve fixture intents to user-visible original source lines."""
import argparse
import json
from pathlib import Path


def config(local=False, thread=None):
    source = Path(__file__).with_name("kernel.py")
    lines = source.read_text().splitlines()
    targets = {"local_buf": "local_buf[1] =", "flag": "flag =", "wide": "wide =", "f16": "f16 =", "bf16": "bf16 =", "f32": "f32 ="} if local else {
        "value": "value =", "other": "other =", "odd": "odd =", "empty": "empty =", "inactive": "inactive =", "tail": "tail ="}
    points = []
    for name, marker in targets.items():
        line = next(i for i, text in enumerate(lines, 1) if text.strip().startswith(marker))
        points.append(dict(id=name, line=line, when="after", buffer=name, block=[0, 0, 0], loops=[], thread=thread))
    return dict(schema=2, source=str(source.resolve()), points=points)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--thread", type=int)
    args = parser.parse_args()
    Path(args.output).write_text(json.dumps(config(args.local, args.thread), indent=2) + "\n")
