#!/usr/bin/env python
"""Copy the first N frames of a LAMMPS dump to a smaller file (for smoke tests).

Streams line-by-line; never loads the whole source. A frame boundary is the
line ``ITEM: TIMESTEP``. Stops after writing N frames.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--frames", type=int, default=5000)
    args = p.parse_args()

    t0 = time.time()
    frames = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.source, "r") as fin, open(args.out, "w") as fout:
        for line in fin:
            if line.startswith("ITEM: TIMESTEP"):
                frames += 1
                if frames > args.frames:
                    break
            fout.write(line)
    written = frames - 1 if frames > args.frames else frames
    print(f"Wrote {written} frames to {args.out} "
          f"({args.out.stat().st_size/1e6:.1f} MB) in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
