"""Run every companion notebook top to bottom and fail if any cell (or any assert) fails.

Usage:
    python tools/check_notebooks.py                 # all notebooks
    python tools/check_notebooks.py --fast          # skip notebooks that download model weights
    python tools/check_notebooks.py notebooks/Episode\\ 07\\ -\\ Learning\\ Loss.ipynb
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parent.parent
HEAVY = ("transformers", "huggingface_hub", "from_pretrained")  # notebooks that download weights


def find_notebooks() -> list[Path]:
    eps = sorted(p for p in ROOT.glob("Episode */notebook.ipynb"))
    return eps + sorted((ROOT / "notebooks").glob("*.ipynb"))


def is_heavy(nb: nbformat.NotebookNode) -> bool:
    code = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    return any(h in code for h in HEAVY)


def run(path: Path, timeout: int) -> tuple[bool, float, str]:
    nb = nbformat.read(path, as_version=4)
    start = time.time()
    try:
        NotebookClient(nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": str(path.parent)}}).execute()
    except CellExecutionError as err:
        lines = str(err).strip().splitlines()
        return False, time.time() - start, "\n".join(lines[-6:])
    return True, time.time() - start, ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", type=Path)
    ap.add_argument("--fast", action="store_true", help="skip notebooks that download model weights")
    ap.add_argument("--timeout", type=int, default=900, help="seconds per cell")
    args = ap.parse_args()

    paths = args.paths or find_notebooks()
    failed = skipped = 0
    for p in paths:
        nb = nbformat.read(p, as_version=4)
        name = p.relative_to(ROOT) if p.is_absolute() else p
        if args.fast and is_heavy(nb):
            print(f"SKIP  {name}  (downloads model weights)")
            skipped += 1
            continue
        ok, secs, msg = run(p.resolve(), args.timeout)
        print(f"{'PASS' if ok else 'FAIL'}  {name}  ({secs:.0f} s)", flush=True)
        if not ok:
            failed += 1
            print("      " + msg.replace("\n", "\n      "))
    print(f"\n{len(paths) - failed - skipped} passed, {failed} failed, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
