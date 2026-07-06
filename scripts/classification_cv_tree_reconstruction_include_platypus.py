#!/usr/bin/env python3
"""
Run the same CV to distance matrix to UPGMA pipeline as classification_cv_tree_reconstruction.py,
but **keep Ornithorhynchus anatinus** in the analysis (platypus is excluded in the default script).

Does not modify or overwrite:
  - results/classification_cv_tree_reconstruction/predicted_tree_UPGMA.nwk
  - any files under classification_cv_tree_reconstruction/

Outputs (default):
  - results/classification_cv_tree_reconstruction_with_platypus/predicted_tree_UPGMA.nwk
  - plus comparison plots, matrices, run_config.json, etc.

Run from project root:
  python3 scripts/classification_cv_tree_reconstruction_include_platypus.py

Optional: pass through the same flags as the base script (--n-folds, --n-estimators, --scheme, etc.).
Override output with --output-dir <path>.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "results" / "classification_cv_tree_reconstruction_with_platypus"
)

sys.path.insert(0, str(SCRIPTS_DIR))

import classification_cv_tree_reconstruction as cvt  # noqa: E402

# Retain platypus (base module defaults to EXCLUDE_PLATYPUS = True).
cvt.EXCLUDE_PLATYPUS = False


def main() -> None:
    argv = list(sys.argv[1:])
    if not any(a == "--output-dir" or a.startswith("--output-dir=") for a in argv):
        argv = ["--output-dir", str(DEFAULT_OUTPUT)] + argv
    sys.argv = ["classification_cv_tree_reconstruction_include_platypus.py"] + argv
    cvt.main()


if __name__ == "__main__":
    main()
