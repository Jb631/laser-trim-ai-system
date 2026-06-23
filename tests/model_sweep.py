"""Shared discovery + classification for the per-model parse sweep.

Used by both the baseline generator (``gen_parse_baseline.py``) and the
regression test (``test_parse_all_models.py``) so they agree on exactly which
files are checked and how each is classified.

A "representative" file is the first (sorted) sample file for each
(file_type, model) pair across the bundled sample directories. Parsing one
file per model keeps the sweep fast while still covering every model.
"""
from __future__ import annotations

import glob
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Repo root = parent of tests/
ROOT = Path(__file__).resolve().parent.parent

# Make the package importable whether or not it's pip-installed or a conftest
# adds it (main has no conftest; the work machine uses `pip install -e .`).
_SRC = str(ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Sample globs per file type. Relative to repo root.
SAMPLE_GLOBS: Dict[str, List[str]] = {
    "trim": [
        "test_files/Laser Data Samples/*.xls",
        "Work Files/Sample_Base_2026-04-10/LTS/*/*.xls",
        "Work Files/Sample_Base_2026-04-10/DLTS/*/*.xls",
    ],
    "final_test": [
        "Work Files/Sample_Base_2026-04-10/Test Station/*/*.xls",
    ],
    "smoothness": [
        "Work Files/Sample_Base_2026-04-10/Smoothness_Sample_2026-04-10/Test Station/*/*.xlsx",
    ],
}


def model_of(path: str) -> str:
    """Best-effort model number from a sample filename."""
    base = os.path.basename(path)
    return base.split("_")[0].split("-sn")[0].split("-sntest")[0]


def discover_representatives() -> List[Tuple[str, str, str]]:
    """Return sorted (file_type, model, relpath) — one rep file per (type, model)."""
    reps: Dict[Tuple[str, str], str] = {}
    for ftype, patterns in SAMPLE_GLOBS.items():
        found: List[str] = []
        for pat in patterns:
            found.extend(glob.glob(str(ROOT / pat)))
        for f in sorted(found):
            key = (ftype, model_of(f))
            reps.setdefault(key, f)
    out = []
    for (ftype, model), f in reps.items():
        out.append((ftype, model, os.path.relpath(f, ROOT)))
    return sorted(out)


def _finite(seq) -> bool:
    return all(isinstance(x, (int, float)) and math.isfinite(x) for x in seq)


def classify(ftype: str, relpath: str) -> Dict[str, Any]:
    """Parse one file and return a status dict.

    status is one of:
      - "ok":     parsed, produced non-empty aligned data
      - "empty":  parsed but produced no usable track data
      - "error":  parser raised
    Extra metric fields are included for golden-value comparison.
    """
    # Imports are local so the module is importable without src on sys.path
    # until a caller has set it up.
    from laser_trim_analyzer.core.parser import ExcelParser
    from laser_trim_analyzer.core.final_test_parser import FinalTestParser
    from laser_trim_analyzer.core.smoothness_parser import SmoothnessParser

    path = ROOT / relpath
    try:
        if ftype == "trim":
            d = ExcelParser().parse_file(path)
            tracks = d.get("tracks") or []
            if not tracks:
                return {"status": "empty"}
            ok_tracks = 0
            metrics: Dict[str, Any] = {"n_tracks": len(tracks)}
            for t in tracks:
                # Test-sweep-only tracks legitimately carry their data in
                # untrimmed_* (is_untrimmed_only) and leave positions/errors
                # empty; normal trimmed tracks carry it in positions/errors.
                # Accept either source.
                if t.get("is_untrimmed_only") or not t.get("positions"):
                    pos = t.get("untrimmed_positions") or []
                    err = t.get("untrimmed_errors") or []
                else:
                    pos = t.get("positions") or []
                    err = t.get("errors") or []
                if not pos or not err or len(pos) != len(err):
                    continue  # this track has no usable data; check the rest
                # Positions must be finite; error columns may carry NaN blanks
                # (equipment dropouts) which downstream treats as fail points.
                if not _finite(pos):
                    return {"status": "error", "reason": "non-finite positions"}
                ok_tracks += 1
                if "n_points" not in metrics:
                    ue = t.get("untrimmed_errors") or []
                    metrics["post_sigma"] = round(float(np.nanstd(err)), 6)
                    metrics["untr_n"] = len(ue)
                    metrics["untr_sigma"] = (
                        round(float(np.nanstd(ue)), 6) if ue else None
                    )
                    metrics["n_points"] = len(pos)
            if ok_tracks == 0:
                return {"status": "empty", "n_tracks": len(tracks)}
            metrics["status"] = "ok"
            return metrics

        if ftype == "final_test":
            d = FinalTestParser().parse_file(path)
            tracks = d.get("tracks") or []
            if not tracks:
                return {"status": "empty"}
            t0 = tracks[0]
            errs = t0.get("errors") or []
            if not errs:
                return {"status": "empty", "n_tracks": len(tracks)}
            return {
                "status": "ok",
                "n_tracks": len(tracks),
                "n_points": len(errs),
                "linearity_pass": bool(t0.get("linearity_pass")),
            }

        if ftype == "smoothness":
            d = SmoothnessParser().parse_file(path)
            tracks = d.get("tracks") or []
            if not tracks:
                return {"status": "empty"}
            t0 = tracks[0]
            vals = t0.get("smoothness_values") or []
            if not vals:
                return {"status": "empty"}
            return {
                "status": "ok",
                "n_points": len(vals),
                "max_smoothness": round(float(t0.get("max_smoothness") or 0.0), 6),
                "smoothness_pass": bool(t0.get("smoothness_pass")),
            }

    except Exception as e:  # noqa: BLE001 - we want to record any failure
        return {"status": "error", "reason": f"{type(e).__name__}: {e}"[:160]}

    return {"status": "error", "reason": f"unknown file type {ftype}"}
