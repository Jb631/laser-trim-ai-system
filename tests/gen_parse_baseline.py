"""Regenerate the per-model parse baseline fixture.

Run after an INTENTIONAL parser change to refresh the recorded state:

    python tests/gen_parse_baseline.py

The committed baseline (tests/fixtures/parse_baseline.json) is what
test_parse_all_models.py asserts against. Regenerating it is a deliberate act:
review the diff before committing so a real regression can't be rubber-stamped.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model_sweep import classify, discover_representatives  # noqa: E402

OUT = Path(__file__).resolve().parent / "fixtures" / "parse_baseline.json"


def main() -> None:
    reps = discover_representatives()
    baseline = {}
    counts = {"ok": 0, "empty": 0, "error": 0}
    for ftype, model, relpath in reps:
        res = classify(ftype, relpath)
        baseline[relpath] = {"type": ftype, "model": model, **res}
        counts[res["status"]] = counts.get(res["status"], 0) + 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(baseline, indent=1, sort_keys=True))
    print(f"Wrote {len(baseline)} entries to {OUT.relative_to(Path.cwd())}")
    print("Status counts:", counts)


if __name__ == "__main__":
    main()
