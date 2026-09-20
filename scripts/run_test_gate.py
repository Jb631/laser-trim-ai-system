"""Run test files ONE PER PROCESS and report real counts from junit.

    python scripts/run_test_gate.py                      # every tests/test_*.py  (the whole suite)
    python scripts/run_test_gate.py tests/test_a.py ...  # just these

Why this exists (2026-09-20). Four ways a run lied in one day: a hand-picked list missed the test
that mattered; a zsh variable that did not word-split made pytest collect ZERO tests while a wrapper
printed GREEN; `pytest -q` on top of the `-q` already in pyproject printed no pass count at all; and
the suite "took 2.5 hours" because two GUI test files in one process could hang forever at full CPU.
So: one file per process with a hard time limit (a hang is killed and NAMED, it cannot freeze the
gate), counts read from the junit <testsuite> element, a file that collects nothing is a FAILURE,
and with no arguments it runs everything -- the gate is the whole suite, not a list someone chose.
Exit code = number of files that were not OK.
"""
import os
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LIMIT_S = 900


def main() -> int:
    files = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not files:
        files = sorted(str(p.relative_to(REPO)) for p in (REPO / "tests").glob("test_*.py"))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    bad, total, skipped, started = [], 0, 0, time.time()
    for f in files:
        fd, jx = tempfile.mkstemp(suffix=".xml")
        os.close(fd)
        os.remove(jx)
        t = time.time()
        try:
            r = subprocess.run([sys.executable, "-B", "-m", "pytest", f, "-p", "no:cacheprovider",
                                f"--junitxml={jx}"], cwd=REPO, capture_output=True, text=True,
                               timeout=LIMIT_S, env=env)
        except subprocess.TimeoutExpired:
            bad.append(f)
            print(f"HANG {time.time()-t:5.0f}s  killed after {LIMIT_S} s  {f}", flush=True)
            continue
        if not os.path.exists(jx):
            bad.append(f)
            print(f"FAIL {time.time()-t:5.0f}s  no junit written (exit {r.returncode})  {f}", flush=True)
            continue
        root = ET.parse(jx).getroot()
        os.remove(jx)
        suite = root if root.tag == "testsuite" else root[0]
        n, fl, er, sk = (int(suite.attrib.get(k, 0)) for k in ("tests", "failures", "errors", "skipped"))
        ok = n > 0 and fl == 0 and er == 0 and r.returncode == 0
        if not ok:
            bad.append(f)
        total += n
        skipped += sk
        print(f"{'OK  ' if ok else 'FAIL'} {time.time()-t:5.0f}s tests={n:<4} fail={fl} err={er} skip={sk:<3} {f}",
              flush=True)
        if not ok:
            for case in suite:
                b = case.find("failure") if case.find("failure") is not None else case.find("error")
                if b is not None:
                    print(f"       {case.attrib.get('name')} :: {(b.attrib.get('message') or '')[:160]}", flush=True)
    print(f"\n{len(files) - len(bad)} of {len(files)} files OK · {total:,} tests run · {skipped} skipped · "
          f"{time.time()-started:.0f}s · {'GREEN' if not bad else 'RED: ' + ', '.join(bad)}")
    return len(bad)


if __name__ == "__main__":
    raise SystemExit(main())
