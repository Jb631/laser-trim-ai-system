# Process Findings Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The app works out, per model and after every ingest, what the process data says should change to raise yield or cut trimming — and shows it on the Model page and in one ranked list across models.

**Architecture:** A new `laser_trim_analyzer/findings/` package. One loader turns a model's stored tracks and captured trim passes into plain `TrackView` objects; analyzers are pure functions over those (no SQL, no Tk), each returning zero or more `Finding`s; an engine ranks them and caches them in two new tables; the screens only ever read the cache. Intermediate sweeps are graded by `margin_ratio`, a pure re-statement of the app's linearity rule that the engine re-verifies per model against the app's stored verdicts and refuses to use where it does not hold.

**Tech Stack:** Python 3.12+, SQLAlchemy 2.0 (SQLite), customtkinter, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-17-process-recommendations-design.md` (sections "Finding catalogue", "4. The engine and its surfaces", "5. Testing"). The spec deferred this plan until real trajectory data could be looked at; that happened on 2026-09-20 on the home slice (`Work Files/dev_db/slice.db`: 9,118 trim analyses, 19,903 captured passes) and is recorded in `TRACKER.md` section B.

## This plan's code has already run

Tasks 1–7 and the Task 9 widget were prototyped on 2026-09-20 and executed against the real slice **before** this plan was written, because the previous plan shipped nine defects in code that had never run. What the prototype produced, so an implementer can tell a transcription slip from a real problem:

| Model | Tracks | Yardstick vs stored verdict | Findings |
|---|---|---|---|
| 8232-1 | 6,635 | 100.0% of 6,508 | 4 — ink target (+3.9 pts ≈ 34 units/yr, ρ = −0.15, within laser 1 and one recipe) and three recipe changes (laser 2, 2019: 84% → 32%; 2020: back to 84%; laser 1, 2024: one cut → two, 24% → 50%) |
| 8340-1 | 2,320 | 100.0% of 2,320 | 0 — four recipe runs in history, nothing worth acting on |
| 2475-10 | 162 | 100.0% of 156 | 1 — 20% arrive already inside linearity limits; 27 of those 31 are below the resistance floor |

`refresh_findings(db)` over the whole slice: 5 findings in 16 s. The 20 unit tests below passed against the prototype. **Transcribe the code as written; if a test fails, suspect the transcription first.**

## Global Constraints

Every task's requirements include these.

- **The app is an analysis tool, never a screening tool.** No output routes, rejects or grades a unit in flight. Findings point at a lever, never at a part.
- **Only four levers exist:** laser settings (same day), laser limit table (same day), ink formulation / incoming resistance (next lot), deposition / upstream (next lot, or ECN). **The ATP linearity and resistance specs can never be a lever** — `Finding` raises on anything outside `LEVERS`.
- **It must be able to say nothing.** Thin sample, weak relationship, or "this lever is not your problem here" produce NO finding. Every analyzer ships with a negative test. No analyzer may manufacture a recommendation.
- **Linearity is zero-tolerance against per-point limits.** Bands are position-dependent ("bowtie"); never treat `linearity_spec` as "the band". A blank error or blank limit is UNGRADED, never 0.0.
- **Final-test verdicts are not used by these analyzers.** Units may be hand-trimmed between the laser and final test (decided per failure — there is no list of hand-trim models), and 98.5% of stored final-test rows await re-grading. "Good" here means good *at the laser*.
- **On lasers 1 and 3 (LTS, LTS3) the `Lin Error` sheet is a duplicate of the last real cut.** Exclude it whenever cuts are counted.
- **Show the shop's laser names, never the code's letters:** `laser_label()` from `core/models.py` (A is laser TWO).
- **Workers never call Tk.** Background work posts to the UI with `self.safe_after(...)`.
- **Database safety.** Never open `data/analysis.db` read-write from a test or script. Any test that constructs a `Processor` must inject BOTH `laser_trim_analyzer.database.manager._db_manager` AND `laser_trim_analyzer.database._db_manager`. QA harnesses run on a copy and the copy is deleted.
- **No weak checks.** Every new test and sweep check is made to FAIL first against a deliberate break, and the break reverted from a `cp` backup (never `git checkout --`). A run that collects zero tests is a failed run. Counts come from the junit `<testsuite>` element.
- **Fail-first steps: beat the bytecode cache.** A same-length edit made within the same second as the last run leaves Python serving the OLD `.pyc` (it validates by mtime + size), so a break appears not to fail and a restore appears not to restore. This bit the plan's author while verifying these very steps. Run every fail-first check as `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest ... -p no:cacheprovider` after `find src tests -name __pycache__ -type d -prune -exec rm -rf {} +`, and confirm each restore with `cmp`.
- **`core/processor.py` is not touched.** `core/ingest_run.py` gains exactly one guarded phase (Task 8).
- **Do not run the full pytest suite** (~2.5 h). The gate is in Task 11.
- Commit trailer: `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`. Never `git add -A`. Do not push; the controller pushes after the final review.

## File Structure

| File | Responsibility |
|---|---|
| `src/laser_trim_analyzer/findings/__init__.py` | package marker |
| `src/laser_trim_analyzer/findings/grading.py` | `margin_ratio`, `in_limits` — grade any sweep by the app's rule |
| `src/laser_trim_analyzer/findings/stats.py` | `pct`, `spearman` — dependency-free |
| `src/laser_trim_analyzer/findings/model.py` | `Finding`, `LEVERS`, `rank` |
| `src/laser_trim_analyzer/findings/data.py` | `TrackView`, `PassView`, `load_model_tracks`, `yardstick_fidelity` — the only SQL the analyzers depend on |
| `src/laser_trim_analyzer/findings/analyzers/recipe_change.py` | setting-change detection (spec finding 9) |
| `src/laser_trim_analyzer/findings/analyzers/trim_effort.py` | trim avoidance + pass effectiveness (findings 4 and 6, plus the new "trim avoidance") |
| `src/laser_trim_analyzer/findings/analyzers/ink_target.py` | ink target, held to one laser and one recipe (finding 1) |
| `src/laser_trim_analyzer/findings/engine.py` | `compute_for_model`, `refresh_findings` |
| `src/laser_trim_analyzer/database/models.py` | + `ProcessFinding`, `ModelProcessFacts` |
| `src/laser_trim_analyzer/database/manager.py` | + `replace_process_findings`, `get_process_findings`, `get_process_facts` |
| `src/laser_trim_analyzer/core/ingest_run.py` | + one post-batch phase |
| `scripts/refresh_findings.py` | compute findings for a database from the command line |
| `src/laser_trim_analyzer/gui/v6/widgets/findings_tab.py` | Model page tab |
| `src/laser_trim_analyzer/gui/v6/pages/model_page.py` | + the tab and its loader |
| `src/laser_trim_analyzer/gui/v6/pages/findings_page.py` | ranked cross-model list |
| `src/laser_trim_analyzer/gui/v6/sidebar.py`, `gui/v6/app.py` | register the page |
| `scripts/app_qa_sweep.py` | + `check_findings_engine` |
| `tests/findings_helpers.py` + seven `tests/test_findings_*.py` | synthetic `TrackView` builders and the tests |

---

### Task 1: Grade any sweep by the app's rule

**Files:**
- Create: `src/laser_trim_analyzer/findings/__init__.py` (empty), `src/laser_trim_analyzer/findings/grading.py`, `src/laser_trim_analyzer/findings/stats.py`
- Test: `tests/test_findings_grading.py`

**Interfaces:**
- Produces: `margin_ratio(errors, upper, lower, min_points=3) -> Optional[float]`, `in_limits(errors, upper, lower) -> Optional[bool]`, `pct(flags) -> Optional[float]` (0–100), `spearman(xs, ys) -> Optional[float]`.

- [ ] **Step 1: Write the failing test** — `tests/test_findings_grading.py`

```python
import pytest

from laser_trim_analyzer.findings.grading import margin_ratio, in_limits
from laser_trim_analyzer.findings.stats import spearman, pct

BAND = 0.10
N_POINTS = 12


def sweep(worst: float):
    """A sweep whose best-offset worst point is `worst` x the band (1.0 = exactly at the limit)."""
    half = worst * BAND
    errors = tuple(half if i % 2 else -half for i in range(N_POINTS))
    return errors, tuple([BAND] * N_POINTS), tuple([-BAND] * N_POINTS)


def test_margin_ratio_is_one_exactly_at_the_limit():
    e, u, l = sweep(1.0)
    assert margin_ratio(e, u, l) == pytest.approx(1.0, abs=1e-6)
    assert in_limits(e, u, l) is True

def test_a_constant_offset_is_free():
    e, u, l = sweep(0.5)
    shifted = tuple(x + 5 * BAND for x in e)             # far outside the band before the offset
    assert margin_ratio(shifted, u, l) == pytest.approx(0.5, abs=1e-6)

def test_a_bowtie_waist_is_graded_point_by_point():
    # Two neighbouring points in the narrow waist that need OPPOSITE offsets: no single
    # offset fits both, however wide the band is elsewhere. (One such point alone can be
    # rescued by an offset -- the first draft of this test got that wrong.)
    e = [0.0] * 12; u = [0.10] * 12; l = [-0.10] * 12
    u[5] = u[6] = 0.01; l[5] = l[6] = -0.01
    e[5], e[6] = 0.03, -0.03
    assert in_limits(e, u, l) is False
    e[5], e[6] = 0.005, -0.005
    assert in_limits(e, u, l) is True

def test_blank_cells_are_ungraded_never_zero():
    e, u, l = map(list, sweep(0.5))
    e[3] = None; u[4] = None                              # ignored rows
    assert margin_ratio(e, u, l) == pytest.approx(0.5, abs=1e-6)
    assert margin_ratio([None] * 12, u, l) is None        # nothing measured: no grade, not a pass

def test_booleans_are_not_measurements():
    assert margin_ratio([True] * 12, [0.1] * 12, [-0.1] * 12) is None


def test_spearman_basics():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 2, 3, 4], [1, 1, 1, 1]) is None   # no variation: nothing to report
    assert pct([]) is None and pct([True, False]) == 50.0
```

- [ ] **Step 2: Run it and see it fail**

Run: `.venv/bin/python -m pytest tests/test_findings_grading.py -q -p no:cacheprovider --junitxml=/tmp/j1.xml`
Expected: collection error, `ModuleNotFoundError: No module named 'laser_trim_analyzer.findings'`.

- [ ] **Step 3: Implement** — create the empty `src/laser_trim_analyzer/findings/__init__.py`, then `grading.py`:

```python
"""Grade ANY sweep the way the app grades the final one.

`margin_ratio` is the app's linearity rule as one number: slide the whole
curve by the best single offset, then take the worst point as a fraction of
its own half-band. <= 1.0 means every graded point fits inside its per-point
limits -- the zero-tolerance rule. Points with a blank error or blank limit
are ungraded (never 0.0), which is how the stations mark ignored rows.

Checked 2026-09-20 against the app's stored verdict: 8,984 of 8,984 tracks on
the 8232-1/8340-1 slice, and 542 of 544 across 255 models in the sample base.
The engine re-checks this per model (`yardstick_fidelity`) and stays silent
where it does not hold.
"""
from typing import Optional, Sequence


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and x == x


def margin_ratio(errors: Sequence, upper: Sequence, lower: Sequence,
                 min_points: int = 3) -> Optional[float]:
    pts = [(e - (u + l) / 2.0, (u - l) / 2.0)
           for e, u, l in zip(errors or (), upper or (), lower or ())
           if _num(e) and _num(u) and _num(l) and u > l]
    if len(pts) < min_points:
        return None
    lo = min(d for d, _ in pts)
    hi = max(d for d, _ in pts)

    def worst(offset: float) -> float:
        return max(abs(d - offset) / h for d, h in pts)

    for _ in range(64):                       # worst() is convex in the offset; (2/3)^64 ~ 5e-12
        a = lo + (hi - lo) / 3.0
        b = hi - (hi - lo) / 3.0
        if worst(a) < worst(b):
            hi = b
        else:
            lo = a
    return worst((lo + hi) / 2.0)


_EDGE = 1e-9      # a sweep lying exactly ON its limit passes (the app's rule is inclusive);
                  # without this the search's last few ulps decide a knife-edge case.


def in_limits(errors: Sequence, upper: Sequence, lower: Sequence) -> Optional[bool]:
    r = margin_ratio(errors, upper, lower)
    return None if r is None else r <= 1.0 + _EDGE
```

and `stats.py`:

```python
"""Small, dependency-free statistics the analyzers share."""
from statistics import mean
from typing import List, Optional, Sequence


def pct(flags: Sequence[bool]) -> Optional[float]:
    flags = list(flags)
    return (100.0 * sum(1 for f in flags if f) / len(flags)) if flags else None


def _ranks(xs: Sequence[float]) -> List[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2.0 + 1.0       # ties share the average rank
        i = j + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = mean(rx), mean(ry)
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if sxx == 0 or syy == 0:
        return None                                   # no variation: no relationship to report
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / (sxx * syy) ** 0.5
```

- [ ] **Step 4: Run it and see it pass**

Run the Step 2 command. Expected: junit `tests="6" failures="0" errors="0"`.

- [ ] **Step 5: Prove the test can fail.** `cp src/laser_trim_analyzer/findings/grading.py /tmp/grading.bak`, then in `margin_ratio` change `zip(errors or (), upper or (), lower or ())` to `zip([0.0 if x is None else x for x in (errors or ())], upper or (), lower or ())` — a blank error becoming a measurement of zero, the exact mistake this project made on the final-test side. Run: `test_blank_cells_are_ungraded_never_zero` must FAIL on its last line (a sweep with nothing measured now "passes"). Restore with `cp /tmp/grading.bak src/laser_trim_analyzer/findings/grading.py` and confirm green.

- [ ] **Step 6: Commit**

```bash
git add src/laser_trim_analyzer/findings/__init__.py src/laser_trim_analyzer/findings/grading.py src/laser_trim_analyzer/findings/stats.py tests/test_findings_grading.py
git commit -m "feat: grade any sweep by the app's own linearity rule"
```

---

### Task 2: What a finding is, and how findings are ranked

**Files:**
- Create: `src/laser_trim_analyzer/findings/model.py`
- Test: `tests/test_findings_model.py`

**Interfaces:**
- Produces: `LEVERS: Dict[str, Tuple[label, lead_time]]` with keys `laser_settings`, `laser_limit_table`, `ink`, `deposition`; `Finding(...)` dataclass (fields below; raises `ValueError` on an unknown lever, or on a claimed gain with no `gain_definition`); properties `lever_label`, `lead_time`, `units_per_year`; `to_dict()`; `rank(findings) -> List[Finding]`.

- [ ] **Step 1: Write the failing test** — `tests/test_findings_model.py`

```python
import pytest

from laser_trim_analyzer.findings.model import Finding, LEVERS, rank


def _f(**kw):
    base = dict(model="M", analyzer="a", category="c", lever="ink", title="t", summary="s",
                systems=("A",), n_units=10, strength_name="n", strength_value=1.0)
    base.update(kw); return Finding(**base)

def test_the_atp_spec_can_never_be_a_lever():
    for bad in ("atp", "ATP linearity spec", "customer_spec", "drawing", ""):
        with pytest.raises(ValueError):
            _f(lever=bad)
    assert "atp" not in " ".join(LEVERS).lower()

def test_a_claimed_gain_must_be_defined():
    with pytest.raises(ValueError):
        _f(expected_gain_points=5.0)
    assert _f(expected_gain_points=5.0, gain_definition="x", annual_volume=1000).units_per_year == 50.0

def test_ranking_is_units_per_year_then_volume():
    a = _f(model="A", expected_gain_points=10.0, gain_definition="x", annual_volume=100)    # 10 units/yr
    b = _f(model="B", expected_gain_points=2.0, gain_definition="x", annual_volume=5000)    # 100 units/yr
    c = _f(model="C", annual_volume=9000)                                                   # no gain
    d = _f(model="D", annual_volume=50)
    assert [f.model for f in rank([a, c, d, b])] == ["B", "A", "C", "D"]
```

- [ ] **Step 2: Run it and see it fail** — `.venv/bin/python -m pytest tests/test_findings_model.py -q -p no:cacheprovider`. Expected: `ModuleNotFoundError: ...findings.model`.

- [ ] **Step 3: Implement** — `src/laser_trim_analyzer/findings/model.py`

```python
"""What a finding is, which levers it may name, and how findings are ranked."""
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

# The ONLY levers a finding may name, with their lead times (spec: "The levers,
# and their lead times"). The ATP linearity and resistance specs are the
# customer's drawing: they are deliberately absent, so no analyzer can ever
# recommend changing them -- Finding() raises on any other key.
LEVERS: Dict[str, Tuple[str, str]] = {
    "laser_settings":    ("Laser settings", "same day"),
    "laser_limit_table": ("Laser limit table", "same day"),
    "ink":               ("Ink formulation (incoming resistance)", "next lot"),
    "deposition":        ("Deposition / upstream", "next lot, or ECN"),
}


@dataclass
class Finding:
    model: str
    analyzer: str
    category: str
    lever: str
    title: str
    summary: str
    systems: Tuple[str, ...]
    n_units: int
    strength_name: str
    strength_value: Optional[float]
    expected_gain_points: Optional[float] = None   # yield points; None = cannot say honestly
    gain_definition: str = ""
    annual_volume: int = 0
    evidence: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.lever not in LEVERS:
            raise ValueError(f"{self.lever!r} is not a lever anyone can move; allowed: {sorted(LEVERS)}")
        if self.expected_gain_points is not None and not self.gain_definition:
            raise ValueError("a finding that claims a gain must say how the gain is defined")

    @property
    def lever_label(self) -> str:
        return LEVERS[self.lever][0]

    @property
    def lead_time(self) -> str:
        return LEVERS[self.lever][1]

    @property
    def units_per_year(self) -> Optional[float]:
        if self.expected_gain_points is None:
            return None
        return self.expected_gain_points / 100.0 * self.annual_volume

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.update(lever_label=self.lever_label, lead_time=self.lead_time,
                 units_per_year=self.units_per_year, systems=list(self.systems))
        return d


def rank(findings: List[Finding]) -> List[Finding]:
    """Recoverable units per year first; findings that cannot state a gain last, by volume."""
    with_gain = [f for f in findings if f.units_per_year is not None]
    without = [f for f in findings if f.units_per_year is None]
    with_gain.sort(key=lambda f: (-f.units_per_year, f.model, f.analyzer))
    without.sort(key=lambda f: (-f.annual_volume, f.model, f.analyzer))
    return with_gain + without
```

- [ ] **Step 4: Run it and see it pass.** Expected: 3 passed.

- [ ] **Step 5: Prove the ATP test can fail.** Back up `model.py`, add `"atp_spec": ("ATP specification", "never")` to `LEVERS`, run: `test_the_atp_spec_can_never_be_a_lever` must FAIL on its last line. Restore from the backup.

- [ ] **Step 6: Commit**

```bash
git add src/laser_trim_analyzer/findings/model.py tests/test_findings_model.py
git commit -m "feat: a finding names a lever someone can move, or it does not exist"
```

---
### Task 3: One read of a model's tracks, shaped for the analyzers

**Files:**
- Create: `src/laser_trim_analyzer/findings/data.py`, `tests/findings_helpers.py`
- Test: `tests/test_findings_data.py`

**Interfaces:**
- Consumes: `in_limits` (Task 1).
- Produces: frozen dataclasses `PassView(index, sheet, errors, upper, lower, cut_setting)` and `TrackView(track_id, file_date, system, untrimmed_errors, untrimmed_resistance, trimmed_resistance, final_errors, final_upper, final_lower, linearity_pass, initial_r_low, initial_r_high, final_r_low, final_r_high, passes)` with property `recipe -> (n_cuts, (cut_setting, ...))`; `load_model_tracks(db, model) -> List[TrackView]` (ordered by date; `passes` = REAL cuts only); `yardstick_fidelity(tracks) -> {"n", "agreement", "faithful"}`.
- Produces for later tests: `tests/findings_helpers.py` — `sweep`, `make_pass`, `make_track`, `days`, `label`, `START`, `era`, `ink_tracks`.

The values pinned below were read off the real fixtures on 2026-09-20.

- [ ] **Step 1: Write the failing test** — `tests/test_findings_data.py`

```python
from pathlib import Path

import pytest

FIXTURES = sorted(Path("tests/fixtures/trim").glob("*.xls"))


@pytest.fixture
def fixture_db(tmp_path, monkeypatch):
    """The four real 8232-1 workbooks through the real pipeline into a throwaway database."""
    from laser_trim_analyzer.core.processor import Processor
    from laser_trim_analyzer.database import manager as mgr
    import laser_trim_analyzer.database as dbpkg
    db = mgr.DatabaseManager(tmp_path / "findings.db")
    monkeypatch.setattr(mgr, "_db_manager", db, raising=False)      # BOTH globals, or get_database()
    monkeypatch.setattr(dbpkg, "_db_manager", db, raising=False)    # builds one at the configured path
    proc = Processor(use_ml=False)
    for f in FIXTURES:
        db.save_analysis(proc.process_file(f))
    return db


def test_the_four_fixtures_are_present():
    assert [f.name for f in FIXTURES] == ["dlts_8232-1_242.xls", "dlts_8232-1_243.xls",
                                          "lts_8232-1_193.xls", "lts_8232-1_194.xls"]


def test_loader_returns_real_cuts_only(fixture_db):
    from laser_trim_analyzer.findings.data import load_model_tracks
    tracks = load_model_tracks(fixture_db, "8232-1")
    assert len(tracks) == 4
    cuts = sorted((t.system, len(t.passes)) for t in tracks)
    # Laser 1 (LTS = system B) files hold Trim 1, Trim 2 and a duplicate `Lin Error` sheet:
    # TWO real cuts, not three.
    assert cuts == [("A", 2), ("A", 3), ("B", 2), ("B", 2)]
    assert all(not p.sheet.lower().startswith("lin error") for t in tracks for p in t.passes)


def test_recipe_is_what_the_laser_was_told_to_do(fixture_db):
    from laser_trim_analyzer.findings.data import load_model_tracks
    recipes = sorted(t.recipe for t in load_model_tracks(fixture_db, "8232-1"))
    assert recipes == [(2, (0.75, 0.88)), (2, (4100.0, 4100.0)), (2, (4100.0, 4100.0)),
                       (3, (0.75, 0.88, 0.88))]


def test_station_resistance_limits_come_through(fixture_db):
    from laser_trim_analyzer.findings.data import load_model_tracks
    tracks = load_model_tracks(fixture_db, "8232-1")
    assert {(t.final_r_low, t.final_r_high) for t in tracks} == {(5000.0, 5500.0)}
    assert {(t.initial_r_low, t.initial_r_high) for t in tracks if t.system == "A"} == {(4200.0, 4600.0)}
    # Laser 1 files carry no configured incoming window (spec: "Format asymmetry").
    assert {(t.initial_r_low, t.initial_r_high) for t in tracks if t.system == "B"} == {(None, None)}


def test_yardstick_reproduces_the_apps_verdict_on_the_fixtures(fixture_db):
    from laser_trim_analyzer.findings.data import load_model_tracks, yardstick_fidelity
    y = yardstick_fidelity(load_model_tracks(fixture_db, "8232-1"))
    assert y["n"] == 4 and y["agreement"] == 1.0
    assert y["faithful"] is False           # four tracks cannot vouch for a model (needs 30)


def test_an_unknown_model_is_empty_not_an_error(fixture_db):
    from laser_trim_analyzer.findings.data import load_model_tracks
    assert load_model_tracks(fixture_db, "no-such-model") == []

```

- [ ] **Step 2: Run it and see it fail** — `.venv/bin/python -m pytest tests/test_findings_data.py -q -p no:cacheprovider`. Expected: `test_the_four_fixtures_are_present` passes; the rest error with `ModuleNotFoundError: ...findings.data`.

- [ ] **Step 3: Implement** — `src/laser_trim_analyzer/findings/data.py`

```python
"""One read of a model's tracks, shaped for the analyzers. No analyzer writes SQL."""
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text

from .grading import in_limits


@dataclass(frozen=True)
class PassView:
    index: int
    sheet: str
    errors: Tuple
    upper: Tuple
    lower: Tuple
    cut_setting: Optional[float]


@dataclass(frozen=True)
class TrackView:
    track_id: int
    file_date: datetime
    system: str                       # the code's letter: A / B / C
    untrimmed_errors: Optional[Tuple]
    untrimmed_resistance: Optional[float]
    trimmed_resistance: Optional[float]
    final_errors: Optional[Tuple]
    final_upper: Optional[Tuple]
    final_lower: Optional[Tuple]
    linearity_pass: Optional[bool]    # the app's stored verdict
    initial_r_low: Optional[float]
    initial_r_high: Optional[float]
    final_r_low: Optional[float]
    final_r_high: Optional[float]
    passes: Tuple[PassView, ...]      # REAL cuts only, in order

    @property
    def recipe(self) -> Tuple:
        """(number of cuts, cut-length setting of each cut) -- what the laser was told to do."""
        return (len(self.passes),
                tuple(round(p.cut_setting, 2) if p.cut_setting is not None else None
                      for p in self.passes))


def _arr(js) -> Optional[Tuple]:
    if js is None:
        return None
    v = json.loads(js) if isinstance(js, (str, bytes)) else js
    return tuple(v) if isinstance(v, list) else None


def _date(v) -> Optional[datetime]:
    if isinstance(v, datetime):
        return v
    try:
        return datetime.fromisoformat(str(v).replace("T", " ").split(".")[0])
    except (TypeError, ValueError):
        return None


def _is_real_cut(sheet: Optional[str]) -> bool:
    # Lasers 1 and 3 (LTS, LTS3) write a final `Lin Error` sheet that repeats the
    # last real cut (spec, Known limits). Counting it would add a pass that never ran.
    return not (sheet or "").strip().lower().startswith("lin error")


def load_model_tracks(db, model: str) -> List[TrackView]:
    with db.session() as s:
        pass_rows = s.execute(text(
            "SELECT p.track_result_id, p.pass_index, p.sheet, p.errors, p.upper_limits, "
            "       p.lower_limits, p.laser_cut_length "
            "FROM trim_passes p JOIN track_results t ON t.id = p.track_result_id "
            "JOIN analysis_results a ON a.id = t.analysis_id "
            "WHERE a.model = :m ORDER BY p.track_result_id, p.pass_index"), {"m": model}).fetchall()
        track_rows = s.execute(text(
            "SELECT t.id, a.file_date, a.system, t.untrimmed_errors, t.untrimmed_resistance, "
            "       t.trimmed_resistance, t.error_data, t.upper_limits, t.lower_limits, t.linearity_pass, "
            "       s.initial_resistance_low, s.initial_resistance_high, "
            "       s.final_resistance_low, s.final_resistance_high "
            "FROM track_results t JOIN analysis_results a ON a.id = t.analysis_id "
            "LEFT JOIN trim_setup s ON s.analysis_id = a.id "
            "WHERE a.model = :m AND a.system IN ('A','B','C') "
            "ORDER BY a.file_date, t.id"), {"m": model}).fetchall()
    passes: Dict[int, List[PassView]] = {}
    for tid, idx, sheet, err, up, lo, cut in pass_rows:
        if not _is_real_cut(sheet):
            continue
        e, u, l = _arr(err), _arr(up), _arr(lo)
        if e is None or u is None or l is None:
            continue
        passes.setdefault(tid, []).append(PassView(idx, sheet or "", e, u, l, cut))
    out: List[TrackView] = []
    for r in track_rows:
        d = _date(r[1])
        if d is None:
            continue
        out.append(TrackView(
            track_id=r[0], file_date=d, system=str(r[2]),
            untrimmed_errors=_arr(r[3]), untrimmed_resistance=r[4], trimmed_resistance=r[5],
            final_errors=_arr(r[6]), final_upper=_arr(r[7]), final_lower=_arr(r[8]),
            linearity_pass=None if r[9] is None else bool(r[9]),
            initial_r_low=r[10], initial_r_high=r[11], final_r_low=r[12], final_r_high=r[13],
            passes=tuple(passes.get(r[0], ()))))
    return out


def yardstick_fidelity(tracks: List[TrackView]) -> Dict[str, Any]:
    """Does margin_ratio reproduce the app's stored verdict on THIS model's final sweeps?"""
    same = n = 0
    for t in tracks:
        if t.linearity_pass is None or not t.final_errors:
            continue
        g = in_limits(t.final_errors, t.final_upper, t.final_lower)
        if g is None:
            continue
        n += 1
        same += (g == t.linearity_pass)
    return {"n": n, "agreement": (same / n) if n else None,
            "faithful": bool(n >= 30 and same / n >= 0.99)}
```

- [ ] **Step 4: Run it and see it pass.** Expected: 6 passed.

- [ ] **Step 5: Prove the Lin Error test can fail.** Back up `data.py`; make `_is_real_cut` `return True`; run: `test_loader_returns_real_cuts_only` must FAIL with `("B", 3)` in the output and `test_recipe_is_what_the_laser_was_told_to_do` must FAIL too. Restore from the backup.

- [ ] **Step 6: Add the shared test helpers** — `tests/findings_helpers.py` (used by Tasks 4–7; `tests/` is on `sys.path`, so tests import it as `from findings_helpers import ...`)

```python
"""Build synthetic TrackViews for analyzer tests: small, deterministic, no database."""
from datetime import datetime, timedelta
from laser_trim_analyzer.findings.data import PassView, TrackView

BAND = 0.10                      # every synthetic point has limits of +/- BAND
N_POINTS = 12


def sweep(worst: float):
    """A sweep whose best-offset worst point is `worst` x the band (1.0 = exactly at the limit)."""
    half = worst * BAND
    errors = tuple(half if i % 2 else -half for i in range(N_POINTS))
    return errors, tuple([BAND] * N_POINTS), tuple([-BAND] * N_POINTS)


def make_pass(index: int, worst: float, cut=None, sheet=None) -> PassView:
    e, u, l = sweep(worst)
    return PassView(index, sheet or f"Trim {index}", e, u, l, cut)


def make_track(i: int, *, date: datetime, system="B", untrimmed_worst=2.0, passes=((0.8, 1.0),),
               r_in=4500.0, r_out=5200.0, linearity_pass=None, final_r=(5000.0, 5500.0),
               initial_r=(None, None)) -> TrackView:
    """passes = ((worst, cut_setting), ...). linearity_pass defaults to the last pass's grade."""
    pv = tuple(make_pass(k + 1, w, c) for k, (w, c) in enumerate(passes))
    last = passes[-1][0] if passes else untrimmed_worst
    fe, fu, fl = sweep(last)
    ue, _, _ = sweep(untrimmed_worst)
    return TrackView(track_id=i, file_date=date, system=system, untrimmed_errors=ue,
                     untrimmed_resistance=r_in, trimmed_resistance=r_out,
                     final_errors=fe, final_upper=fu, final_lower=fl,
                     linearity_pass=(last <= 1.0) if linearity_pass is None else linearity_pass,
                     initial_r_low=initial_r[0], initial_r_high=initial_r[1],
                     final_r_low=final_r[0], final_r_high=final_r[1], passes=pv)


def days(start: datetime, n: int, step_days: float = 1.0):
    return [start + timedelta(days=k * step_days) for k in range(n)]


def label(system: str) -> str:
    return {"A": "Laser 2 (DLTS)", "B": "Laser 1 (LTS)", "C": "Laser 3 (LTS3)"}.get(system, system)


START = datetime(2024, 1, 1)


def era(first_id: int, start: datetime, n: int, cuts, good_share: float, r_in: float = 4500.0, system: str = "B"):
    """`n` tracks, two a day, all on one recipe. `cuts` = the cut-length setting of each cut;
    `good_share` of them end inside limits after the LAST cut (deterministic, not random)."""
    out = []
    for k, d in enumerate(days(start, n, 0.5)):
        good = (k % 100) < good_share * 100
        ps = tuple((0.8 if (good and j == len(cuts) - 1) else 1.5, c) for j, c in enumerate(cuts))
        out.append(make_track(first_id + k, date=d, passes=ps, r_in=r_in, system=system))
    return out


def ink_tracks(n: int, start: datetime, cuts, p_of_r, first_id: int = 0, r_lo: float = 4000.0, r_hi: float = 5000.0):
    """`n` tracks on one recipe whose chance of ending good depends on incoming resistance through
    `p_of_r(r)`. Resistance is spread across the era rather than along time, and both it and the
    outcome are deterministic, so a test never flakes."""
    out = []
    for k, d in enumerate(days(start, n, 0.5)):
        r = r_lo + (r_hi - r_lo) * (k * 7919 % n) / n
        good = ((k * 104729) % 1000) / 1000.0 < p_of_r(r)
        ps = tuple((0.8 if (good and j == len(cuts) - 1) else 1.5, c) for j, c in enumerate(cuts))
        out.append(make_track(first_id + k, date=d, passes=ps, r_in=r))
    return out
```

- [ ] **Step 7: Commit**

```bash
git add src/laser_trim_analyzer/findings/data.py tests/findings_helpers.py tests/test_findings_data.py
git commit -m "feat: load a model's tracks and real cuts for the findings analyzers"
```

---

### Task 4: Setting-change detection

**Files:**
- Create: `src/laser_trim_analyzer/findings/analyzers/__init__.py` (empty), `src/laser_trim_analyzer/findings/analyzers/recipe_change.py`
- Test: `tests/test_findings_recipe_change.py`

**Interfaces:**
- Consumes: `TrackView.recipe`, `Finding`, `pct`.
- Produces: `analyze(model, tracks, laser_label) -> (history: List[dict], findings: List[Finding])`; `describe(recipe) -> str` (reused by Task 6). `laser_label` is a callable `system -> str`, injected so analyzers never import the GUI-facing name table themselves.

A change is a FINDING only when both sides hold at least `MIN_SIDE = 100` graded tracks and the result moved at least `MIN_MOVE_POINTS = 10`. Every stable run is HISTORY regardless. The first prototype raised five "findings" for 8232-1 including moves of −1 and +2 points and one resting on 31 tracks; these thresholds are why that stopped.

- [ ] **Step 1: Write the failing test** — `tests/test_findings_recipe_change.py`

```python
from datetime import datetime

import pytest

from laser_trim_analyzer.findings.analyzers import recipe_change
from findings_helpers import START, era, label


def test_a_recipe_change_that_moved_the_result_is_a_finding():
    tracks = era(0, START, 240, (1.0,), 0.25) + era(1000, datetime(2024, 7, 1), 240, (1.0, 2.0), 0.55)
    history, findings = recipe_change.analyze("M", tracks, label)
    assert len(history) == 2
    assert len(findings) == 1
    f = findings[0]
    assert "1 cut" in f.title and "2 cuts" in f.title and f.lever == "laser_settings"
    assert f.expected_gain_points is None                 # a detection never claims a gain
    assert f.evidence["moved_points"] == pytest.approx(30.0, abs=3.0)
    assert "not the only thing that changed" in f.summary

def test_a_constant_recipe_says_nothing():
    history, findings = recipe_change.analyze("M", era(0, START, 400, (1.0,), 0.4), label)
    assert findings == [] and len(history) == 1

def test_a_change_that_moved_nothing_is_history_not_a_finding():
    tracks = era(0, START, 240, (1.0,), 0.40) + era(1000, datetime(2024, 7, 1), 240, (1.0, 2.0), 0.42)
    history, findings = recipe_change.analyze("M", tracks, label)
    assert findings == [] and len(history) == 2

def test_a_change_resting_on_a_thin_side_says_nothing():
    tracks = era(0, START, 40, (1.0,), 0.10) + era(1000, datetime(2024, 7, 1), 240, (1.0, 2.0), 0.60)
    _, findings = recipe_change.analyze("M", tracks, label)
    assert findings == []
```

- [ ] **Step 2: Run it and see it fail.** Expected: `ModuleNotFoundError: ...findings.analyzers`.

- [ ] **Step 3: Implement** — empty `analyzers/__init__.py`, then `recipe_change.py`:

```python
"""Setting-change detection: the laser's recipe changed -- did the result follow?

A track's recipe is what the laser was told to do: how many cuts, and the
cut-length setting of each. Quarter by quarter, per laser, find the dominant
recipe; when a STABLE quarter's recipe differs from the previous stable one,
that is a change, and the trim result either side of it is the evidence.
Reports what happened. Never claims the recipe CAUSED it: incoming resistance
either side is disclosed beside it, because the two often move together.
"""
from collections import Counter
from statistics import median
from typing import Any, Dict, List, Tuple

from ..model import Finding
from ..stats import pct

MIN_BUCKET = 30        # tracks in a quarter before its recipe counts as known
DOMINANT = 0.80        # share one recipe needs for the quarter to count as stable
MIN_SIDE = 100         # graded tracks needed on EACH side before a change is a finding
MIN_MOVE_POINTS = 10.0 # a change that moved the result less than this is history, not a finding
MAX_EVENTS = 3         # per laser, most recent first


def _quarter(d) -> str:
    return f"{d.year} Q{(d.month - 1) // 3 + 1}"


def describe(recipe) -> str:
    n, cuts = recipe
    settings = ", ".join("?" if c is None else f"{c:g}" for c in cuts)
    word = "cut" if n == 1 else "cuts"
    return f"{n} {word}" + (f" (cut length {settings})" if any(c is not None for c in cuts) else "")


def _side(tracks) -> dict:
    graded = [t.linearity_pass for t in tracks if t.linearity_pass is not None]
    rs = [t.untrimmed_resistance for t in tracks if t.untrimmed_resistance]
    return {"n": len(tracks), "trim_pass_pct": pct(graded), "graded_n": len(graded),
            "median_incoming_r": median(rs) if rs else None,
            "first": min(t.file_date for t in tracks).date().isoformat(),
            "last": max(t.file_date for t in tracks).date().isoformat()}


def analyze(model: str, tracks, laser_label) -> Tuple[List[Dict[str, Any]], List[Finding]]:
    """(recipe history for the facts strip, findings). Every stable run is HISTORY;
    only a change that moved the result, with enough units either side, is a FINDING."""
    history: List[Dict[str, Any]] = []
    findings: List[Finding] = []
    for system in sorted({t.system for t in tracks}):
        cut = [t for t in tracks if t.system == system and t.passes]
        buckets = {}
        for t in cut:
            buckets.setdefault(_quarter(t.file_date), []).append(t)
        stable = []                                    # [(quarter, recipe, tracks)] in time order
        for q in sorted(buckets, key=lambda k: min(t.file_date for t in buckets[k])):
            ts = buckets[q]
            recipe, count = Counter(t.recipe for t in ts).most_common(1)[0]
            if len(ts) >= MIN_BUCKET and count / len(ts) >= DOMINANT:
                stable.append((q, recipe, [t for t in ts if t.recipe == recipe]))
        runs = []                                      # consecutive stable quarters, same recipe
        for q, recipe, ts in stable:
            if runs and runs[-1]["recipe"] == recipe:
                runs[-1]["tracks"] += ts
                runs[-1]["quarters"].append(q)
            else:
                runs.append({"recipe": recipe, "tracks": list(ts), "quarters": [q]})
        for run in runs:
            history.append({"system": system, "recipe": describe(run["recipe"]),
                            "quarters": run["quarters"], **_side(run["tracks"])})
        events = []
        for before, after in zip(runs, runs[1:]):
            b, a = _side(before["tracks"]), _side(after["tracks"])
            if b["trim_pass_pct"] is None or a["trim_pass_pct"] is None:
                continue
            if min(b["graded_n"], a["graded_n"]) < MIN_SIDE:
                continue                                   # too thin to call
            if abs(a["trim_pass_pct"] - b["trim_pass_pct"]) < MIN_MOVE_POINTS:
                continue                                   # it changed; nothing happened
            events.append((before, after, b, a))
        for before, after, b, a in events[-MAX_EVENTS:]:
            moved = a["trim_pass_pct"] - b["trim_pass_pct"]
            findings.append(Finding(
                model=model, analyzer="recipe_change", category="Setting change",
                lever="laser_settings", systems=(system,),
                title=f"{laser_label(system)}: recipe changed from {describe(before['recipe'])} "
                      f"to {describe(after['recipe'])}",
                summary=(f"Between {before['quarters'][-1]} and {after['quarters'][0]} the recipe on "
                         f"{laser_label(system)} changed. Units leaving the laser inside their trim limits went "
                         f"from {b['trim_pass_pct']:.0f}% ({b['graded_n']:,} tracks) to "
                         f"{a['trim_pass_pct']:.0f}% ({a['graded_n']:,} tracks), a move of {moved:+.0f} points. "
                         f"Median incoming resistance was {b['median_incoming_r']:,.0f} before and "
                         f"{a['median_incoming_r']:,.0f} after, so the recipe is not the only thing that changed."
                         if b["median_incoming_r"] and a["median_incoming_r"] else
                         f"Between {before['quarters'][-1]} and {after['quarters'][0]} the recipe on "
                         f"{laser_label(system)} changed; trim pass went from {b['trim_pass_pct']:.0f}% to "
                         f"{a['trim_pass_pct']:.0f}%."),
                n_units=b["n"] + a["n"],
                strength_name="tracks on the smaller side of the change",
                strength_value=float(min(b["graded_n"], a["graded_n"])),
                expected_gain_points=None,             # a detection, not a recommendation
                evidence={"before": {**b, "recipe": describe(before["recipe"]), "quarters": before["quarters"]},
                          "after": {**a, "recipe": describe(after["recipe"]), "quarters": after["quarters"]},
                          "moved_points": moved}))
    return history, findings
```

- [ ] **Step 4: Run it and see it pass.** Expected: 4 passed.

- [ ] **Step 5: Prove the silence tests can fail.** Back up the file; set `MIN_MOVE_POINTS = 0.0`: `test_a_change_that_moved_nothing_is_history_not_a_finding` must FAIL. Restore. Set `MIN_SIDE = 1`: `test_a_change_resting_on_a_thin_side_says_nothing` must FAIL. Restore from the backup.

- [ ] **Step 6: Commit**

```bash
git add src/laser_trim_analyzer/findings/analyzers/__init__.py src/laser_trim_analyzer/findings/analyzers/recipe_change.py tests/test_findings_recipe_change.py
git commit -m "feat: detect a recipe change and report what the result did either side"
```

---

### Task 5: Trim avoidance and pass effectiveness

**Files:**
- Create: `src/laser_trim_analyzer/findings/analyzers/trim_effort.py`
- Test: `tests/test_findings_trim_effort.py`

**Interfaces:**
- Consumes: `in_limits`, `Finding`, `pct`, `TrackView`.
- Produces: `analyze(model, tracks, laser_label) -> (facts: Dict[system, dict], findings: List[Finding])`. Fact keys per system: `tracks_cut`, `cuts` (`{"1": n, "2": n, "3+": n}`), `graded_untrimmed_n`, `arrive_in_spec_n`, `arrive_in_spec_pct`, `arrive_in_spec_below_r_floor_n`, `in_limits_after_cut1_pct`, `after_cut1_n`, `multi_cut_n`, `multi_in_limits_after_first_pct`, `multi_in_limits_after_last_pct`.

The untrimmed sweep is graded against the FIRST cut's limits, and only when both share a position grid (same length) — true for 3,000 of 3,000 8232-1 tracks checked. Facts are always returned; findings only above the thresholds.

- [ ] **Step 1: Write the failing test** — `tests/test_findings_trim_effort.py`

```python
import pytest

from laser_trim_analyzer.findings.analyzers import trim_effort
from findings_helpers import START, days, make_track, label


def test_units_arriving_in_spec_below_the_resistance_floor_point_at_the_ink():
    ds = days(START, 200)
    tracks = [make_track(i, date=d, system="A", untrimmed_worst=0.6 if i % 4 == 0 else 2.0,
                         passes=((0.7, 0.75),), r_in=4300.0) for i, d in enumerate(ds)]
    facts, findings = trim_effort.analyze("M", tracks, label)
    assert facts["A"]["arrive_in_spec_pct"] == pytest.approx(25.0)
    f = [x for x in findings if x.category == "Trim avoidance"]
    assert len(f) == 1 and f[0].lever == "ink" and f[0].expected_gain_points is None

def test_a_model_where_nothing_arrives_in_spec_says_nothing():
    tracks = [make_track(i, date=d, untrimmed_worst=2.5, passes=((1.4, 1.0), (0.8, 2.0)))
              for i, d in enumerate(days(START, 200))]
    facts, findings = trim_effort.analyze("M", tracks, label)
    assert findings == []
    assert facts["B"]["arrive_in_spec_pct"] == 0.0
    assert facts["B"]["multi_in_limits_after_first_pct"] == 0.0
    assert facts["B"]["multi_in_limits_after_last_pct"] == 100.0

def test_later_cuts_that_add_nothing_are_a_finding():
    tracks = [make_track(i, date=d, untrimmed_worst=2.5,
                         passes=((1.4, 1.0), (1.4, 2.0)) if i % 2 else ((0.8, 1.0), (0.8, 2.0)))
              for i, d in enumerate(days(START, 200))]
    _, findings = trim_effort.analyze("M", tracks, label)
    f = [x for x in findings if x.category == "Pass effectiveness"]
    assert len(f) == 1 and f[0].lever == "laser_settings"
```

- [ ] **Step 2: Run it and see it fail.** Expected: `ImportError: cannot import name 'trim_effort'`.

- [ ] **Step 3: Implement** — `src/laser_trim_analyzer/findings/analyzers/trim_effort.py`

```python
"""How much of the trimming is needed, and what each cut buys.

Company goal (James, 2026-09-20): "not have to trim at all and if we do trim
as little as possible to not overtrim and not tie up capacity at the laser."
Two questions follow. How many units arrive ALREADY inside their linearity
limits, and are cut anyway? And on tracks that get more than one cut, what
does the last cut add? Facts are always returned (the Model page shows them);
a FINDING is raised only where there is something a person could act on.
"""
from typing import Any, Dict, List, Tuple

from ..grading import in_limits
from ..model import Finding
from ..stats import pct

MIN_N = 100
ARRIVE_IN_SPEC_PCT = 10.0      # below this, "don't trim" is not yet a realistic lever
USELESS_PASS_POINTS = 5.0      # a last cut adding less than this is laser time for nothing


def _facts_for(tracks) -> Dict[str, Any]:
    cut = [t for t in tracks if t.passes]
    arrive = []                                        # graded BEFORE any cut, against cut 1's limits
    for t in cut:
        p1 = t.passes[0]
        if t.untrimmed_errors and len(t.untrimmed_errors) == len(p1.errors):
            g = in_limits(t.untrimmed_errors, p1.upper, p1.lower)
            if g is not None:
                arrive.append((t, g))
    after1 = [g for g in (in_limits(t.passes[0].errors, t.passes[0].upper, t.passes[0].lower) for t in cut) if g is not None]
    multi = [t for t in cut if len(t.passes) >= 2]
    m_first = [in_limits(t.passes[0].errors, t.passes[0].upper, t.passes[0].lower) for t in multi]
    m_last = [in_limits(t.passes[-1].errors, t.passes[-1].upper, t.passes[-1].lower) for t in multi]
    pairs = [(a, b) for a, b in zip(m_first, m_last) if a is not None and b is not None]
    counts: Dict[str, int] = {}
    for t in cut:
        k = str(len(t.passes)) if len(t.passes) < 3 else "3+"
        counts[k] = counts.get(k, 0) + 1
    in_spec = [t for t, g in arrive if g]
    below_floor = [t for t in in_spec if t.final_r_low and t.untrimmed_resistance
                   and t.untrimmed_resistance < t.final_r_low]
    return {"tracks_cut": len(cut), "cuts": counts,
            "graded_untrimmed_n": len(arrive),
            "arrive_in_spec_n": len(in_spec),
            "arrive_in_spec_pct": pct([g for _, g in arrive]),
            "arrive_in_spec_below_r_floor_n": len(below_floor),
            "in_limits_after_cut1_pct": pct(after1), "after_cut1_n": len(after1),
            "multi_cut_n": len(pairs),
            "multi_in_limits_after_first_pct": pct([a for a, _ in pairs]),
            "multi_in_limits_after_last_pct": pct([b for _, b in pairs])}


def analyze(model: str, tracks, laser_label) -> Tuple[Dict[str, Any], List[Finding]]:
    facts: Dict[str, Any] = {}
    findings: List[Finding] = []
    for system in sorted({t.system for t in tracks}):
        f = _facts_for([t for t in tracks if t.system == system])
        if not f["tracks_cut"]:
            continue
        facts[system] = f
        if (f["graded_untrimmed_n"] >= MIN_N and f["arrive_in_spec_pct"] is not None
                and f["arrive_in_spec_pct"] >= ARRIVE_IN_SPEC_PCT):
            mostly_r = f["arrive_in_spec_below_r_floor_n"] * 2 >= f["arrive_in_spec_n"]
            findings.append(Finding(
                model=model, analyzer="trim_effort", category="Trim avoidance",
                lever="ink" if mostly_r else "laser_settings", systems=(system,),
                title=f"{laser_label(system)}: {f['arrive_in_spec_pct']:.0f}% of units arrive already inside "
                      f"their linearity limits and are cut anyway",
                summary=(f"{f['arrive_in_spec_n']:,} of {f['graded_untrimmed_n']:,} tracks fit their per-point "
                         f"limits before the first cut. "
                         + (f"{f['arrive_in_spec_below_r_floor_n']:,} of them arrive below the final resistance "
                            f"floor, so they are being trimmed to raise resistance, not to fix linearity: the "
                            f"lever is the incoming-resistance target."
                            if mostly_r else
                            "Most of them already meet the final resistance floor too, so the cut itself is the "
                            "question: these units may not need the laser at all.")),
                n_units=f["graded_untrimmed_n"],
                strength_name="share arriving inside limits (%)",
                strength_value=f["arrive_in_spec_pct"],
                expected_gain_points=None,             # saves trimming, not yield points
                evidence={"facts": f}))
        if (f["multi_cut_n"] >= MIN_N and f["multi_in_limits_after_last_pct"] is not None):
            added = f["multi_in_limits_after_last_pct"] - f["multi_in_limits_after_first_pct"]
            if added < USELESS_PASS_POINTS:
                findings.append(Finding(
                    model=model, analyzer="trim_effort", category="Pass effectiveness",
                    lever="laser_settings", systems=(system,),
                    title=f"{laser_label(system)}: the extra cuts add only {added:+.0f} points",
                    summary=(f"On {f['multi_cut_n']:,} tracks that received more than one cut, "
                             f"{f['multi_in_limits_after_first_pct']:.0f}% were inside limits after the first cut and "
                             f"{f['multi_in_limits_after_last_pct']:.0f}% after the last. The later cuts are using "
                             f"laser time without moving the result."),
                    n_units=f["multi_cut_n"], strength_name="points added by the later cuts",
                    strength_value=added, expected_gain_points=None, evidence={"facts": f}))
    return facts, findings
```

- [ ] **Step 4: Run it and see it pass.** Expected: 3 passed.

- [ ] **Step 5: Prove the silence test can fail.** Back up; set `ARRIVE_IN_SPEC_PCT = -1.0`: `test_a_model_where_nothing_arrives_in_spec_says_nothing` must FAIL (a 0% finding appears). Restore from the backup.

- [ ] **Step 6: Commit**

```bash
git add src/laser_trim_analyzer/findings/analyzers/trim_effort.py tests/test_findings_trim_effort.py
git commit -m "feat: how many units arrive already in spec, and what each cut buys"
```

---

### Task 6: The ink target, held to one laser and one recipe

**Files:**
- Create: `src/laser_trim_analyzer/findings/analyzers/ink_target.py`
- Test: `tests/test_findings_ink_target.py`

**Interfaces:**
- Consumes: `describe` (Task 4), `spearman`, `pct`, `Finding`.
- Produces: `analyze(model, tracks, laser_label) -> List[Finding]` (zero or one). Evidence keys: `bins` (five dicts `r_low, r_high, n, success_pct`), `window`, `overall_pct`, `recipe`, `configured_incoming`, `period`.

**Expected gain, as the spec requires it stated:** the good-at-laser rate inside the recommended window minus the rate over the whole group, where good = the app's stored trim linearity verdict AND the final resistance inside the station's own final limits (when the file carries them). **Strength of support:** Spearman correlation between incoming resistance and good-at-laser.

The last test below is the reason this analyzer exists in this shape: 8232-1's history showed a −0.63 correlation between resistance and yield that turned out to have a laser move and a one-cut → two-cut recipe change folded into it.

- [ ] **Step 1: Write the failing test** — `tests/test_findings_ink_target.py`

```python
from datetime import datetime

from laser_trim_analyzer.findings.analyzers import ink_target
from findings_helpers import START, ink_tracks, label


def test_resistance_that_separates_good_from_bad_is_a_finding_with_a_defined_gain():
    tracks = ink_tracks(600, START, (1.0, 2.0), lambda r: 0.75 if r < 4400 else 0.25)
    (f,) = ink_target.analyze("M", tracks, label)
    assert f.lever == "ink" and "lower" in f.title
    assert f.expected_gain_points > 10 and f.gain_definition
    assert f.strength_value < -0.10
    assert len(f.evidence["bins"]) == 5 and f.evidence["window"]["n"] >= 0.4 * 600

def test_resistance_that_separates_nothing_says_nothing():
    assert ink_target.analyze("M", ink_tracks(600, START, (1.0, 2.0), lambda r: 0.5), label) == []

def test_a_thin_sample_says_nothing():
    assert ink_target.analyze("M", ink_tracks(150, START, (1.0, 2.0), lambda r: 0.9 if r < 4400 else 0.1), label) == []

def test_a_recipe_change_cannot_masquerade_as_a_resistance_effect():
    # Era 1: one cut, HIGH resistance, 20% good.  Era 2: two cuts, LOW resistance, 60% good.
    # Pooled, low resistance "predicts" success. Inside either era it predicts nothing.
    old = ink_tracks(500, START, (1.0,), lambda r: 0.20, first_id=0, r_lo=4500.0, r_hi=5000.0)
    new = ink_tracks(500, datetime(2025, 1, 1), (1.0, 2.0), lambda r: 0.60, first_id=5000, r_lo=4000.0, r_hi=4500.0)
    from laser_trim_analyzer.findings.stats import spearman
    pooled = spearman([t.untrimmed_resistance for t in old + new], [1.0 if t.linearity_pass else 0.0 for t in old + new])
    assert pooled < -0.25                                  # the trap is really there...
    assert ink_target.analyze("M", old + new, label) == [] # ...and the analyzer does not fall in
```

- [ ] **Step 2: Run it and see it fail.** Expected: `ImportError: cannot import name 'ink_target'`.

- [ ] **Step 3: Implement** — `src/laser_trim_analyzer/findings/analyzers/ink_target.py`

```python
"""Ink target: does incoming resistance move the result -- with everything else held still?

8232-1's yield history looked like a resistance story (corr -0.63) until two
other changes turned up inside it: a move between lasers and a change from
one cut to two. So this analyzer never pools. It works inside ONE laser and
ONE recipe -- the most recent such group with enough units -- and only there
asks whether incoming resistance separates good from bad.

Success is the app's stored trim verdict AND the final resistance landing
inside the station's own final limits (when the file carries them). Final-
test verdicts are deliberately not used: units may be hand-trimmed between
the laser and final test, so final test does not grade the laser's work.
"""
from typing import List, Optional

from ..model import Finding
from ..stats import pct, spearman
from .recipe_change import describe

MIN_N = 200
BINS = 5
MIN_ABS_RHO = 0.10
MIN_GAIN_POINTS = 3.0
MIN_WINDOW_SHARE = 0.40        # a window nobody can hit is not a recommendation


def _success(t) -> Optional[bool]:
    if t.linearity_pass is None:
        return None
    if t.final_r_low and t.final_r_high and t.trimmed_resistance:
        return bool(t.linearity_pass and t.final_r_low <= t.trimmed_resistance <= t.final_r_high)
    return bool(t.linearity_pass)


def analyze(model: str, tracks, laser_label) -> List[Finding]:
    groups = {}
    for t in tracks:
        if t.passes and t.untrimmed_resistance and _success(t) is not None:
            groups.setdefault((t.system, t.recipe), []).append(t)
    eligible = [(k, v) for k, v in groups.items() if len(v) >= MIN_N]
    if not eligible:
        return []                                       # thin sample: say nothing
    (system, recipe), ts = max(eligible, key=lambda kv: max(t.file_date for t in kv[1]))
    ts = sorted(ts, key=lambda t: t.untrimmed_resistance)
    rs = [t.untrimmed_resistance for t in ts]
    ok = [1.0 if _success(t) else 0.0 for t in ts]
    rho = spearman(rs, ok)
    if rho is None or abs(rho) < MIN_ABS_RHO:
        return []                                       # resistance is not the lever here
    n = len(ts)
    edges = [round(i * n / BINS) for i in range(BINS + 1)]
    bins = []
    for lo, hi in zip(edges, edges[1:]):
        chunk = ts[lo:hi]
        bins.append({"r_low": chunk[0].untrimmed_resistance, "r_high": chunk[-1].untrimmed_resistance,
                     "n": len(chunk), "success_pct": pct([_success(t) for t in chunk])})
    overall = pct([bool(x) for x in ok])
    best = None
    for width in (2, 3):
        for i in range(0, BINS - width + 1):
            chunk = ts[edges[i]:edges[i + width]]
            if len(chunk) / n < MIN_WINDOW_SHARE:
                continue
            rate = pct([_success(t) for t in chunk])
            if best is None or rate > best["success_pct"]:
                best = {"r_low": chunk[0].untrimmed_resistance, "r_high": chunk[-1].untrimmed_resistance,
                        "n": len(chunk), "success_pct": rate}
    if best is None:
        return []
    gain = best["success_pct"] - overall
    if gain < MIN_GAIN_POINTS:
        return []
    direction = "lower" if rho < 0 else "higher"
    # The station's CURRENT incoming window: from the most recent track that carries one.
    # `ts` was sorted by RESISTANCE for the binning above, so walking it backwards finds the
    # highest-resistance track, not the newest -- a stale window reported as the current one.
    with_window = [t for t in ts if t.initial_r_low and t.initial_r_high]
    configured = None
    if with_window:
        newest = max(with_window, key=lambda t: t.file_date)
        configured = (newest.initial_r_low, newest.initial_r_high)
    first, last = min(t.file_date for t in ts).date(), max(t.file_date for t in ts).date()
    return [Finding(
        model=model, analyzer="ink_target", category="Ink target",
        lever="ink", systems=(system,),
        title=f"Incoming resistance: aim {direction} -- {best['r_low']:,.0f} to {best['r_high']:,.0f} Ω does best",
        summary=(f"Within {laser_label(system)} running {describe(recipe)} ({first} to {last}, {n:,} tracks), units "
                 f"arriving between {best['r_low']:,.0f} and {best['r_high']:,.0f} Ω left the laser good "
                 f"{best['success_pct']:.0f}% of the time against {overall:.0f}% overall. "
                 + (f"The station is set to accept {configured[0]:,.0f} to {configured[1]:,.0f} Ω incoming. "
                    if configured else "These files carry no configured incoming window, so this is a computed "
                    "target, not a comparison with a setting. ")
                 + "Laser and recipe are held constant, so neither explains the difference."),
        n_units=n, strength_name="Spearman correlation, incoming resistance vs good-at-laser",
        strength_value=rho, expected_gain_points=gain,
        gain_definition=("good-at-laser rate inside the recommended window minus the rate over the whole group; "
                         "good = the app's trim linearity verdict AND final resistance inside the station's "
                         "final limits"),
        evidence={"bins": bins, "window": best, "overall_pct": overall, "recipe": describe(recipe),
                  "configured_incoming": configured, "period": [first.isoformat(), last.isoformat()]})]
```


> **Corrected during execution (2026-09-20, Task 4–6 review).** The prototype built `configured` by walking the RESISTANCE-sorted list backwards, so it reported the highest-resistance track's incoming window as the station's current one; and `trim_effort`'s `MIN_N` gate had no test. Both were the plan author's defects, transcribed faithfully. Fixed in a fix round with three added tests (`test_the_configured_window_reported_is_the_most_recent_one…`, `test_no_configured_window_is_said_plainly`, `test_a_thin_sample_says_nothing_but_still_reports_the_facts`); the code block above already carries the fix. Task 5 therefore ends with 4 tests and Task 6 with 6.

- [ ] **Step 4: Run it and see it pass.** Expected: 4 passed.

- [ ] **Step 5: Prove the confound test can fail.** Back up; change the grouping line to `groups.setdefault((t.system, (0, ())), []).append(t)` so every recipe on a laser is pooled: `test_a_recipe_change_cannot_masquerade_as_a_resistance_effect` must FAIL (a pooled finding appears) and no other test may. Restore from the backup.

- [ ] **Step 6: Commit**

```bash
git add src/laser_trim_analyzer/findings/analyzers/ink_target.py tests/test_findings_ink_target.py
git commit -m "feat: the ink target, computed inside one laser and one recipe"
```

---
### Task 7: The engine, and the cache the screens read

**Files:**
- Create: `src/laser_trim_analyzer/findings/engine.py`
- Modify: `src/laser_trim_analyzer/database/models.py` (add two classes after `class TrimSetup`), `src/laser_trim_analyzer/database/manager.py` (add three methods to `DatabaseManager`, next to `get_baseline_requalification`)
- Test: `tests/test_findings_engine_db.py`

**Interfaces:**
- Consumes: everything from Tasks 1–6; `laser_label` from `core/models.py`.
- Produces: `compute_for_model(db, model) -> (facts: dict, findings: List[Finding])`; `refresh_findings(db, models=None) -> int`. `DatabaseManager.replace_process_findings(model, facts, findings: List[dict]) -> int`, `.get_process_findings(model=None) -> List[dict]` (ranked: `units_per_year` descending with NULLs last, then `annual_volume`), `.get_process_facts(model) -> Optional[dict]`.
- `facts` keys: `model`, `tracks`, `annual_volume`, `latest`, `yardstick`, `recipe_history`, `trim_effort` (`None` when the yardstick is not faithful for this model).

New tables need no migration code: `Base.metadata.create_all(self._engine, checkfirst=True)` already runs at startup (`manager.py:284`), which is how `trim_passes` arrived.

- [ ] **Step 1: Write the failing test** — `tests/test_findings_engine_db.py`

```python
from dataclasses import replace

from findings_helpers import START, era, ink_tracks


def _db(tmp_path):
    from laser_trim_analyzer.database.manager import DatabaseManager
    return DatabaseManager(tmp_path / "engine.db")


def _hot():
    return ink_tracks(600, START, (1.0, 2.0), lambda r: 0.75 if r < 4400 else 0.25)


def test_the_two_cache_tables_exist_on_a_fresh_database(tmp_path):
    import sqlalchemy as sa
    db = _db(tmp_path)
    with db.session() as s:
        names = {r[0] for r in s.execute(sa.text("SELECT name FROM sqlite_master WHERE type='table'"))}
    assert {"process_findings", "model_process_facts"} <= names


def test_refresh_stores_ranked_findings_and_always_stores_facts(tmp_path, monkeypatch):
    from laser_trim_analyzer.findings import engine
    db = _db(tmp_path)
    tracks = {"HOT": _hot(), "QUIET": ink_tracks(600, START, (1.0, 2.0), lambda r: 0.5)}
    monkeypatch.setattr(engine, "load_model_tracks", lambda _db, m: tracks[m])
    assert engine.refresh_findings(db, ["HOT", "QUIET"]) == 1
    ranked = db.get_process_findings()
    assert [d["model"] for d in ranked] == ["HOT"]
    hot = ranked[0]
    assert hot["lever"] == "ink" and hot["lever_label"].startswith("Ink") and hot["lead_time"] == "next lot"
    assert hot["units_per_year"] > 0
    assert len(hot["evidence"]["bins"]) == 5           # the evidence survived JSON, not a silent "null"
    assert db.get_process_findings("QUIET") == []
    quiet = db.get_process_facts("QUIET")              # silence still leaves the measurements
    assert quiet["tracks"] == 600 and quiet["yardstick"]["faithful"] is True
    assert db.get_process_facts("never-seen") is None


def test_refresh_replaces_rather_than_appends(tmp_path, monkeypatch):
    from laser_trim_analyzer.findings import engine
    db = _db(tmp_path)
    hot = _hot()
    monkeypatch.setattr(engine, "load_model_tracks", lambda _db, m: hot)
    engine.refresh_findings(db, ["HOT"])
    engine.refresh_findings(db, ["HOT"])
    assert len(db.get_process_findings("HOT")) == 1


def test_one_model_failing_does_not_stop_the_rest(tmp_path, monkeypatch):
    from laser_trim_analyzer.findings import engine
    db = _db(tmp_path)
    hot = _hot()

    def loader(_db, model):
        if model == "BROKEN":
            raise RuntimeError("boom")
        return hot
    monkeypatch.setattr(engine, "load_model_tracks", loader)
    assert engine.refresh_findings(db, ["BROKEN", "HOT"]) == 1
    assert len(db.get_process_findings("HOT")) == 1


def test_an_unfaithful_yardstick_silences_the_grading_analyzer(tmp_path, monkeypatch):
    from laser_trim_analyzer.findings import engine
    db = _db(tmp_path)
    # Stored verdicts that contradict the sweeps: the yardstick cannot vouch for this model.
    liars = [replace(t, linearity_pass=not t.linearity_pass) for t in era(0, START, 200, (1.0,), 0.5)]
    monkeypatch.setattr(engine, "load_model_tracks", lambda _db, m: liars)
    facts, _ = engine.compute_for_model(db, "LIAR")
    assert facts["yardstick"]["faithful"] is False
    assert facts["trim_effort"] is None


def test_a_date_in_the_evidence_is_flattened_not_silently_nulled(tmp_path):
    from datetime import datetime
    db = _db(tmp_path)
    finding = {"model": "X", "analyzer": "a", "category": "c", "lever": "ink", "title": "t", "summary": "s",
               "expected_gain_points": None, "units_per_year": None, "annual_volume": 5, "n_units": 5,
               "evidence": {"when": datetime(2026, 1, 6)}}
    db.replace_process_findings("X", {"tracks": 1, "when": datetime(2026, 1, 6)}, [finding])
    assert db.get_process_findings("X")[0]["evidence"]["when"].startswith("2026-01-06")
    assert db.get_process_facts("X")["when"].startswith("2026-01-06")


def test_every_finding_carries_the_models_annual_volume(tmp_path, monkeypatch):
    from laser_trim_analyzer.findings import engine
    db = _db(tmp_path)
    hot = _hot()                                       # 600 tracks over 300 days: all inside one year
    monkeypatch.setattr(engine, "load_model_tracks", lambda _db, m: hot)
    _, findings = engine.compute_for_model(db, "HOT")
    assert findings and all(f.annual_volume == 600 for f in findings)

```

- [ ] **Step 2: Run it and see it fail.** Expected: the first test fails on the missing table names; the rest with `ImportError: cannot import name 'engine'`.

- [ ] **Step 3: Add the two tables** to `src/laser_trim_analyzer/database/models.py`, directly after the `TrimSetup` class (`Column, Integer, String, Text, Float, DateTime` and `SafeJSON` are already imported/defined in that file):

```python
class ProcessFinding(Base):
    """One cached process finding (see laser_trim_analyzer/findings). Recomputed
    per model after an ingest; the screens only ever read this table."""
    __tablename__ = 'process_findings'

    id = Column(Integer, primary_key=True)
    model = Column(String(64), nullable=False, index=True)
    analyzer = Column(String(48), nullable=False)
    category = Column(String(64), nullable=False)
    lever = Column(String(32), nullable=False)
    title = Column(Text, nullable=False)
    summary = Column(Text, nullable=False)
    expected_gain_points = Column(Float)        # NULL = the analyzer could not state a gain honestly
    units_per_year = Column(Float)              # gain x annual volume; the ranking key
    annual_volume = Column(Integer, default=0)
    n_units = Column(Integer, default=0)
    payload = Column(SafeJSON)                  # Finding.to_dict(), evidence included
    computed_at = Column(DateTime, nullable=False)


class ModelProcessFacts(Base):
    """Per-model measured facts shown above the findings, finding or no finding."""
    __tablename__ = 'model_process_facts'

    model = Column(String(64), primary_key=True)
    facts = Column(SafeJSON)
    computed_at = Column(DateTime, nullable=False)
```

- [ ] **Step 4: Add the three methods** to `DatabaseManager` in `src/laser_trim_analyzer/database/manager.py`, directly after `get_baseline_requalification` (`datetime` and `Optional` are already imported there):

```python
    def replace_process_findings(self, model: str, facts: dict, findings: list) -> int:
        """Replace everything cached for one model. `findings` are Finding.to_dict() dicts.

        Everything is passed through json first: SafeJSON answers an unserialisable
        value by silently storing "null" (that cost 7% of the trim_setup blocks in
        2026-09), so dates and tuples are flattened HERE, loudly, not there, quietly.
        """
        import json as _json
        from laser_trim_analyzer.database.models import ProcessFinding, ModelProcessFacts

        def clean(obj):
            return _json.loads(_json.dumps(obj, default=str))

        now = datetime.utcnow()
        with self.session() as s:
            s.query(ProcessFinding).filter(ProcessFinding.model == model).delete(synchronize_session=False)
            for f in findings:
                d = clean(f)
                s.add(ProcessFinding(
                    model=model, analyzer=d["analyzer"], category=d["category"], lever=d["lever"],
                    title=d["title"], summary=d["summary"],
                    expected_gain_points=d.get("expected_gain_points"),
                    units_per_year=d.get("units_per_year"),
                    annual_volume=int(d.get("annual_volume") or 0), n_units=int(d.get("n_units") or 0),
                    payload=d, computed_at=now))
            row = s.get(ModelProcessFacts, model)
            if row is None:
                s.add(ModelProcessFacts(model=model, facts=clean(facts), computed_at=now))
            else:
                row.facts = clean(facts)
                row.computed_at = now
        return len(findings)

    def get_process_findings(self, model: Optional[str] = None) -> list:
        """Cached findings, ranked: recoverable units per year first, then by volume."""
        from sqlalchemy import text as _text
        import json as _json
        sql = ("SELECT payload FROM process_findings "
               + ("WHERE model = :m " if model else "")
               + "ORDER BY (units_per_year IS NULL), units_per_year DESC, annual_volume DESC, model, id")
        with self.session() as s:
            rows = s.execute(_text(sql), {"m": model} if model else {}).fetchall()
        out = []
        for (payload,) in rows:
            d = _json.loads(payload) if isinstance(payload, (str, bytes)) else payload
            if isinstance(d, dict):
                out.append(d)
        return out

    def get_process_facts(self, model: str) -> Optional[dict]:
        from sqlalchemy import text as _text
        import json as _json
        with self.session() as s:
            row = s.execute(_text("SELECT facts, computed_at FROM model_process_facts WHERE model = :m"),
                            {"m": model}).fetchone()
        if not row:
            return None
        facts = _json.loads(row[0]) if isinstance(row[0], (str, bytes)) else row[0]
        if not isinstance(facts, dict):
            return None
        facts["computed_at"] = str(row[1])
        return facts
```

- [ ] **Step 5: Implement the engine** — `src/laser_trim_analyzer/findings/engine.py`

```python
"""Run every analyzer for a model, rank what they found, and cache it.

Runs after an ingest, on the ingest worker thread (never the Tk thread), for
the models that ingest touched. The screens only ever read the cache.
"""
import logging
from datetime import timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple

from sqlalchemy import text

from .analyzers import ink_target, recipe_change, trim_effort
from .data import load_model_tracks, yardstick_fidelity
from .model import Finding, rank

logger = logging.getLogger(__name__)


def _laser_label(system: str) -> str:
    # The shop's names, never the code's letters (A is laser TWO). Imported late
    # so this package stays importable in a bare test of the analyzers.
    from laser_trim_analyzer.core.models import laser_label
    return laser_label(system)


def compute_for_model(db, model: str) -> Tuple[Dict[str, Any], List[Finding]]:
    tracks = load_model_tracks(db, model)
    facts: Dict[str, Any] = {"model": model, "tracks": len(tracks)}
    if not tracks:
        return facts, []
    latest = max(t.file_date for t in tracks)
    volume = sum(1 for t in tracks if t.file_date >= latest - timedelta(days=365))
    facts["annual_volume"] = volume
    facts["latest"] = latest.date().isoformat()
    fidelity = yardstick_fidelity(tracks)
    facts["yardstick"] = fidelity
    findings: List[Finding] = []
    # Each analyzer is independently guarded: one failing must not blank the rest.
    try:
        history, changes = recipe_change.analyze(model, tracks, _laser_label)  # stored verdicts only
        facts["recipe_history"] = history
        findings += changes
    except Exception:
        logger.exception("findings: recipe_change failed for %s", model)
    try:
        findings += ink_target.analyze(model, tracks, _laser_label)         # stored verdicts only
    except Exception:
        logger.exception("findings: ink_target failed for %s", model)
    if fidelity["faithful"]:
        try:
            effort_facts, effort_findings = trim_effort.analyze(model, tracks, _laser_label)
            facts["trim_effort"] = effort_facts
            findings += effort_findings
        except Exception:
            logger.exception("findings: trim_effort failed for %s", model)
    else:
        # Grading intermediate sweeps is only honest where the yardstick reproduces
        # the app's own verdict. Where it does not, say nothing rather than guess.
        facts["trim_effort"] = None
    for f in findings:
        f.annual_volume = volume
    return facts, rank(findings)


def refresh_findings(db, models: Optional[Iterable[str]] = None) -> int:
    """Recompute and cache findings for `models` (default: every model with trim data).

    Returns how many findings were stored. One model failing never stops the rest.
    """
    if models is None:
        with db.session() as s:
            models = [r[0] for r in s.execute(text(
                "SELECT DISTINCT model FROM analysis_results "
                "WHERE system IN ('A','B','C') AND model IS NOT NULL ORDER BY model"))]
    stored = 0
    for model in models:
        try:
            facts, findings = compute_for_model(db, model)
            stored += db.replace_process_findings(model, facts, [f.to_dict() for f in findings])
        except Exception:
            logger.exception("findings: refresh failed for %s", model)
    return stored
```

- [ ] **Step 6: Run it and see it pass.** Expected: 7 passed.

- [ ] **Step 7: Prove the JSON test can fail.** Back up `manager.py`; in `replace_process_findings` make `clean` return `obj` unchanged. `test_a_date_in_the_evidence_is_flattened_not_silently_nulled` must FAIL: SafeJSON meets the `datetime`, swallows the `TypeError` and stores `"null"`, so the finding vanishes from the ranked list — the same silent loss that cost 7% of the `trim_setup` blocks. Restore from the backup.

- [ ] **Step 8: Commit**

```bash
git add src/laser_trim_analyzer/findings/engine.py src/laser_trim_analyzer/database/models.py src/laser_trim_analyzer/database/manager.py tests/test_findings_engine_db.py
git commit -m "feat: rank a model's findings and cache them for the screens"
```

---

### Task 8: Recompute after an ingest, and from the command line

**Files:**
- Modify: `src/laser_trim_analyzer/core/ingest_run.py` — `_post_batch`, after the drift-advance block
- Create: `scripts/refresh_findings.py`
- Test: `tests/test_findings_ingest_hook.py`

**Interfaces:**
- Consumes: `refresh_findings` (Task 7); `_post_batch(db, models_in_batch, new_trims, phases, on_phase)` and `_say(on_phase, text)` as they exist today.
- Produces: a `"findings"` entry in `phases` whenever the batch saved trims.

The hook calls `engine.refresh_findings` through the module (`_findings.refresh_findings`), not a from-import, so the tests can replace it. It runs on the ingest worker thread; it never touches Tk. Cost measured on the slice: 16 s for three models holding 9,117 tracks — after the full rebuild expect a few minutes per trim folder, once, inside a run that already takes hours.

- [ ] **Step 1: Write the failing test** — `tests/test_findings_ingest_hook.py`

```python
def _db(tmp_path):
    from laser_trim_analyzer.database.manager import DatabaseManager
    return DatabaseManager(tmp_path / "hook.db")


def test_findings_refresh_runs_after_a_batch_that_saved_trims(tmp_path, monkeypatch):
    from laser_trim_analyzer.core import ingest_run
    from laser_trim_analyzer.findings import engine
    seen = []
    monkeypatch.setattr(engine, "refresh_findings", lambda _db, models: seen.append(list(models)) or 0)
    phases = {}
    ingest_run._post_batch(_db(tmp_path), {"M2", "M1"}, 3, phases, None)
    assert seen == [["M1", "M2"]]
    assert "findings" in phases


def test_a_final_test_only_batch_does_not_recompute_findings(tmp_path, monkeypatch):
    from laser_trim_analyzer.core import ingest_run
    from laser_trim_analyzer.findings import engine
    seen = []
    monkeypatch.setattr(engine, "refresh_findings", lambda _db, models: seen.append(list(models)) or 0)
    phases = {}
    ingest_run._post_batch(_db(tmp_path), {"M1"}, 0, phases, None)
    assert seen == [] and "findings" not in phases


def test_a_failing_refresh_never_breaks_the_ingest(tmp_path, monkeypatch):
    from laser_trim_analyzer.core import ingest_run
    from laser_trim_analyzer.findings import engine

    def boom(_db, models):
        raise RuntimeError("findings exploded")
    monkeypatch.setattr(engine, "refresh_findings", boom)
    phases = {}
    ingest_run._post_batch(_db(tmp_path), {"M1"}, 3, phases, None)     # must not raise
    assert "findings" in phases


def test_the_command_line_tool_refuses_the_production_database():
    import subprocess, sys
    r = subprocess.run([sys.executable, "scripts/refresh_findings.py", "data/analysis.db"],
                       capture_output=True, text=True)
    assert r.returncode == 2 and "REFUSED" in r.stdout

```

- [ ] **Step 2: Run it and see it fail.** Expected: the first and third fail (`seen == []`, no `"findings"` phase); the second passes already; the fourth fails (no such script).

- [ ] **Step 3: Add the phase.** In `src/laser_trim_analyzer/core/ingest_run.py`, inside `_post_batch`, directly after the line `phases["advance"] = time.monotonic() - t`, add:

```python
    # Process findings read trim verdicts and captured passes only, so a batch
    # that saved no trims changes nothing they depend on -- the same gate the
    # rematch above uses. Guarded like every other phase here: findings are an
    # aid, and an aid must never be able to fail an ingest.
    if new_trims:
        t = time.monotonic()
        try:
            from laser_trim_analyzer.findings import engine as _findings
            _say(on_phase, "Working out process findings…")
            stored = _findings.refresh_findings(db, sorted(models_in_batch))
            logger.info("Process findings refreshed for %d models (%d findings)",
                        len(models_in_batch), stored)
        except Exception:
            logger.exception("Process findings refresh after batch failed")
        phases["findings"] = time.monotonic() - t
```

- [ ] **Step 4: Create the command-line tool** — `scripts/refresh_findings.py`

```python
"""Work out process findings for a database and print them, ranked.

    python scripts/refresh_findings.py "Work Files/dev_db/slice.db"
    python scripts/refresh_findings.py "Work Files/dev_db/slice.db" 8232-1 8340-1

The app does this by itself after every ingest that saves trim files. This is
for a database that was built another way (scripts/build_dev_db.py), or to
recompute after the analyzers change. It WRITES the two cache tables, so it
refuses data/analysis.db unless --production is given.
"""
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main() -> int:
    args = [a for a in sys.argv[1:] if a != "--production"]
    if not args:
        print(__doc__)
        return 2
    path = Path(args[0])
    is_production = path.resolve() == (REPO / "data" / "analysis.db").resolve() or path.name == "analysis.db"
    if is_production and "--production" not in sys.argv:
        print("REFUSED: that is the production database. The app refreshes its findings itself after an "
              "ingest; pass --production only if you mean to write the cache tables there by hand.")
        return 2
    if not path.exists():
        print(f"no such database: {path}")
        return 2

    import logging
    logging.disable(logging.WARNING)
    from laser_trim_analyzer.database import manager as mgr
    import laser_trim_analyzer.database as dbpkg
    from laser_trim_analyzer.findings.engine import refresh_findings

    db = mgr.DatabaseManager(path)
    mgr._db_manager = db                      # BOTH globals: nothing on this path may open
    dbpkg._db_manager = db                    # the configured database by accident.
    stored = refresh_findings(db, args[1:] or None)
    ranked = db.get_process_findings()
    print(f"{stored} findings stored; {len(ranked)} in the ranked list\n")
    for i, f in enumerate(ranked, 1):
        upy = f.get("units_per_year")
        gain = f"{upy:,.0f} units a year" if upy is not None else "no gain claimed"
        print(f"{i:>3}. {f['model']}  ·  {f['title']}")
        print(f"     {f['category']}  ·  lever: {f['lever_label']} ({f['lead_time']})  ·  {gain}")
        print(textwrap.fill(f["summary"], 100, initial_indent="     ", subsequent_indent="     "))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run it and see it pass.** Expected: 4 passed.

- [ ] **Step 6: Run it for real** on a COPY of the development database, and compare with the table at the top of this plan:

```bash
cp "Work Files/dev_db/slice.db" /tmp/findings_try.db
.venv/bin/python scripts/refresh_findings.py /tmp/findings_try.db
rm -f /tmp/findings_try.db*
```

Expected: `5 findings stored`, 8232-1's ink-target finding first (≈ 34 units a year), then three 8232-1 recipe changes and the 2475-10 trim-avoidance finding. If `Work Files/dev_db/slice.db` is absent on this machine, say so in the report and skip this step.

- [ ] **Step 7: Prove the guard test can fail.** Back up `ingest_run.py`; remove the `try/except` around the refresh: `test_a_failing_refresh_never_breaks_the_ingest` must FAIL with `RuntimeError`. Restore from the backup.

- [ ] **Step 8: Commit**

```bash
git add src/laser_trim_analyzer/core/ingest_run.py scripts/refresh_findings.py tests/test_findings_ingest_hook.py
git commit -m "feat: work out process findings after every ingest that saves trims"
```

---

### Task 9: The Model page "Findings" tab

**Files:**
- Create: `src/laser_trim_analyzer/gui/v6/widgets/findings_tab.py`
- Modify: `src/laser_trim_analyzer/gui/v6/pages/model_page.py`
- Test: `tests/test_findings_tab.py`

**Interfaces:**
- Consumes: `DatabaseManager.get_process_facts`, `.get_process_findings` (Task 7); `laser_label`.
- Produces: `FindingsTab(master, theme)` with `set_data(data: Optional[dict])` where `data = {"facts": dict, "findings": List[dict]}` or `None`.

Text only — no matplotlib — so the chart QA harness has nothing new to render. The widget was exercised on 2026-09-20 with the assertions below.

- [ ] **Step 1: Write the failing test** — `tests/test_findings_tab.py`

```python
import customtkinter as ctk

FACTS = {"tracks": 2351, "yardstick": {"n": 2351, "agreement": 1.0, "faithful": True},
         "trim_effort": {"B": {"tracks_cut": 2351, "cuts": {"1": 1059, "2": 1263, "3+": 29},
                               "graded_untrimmed_n": 2351, "arrive_in_spec_pct": 4.6,
                               "in_limits_after_cut1_pct": 21.0, "multi_cut_n": 1292,
                               "multi_in_limits_after_first_pct": 18.0,
                               "multi_in_limits_after_last_pct": 52.0}},
         "recipe_history": [{"system": "B", "first": "2023-09-22", "last": "2024-06-30",
                             "recipe": "1 cut (cut length 2950)", "n": 817, "trim_pass_pct": 24.0,
                             "median_incoming_r": 4601.0}]}
FINDING = {"model": "HOT", "analyzer": "ink_target", "title": "Incoming resistance: aim lower",
           "category": "Ink target", "lever": "ink", "lever_label": "Ink formulation (incoming resistance)",
           "lead_time": "next lot", "expected_gain_points": 3.9, "units_per_year": 34.0,
           "annual_volume": 873, "summary": "Within Laser 1 (LTS) running 2 cuts…",
           "strength_name": "Spearman", "strength_value": -0.15, "n_units": 346, "evidence": {}}


def _texts(widget):
    out = []
    for c in widget.winfo_children():
        if isinstance(c, ctk.CTkLabel):
            out.append(c.cget("text"))
        out.extend(_texts(c))
    return [x for x in out if x]          # CTkScrollableFrame owns one empty label of its own


def _tab(tk_root):
    from laser_trim_analyzer.gui.v6.theme import ThemeManager
    from laser_trim_analyzer.gui.v6.widgets.findings_tab import FindingsTab
    return FindingsTab(tk_root, theme=ThemeManager())


def test_before_anything_is_computed_it_says_so(tk_root):
    texts = _texts(_tab(tk_root))
    assert len(texts) == 1 and "No process findings have been computed" in texts[0]


def test_no_findings_reads_as_nothing_to_act_on_not_as_a_gap(tk_root):
    tab = _tab(tk_root)
    tab.set_data({"facts": FACTS, "findings": []})
    text = " | ".join(_texts(tab))
    assert "Nothing to act on" in text
    assert "Laser 1 (LTS)" in text and "System B" not in text      # the shop's names, never the letters
    assert "18%" in text and "52%" in text and "RECIPE HISTORY" in text


def test_a_finding_shows_its_lever_lead_time_and_gain(tk_root):
    tab = _tab(tk_root)
    tab.set_data({"facts": FACTS, "findings": [FINDING, {**FINDING, "title": "Recipe changed",
                                                         "units_per_year": None,
                                                         "expected_gain_points": None}]})
    text = " | ".join(_texts(tab))
    assert "Nothing to act on" not in text
    assert "34 units a year" in text and "no gain claimed" in text and "next lot" in text


def test_a_model_the_yardstick_cannot_vouch_for_says_why(tk_root):
    tab = _tab(tk_root)
    tab.set_data({"facts": dict(FACTS, trim_effort=None,
                                yardstick={"n": 40, "agreement": 0.9, "faithful": False}),
                  "findings": []})
    assert any("not graded for this model" in x for x in _texts(tab))


def test_set_data_none_returns_to_the_empty_state(tk_root):
    tab = _tab(tk_root)
    tab.set_data({"facts": FACTS, "findings": [FINDING]})
    tab.set_data(None)
    texts = _texts(tab)
    assert len(texts) == 1 and "No process findings" in texts[0]


def test_the_model_page_loads_the_tab_from_the_cache(make_app):
    # `_seed` gives the model real analyses so the page has something to open;
    # tests/ is on sys.path, so a sibling test module's helper is importable.
    from test_spec3c_model import _seed
    app = make_app()
    _seed(app.db, "HOT", fails_last=12)
    app.db.replace_process_findings("HOT", FACTS, [FINDING])
    app.set_model_route("HOT")
    page = app.page_container.get_page("model")
    page._reload = lambda **kw: None       # suppress on_show's BACKGROUND reload (see test_spec3c_model)
    app.show_page("model")
    del page._reload
    page.reload_now()                      # the synchronous path
    assert "Incoming resistance: aim lower" in " | ".join(_texts(page._findings_tab))

```

- [ ] **Step 2: Run it and see it fail.** Expected: `ModuleNotFoundError: ...widgets.findings_tab`.

- [ ] **Step 3: Create the widget** — `src/laser_trim_analyzer/gui/v6/widgets/findings_tab.py`

```python
"""Model page tab: what the process data says about THIS model, and what to do about it.

Facts first (always shown -- they are measurements), then findings (only where
there is something a person could act on), then the recipe history. A model
with no findings reads as "nothing to act on", never as a gap. Text only: no
chart in v1, so nothing here touches matplotlib or the chart QA harness.
"""
from typing import Any, Dict, List, Optional

import customtkinter as ctk

from laser_trim_analyzer.core.models import laser_label
from laser_trim_analyzer.gui.v6.theme import ThemeManager


def _pct(v) -> str:
    return "—" if v is None else f"{v:.0f}%"


class FindingsTab(ctk.CTkFrame):
    def __init__(self, master, theme: ThemeManager, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.theme = theme
        self._body = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self._body.pack(fill="both", expand=True)
        self.set_data(None)

    # ---- public ----
    def set_data(self, data: Optional[Dict[str, Any]]) -> None:
        for child in self._body.winfo_children():
            child.destroy()
        facts = (data or {}).get("facts")
        findings: List[Dict[str, Any]] = (data or {}).get("findings") or []
        if not facts:
            self._line("No process findings have been computed for this model yet. They are worked out "
                       "after each ingest; Settings can also refresh them.", muted=True)
            return
        self._heading("WHAT WAS MEASURED")
        self._facts(facts)
        self._heading("WHAT TO DO ABOUT IT")
        if not findings:
            self._line("Nothing to act on. No analyzer found a lever worth pulling on this model — "
                       "that is a result, not a gap.", muted=True)
        for f in findings:
            self._card(f)
        history = facts.get("recipe_history") or []
        if history:
            self._heading("RECIPE HISTORY")
            for run in history:
                self._line(f"{laser_label(run.get('system'))} · {run.get('first')} → {run.get('last')} · "
                           f"{run.get('recipe')} · {run.get('n', 0):,} tracks · "
                           f"{_pct(run.get('trim_pass_pct'))} left the laser inside limits · "
                           f"median incoming {run.get('median_incoming_r') or 0:,.0f} Ω", muted=True)

    # ---- pieces ----
    def _heading(self, text: str) -> None:
        t = self.theme
        ctk.CTkLabel(self._body, text=text, font=t.font(t.SIZE_CAPTION), text_color=t.TEXT_SECONDARY,
                     anchor="w").pack(fill="x", pady=(t.SPACE_MD, t.SPACE_SM))

    def _line(self, text: str, *, muted: bool = False) -> None:
        t = self.theme
        ctk.CTkLabel(self._body, text=text, font=t.font(t.SIZE_BODY), anchor="w", justify="left",
                     wraplength=900, text_color=t.TEXT_SECONDARY if muted else t.TEXT_PRIMARY
                     ).pack(fill="x", pady=(0, 2))

    def _facts(self, facts: Dict[str, Any]) -> None:
        y = facts.get("yardstick") or {}
        effort = facts.get("trim_effort")
        if effort is None:
            self._line("Intermediate passes are not graded for this model: the grading yardstick reproduces "
                       f"the app's own verdict on only {_pct((y.get('agreement') or 0) * 100)} of "
                       f"{y.get('n', 0):,} tracks here, so anything built on it would be a guess.", muted=True)
            return
        for system, f in effort.items():
            cuts = " · ".join(f"{k} cut{'s' if k != '1' else ''}: {v:,}" for k, v in sorted(f.get("cuts", {}).items()))
            self._line(f"{laser_label(system)} — {f.get('tracks_cut', 0):,} tracks cut  ({cuts})")
            self._line(f"    arrive already inside linearity limits: {_pct(f.get('arrive_in_spec_pct'))} "
                       f"of {f.get('graded_untrimmed_n', 0):,}    ·    inside limits after the first cut: "
                       f"{_pct(f.get('in_limits_after_cut1_pct'))}", muted=True)
            if f.get("multi_cut_n"):
                self._line(f"    tracks given more than one cut ({f['multi_cut_n']:,}): "
                           f"{_pct(f.get('multi_in_limits_after_first_pct'))} inside limits after the first → "
                           f"{_pct(f.get('multi_in_limits_after_last_pct'))} after the last", muted=True)

    def _card(self, f: Dict[str, Any]) -> None:
        t = self.theme
        card = ctk.CTkFrame(self._body, fg_color=t.CARD, corner_radius=8)
        card.pack(fill="x", pady=(0, t.SPACE_SM))
        ctk.CTkLabel(card, text=f.get("title", ""), font=t.font(t.SIZE_BODY, "bold"), anchor="w",
                     justify="left", wraplength=880, text_color=t.TEXT_PRIMARY
                     ).pack(fill="x", padx=t.SPACE_MD, pady=(t.SPACE_SM, 0))
        upy = f.get("units_per_year")
        gain = (f"{f.get('expected_gain_points', 0):+.1f} yield points ≈ {upy:,.0f} units a year"
                if upy is not None else "no gain claimed")
        ctk.CTkLabel(card, text=f"{f.get('category', '')}  ·  lever: {f.get('lever_label', '')} "
                                f"({f.get('lead_time', '')})  ·  {gain}",
                     font=t.font(t.SIZE_CAPTION), anchor="w", text_color=t.ACCENT
                     ).pack(fill="x", padx=t.SPACE_MD)
        ctk.CTkLabel(card, text=f.get("summary", ""), font=t.font(t.SIZE_BODY), anchor="w", justify="left",
                     wraplength=880, text_color=t.TEXT_PRIMARY).pack(fill="x", padx=t.SPACE_MD, pady=(2, 0))
        strength = f.get("strength_value")
        ctk.CTkLabel(card, text=f"{f.get('strength_name', '')}: "
                                f"{'—' if strength is None else format(strength, '.2f')}  ·  "
                                f"rests on {f.get('n_units', 0):,} tracks",
                     font=t.font(t.SIZE_CAPTION), anchor="w", text_color=t.TEXT_SECONDARY
                     ).pack(fill="x", padx=t.SPACE_MD, pady=(0, t.SPACE_SM))
```

- [ ] **Step 4: Wire it into the Model page** — four small edits in `src/laser_trim_analyzer/gui/v6/pages/model_page.py`:

1. With the other widget imports at the top:
```python
from laser_trim_analyzer.gui.v6.widgets.findings_tab import FindingsTab
```
2. In `build_content`, directly after `self._history_tab.pack(fill="both", expand=True)`:
```python
        self._findings_tab = FindingsTab(self._tabs.add("Findings"), theme=t)
        self._findings_tab.pack(fill="both", expand=True)
```
3. In `_reload`'s `work()`: on the line after `trim_ft, history = {}, {}` add `findings_data = None`, and directly before the line `def apply():` add (each loader on this page is independently guarded, so is this one):
```python
            try:
                process_facts = self.app.db.get_process_facts(model)
                if process_facts:
                    findings_data = {"facts": process_facts,
                                     "findings": self.app.db.get_process_findings(model)}
            except Exception:
                logger.exception("Model %s: process findings failed", model)
```
4. In `apply()`, directly after the `_try("history tab", ...)` line:
```python
                _try("findings tab", lambda: self._findings_tab.set_data(findings_data))
```

- [ ] **Step 5: Run it and see it pass.** Expected: 6 passed. If `test_the_model_page_loads_the_tab_from_the_cache` fails because `_seed`'s signature differs, read `tests/test_spec3c_model.py::_spc_app` and seed the same way it does — do not weaken the assertion.

- [ ] **Step 6: Prove the wiring test can fail.** Back up `model_page.py`; delete the `_try("findings tab", ...)` line: the last test must FAIL. Restore from the backup.

- [ ] **Step 7: Commit**

```bash
git add src/laser_trim_analyzer/gui/v6/widgets/findings_tab.py src/laser_trim_analyzer/gui/v6/pages/model_page.py tests/test_findings_tab.py
git commit -m "feat: a Findings tab on the Model page"
```

---
### Task 10: One ranked list across every model

**Files:**
- Create: `src/laser_trim_analyzer/gui/v6/pages/findings_page.py`
- Modify: `src/laser_trim_analyzer/gui/v6/sidebar.py` (`ITEMS`), `src/laser_trim_analyzer/gui/v6/app.py` (register the page), `tests/test_spec3a_shell.py` and `tests/test_spec3f_home.py` (both pin `Sidebar.ITEMS`)
- Test: `tests/test_findings_page.py`

**Interfaces:**
- Consumes: `DatabaseManager.get_process_findings()` (already ranked); `app.set_model_route(model)`, `app.show_page(name)`; `PageBase` (`build_content`, `on_show`, `safe_after`, `_zone_header`).
- Produces: page key `"findings"`.

This is a visible change to the app's shape: a seventh sidebar entry, placed after "Investigate". Two existing tests pin the sidebar list exactly — they exist to stop an ACCIDENTAL rename, and they must be updated deliberately here, not loosened.

- [ ] **Step 1: Write the failing test** — `tests/test_findings_page.py`

```python
import customtkinter as ctk


def _finding(model, title, units_per_year, volume=100):
    return {"model": model, "analyzer": "a", "category": "Ink target", "lever": "ink",
            "lever_label": "Ink formulation (incoming resistance)", "lead_time": "next lot",
            "title": title, "summary": "s",
            "expected_gain_points": None if units_per_year is None else 5.0,
            "units_per_year": units_per_year, "annual_volume": volume, "n_units": 300, "evidence": {}}


def _buttons(widget):
    out = []
    for c in widget.winfo_children():
        if isinstance(c, ctk.CTkButton):
            out.append(c.cget("text"))
        out.extend(_buttons(c))
    return out


def test_findings_is_in_the_sidebar_right_after_investigate():
    from laser_trim_analyzer.gui.v6.sidebar import Sidebar
    keys = [k for k, _ in Sidebar.ITEMS]
    assert ("findings", "Findings") in Sidebar.ITEMS
    assert keys.index("findings") == keys.index("model") + 1


def test_the_list_is_ranked_and_a_row_opens_the_model(make_app, monkeypatch):
    app = make_app()
    app.db.replace_process_findings("SMALL", {"tracks": 1}, [_finding("SMALL", "no gain here", None, volume=9000)])
    app.db.replace_process_findings("BIG", {"tracks": 1}, [_finding("BIG", "the big one", 500.0)])
    page = app.page_container.get_page("findings")
    page.reload_now()
    rows = _buttons(page)
    assert len(rows) == 2
    assert rows[0].startswith("BIG") and "500 units a year" in rows[0]        # a stated gain outranks volume
    assert rows[1].startswith("SMALL") and "no gain claimed" in rows[1]
    shown = []
    monkeypatch.setattr(app, "show_page", lambda name: shown.append(name))
    page._open("BIG")
    assert shown == ["model"] and app.consume_model_route() == "BIG"


def test_an_empty_cache_says_so(make_app):
    app = make_app()
    page = app.page_container.get_page("findings")
    page.reload_now()
    labels = [c.cget("text") for c in page._list.winfo_children() if isinstance(c, ctk.CTkLabel)]
    assert _buttons(page._list) == [] and any("No findings yet" in x for x in labels)

```

- [ ] **Step 2: Run it and see it fail.** Expected: all three fail (`findings` not in `Sidebar.ITEMS`; `get_page("findings")` is `None`).

- [ ] **Step 3: Create the page** — `src/laser_trim_analyzer/gui/v6/pages/findings_page.py`

```python
"""Findings -- every model's process findings in one ranked list (the front door).

Recoverable units a year first; findings that cannot state a gain after them,
by volume. Reads the cache only: nothing here computes anything.
"""
import logging
import threading
from typing import Any, Dict, List

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.page_base import PageBase

logger = logging.getLogger(__name__)


class FindingsPage(PageBase):
    page_title = "Findings"

    def __init__(self, master, *, theme, app, page_title="Findings"):
        self._rows: List[Dict[str, Any]] = []
        super().__init__(master, theme=theme, app=app, page_title=page_title)

    def build_content(self, parent):
        self._zone_header(parent, "WHAT TO CHANGE, BIGGEST FIRST",
                          "units a year recoverable, then by volume — click a row to open the model")
        self._list = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        self._list.pack(side="top", fill="both", expand=True)

    # ---- data ----
    def reload_now(self):
        """Synchronous reload + apply (the test path)."""
        self._apply(self._query())

    def on_show(self):
        """Load on a background thread; apply on the Tk thread via safe_after."""
        def work():
            rows = self._query()
            self.safe_after(lambda: self._apply(rows))
        threading.Thread(target=work, daemon=True).start()

    def _query(self) -> List[Dict[str, Any]]:
        try:
            return self.app.db.get_process_findings()
        except Exception:
            logger.exception("Findings page: load failed")
            return []

    def _apply(self, rows: List[Dict[str, Any]]) -> None:
        t = self.theme
        self._rows = rows
        for child in self._list.winfo_children():
            child.destroy()
        if not rows:
            ctk.CTkLabel(self._list, font=t.font(t.SIZE_BODY), text_color=t.TEXT_SECONDARY, anchor="w",
                         justify="left", wraplength=900,
                         text="No findings yet. They are worked out after each ingest that saves trim files; "
                              "a model with nothing worth acting on does not appear here."
                         ).pack(fill="x", pady=t.SPACE_SM)
            return
        for f in rows:
            upy = f.get("units_per_year")
            gain = f"{upy:,.0f} units a year" if upy is not None else "no gain claimed"
            ctk.CTkButton(
                self._list, anchor="w", fg_color=t.CARD, hover_color=t.ACCENT_HOVER,
                text_color=t.TEXT_PRIMARY, font=t.font(t.SIZE_BODY), corner_radius=8,
                text=f"{f.get('model', '')}   ·   {f.get('title', '')}\n"
                     f"{f.get('category', '')}  ·  lever: {f.get('lever_label', '')} "
                     f"({f.get('lead_time', '')})  ·  {gain}",
                command=lambda m=f.get("model"): self._open(m),
            ).pack(fill="x", pady=(0, t.SPACE_SM))

    def _open(self, model: str) -> None:
        self.app.set_model_route(model)
        self.app.show_page("model")

```

- [ ] **Step 4: Register it.** In `src/laser_trim_analyzer/gui/v6/sidebar.py` change `ITEMS` to:

```python
    ITEMS: List[Tuple[str, str]] = [
        ("home", "Home"), ("model", "Investigate"), ("findings", "Findings"), ("settings", "Settings"),
        ("dashboard", "Dashboard"), ("triage", "Triage"), ("process", "Process"),
    ]
```

In `src/laser_trim_analyzer/gui/v6/app.py`, with the other page imports add `from laser_trim_analyzer.gui.v6.pages.findings_page import FindingsPage`, and directly after the `"model"` page's `add_page(...)` call add:

```python
        self.page_container.add_page(
            "findings",
            FindingsPage(self.page_container, theme=self.theme, app=self, page_title="Findings"),
        )
```

- [ ] **Step 5: Update the two tests that pin the sidebar.** In `tests/test_spec3a_shell.py` (near line 77) and `tests/test_spec3f_home.py` (near line 99), change the expected list to the seven entries above, and add to each test's docstring: `Findings added 2026-09-20 (process-findings engine), deliberately, after Investigate.` Change nothing else in those tests.

- [ ] **Step 6: Run and see it pass.**

Run: `.venv/bin/python -m pytest tests/test_findings_page.py tests/test_spec3a_shell.py tests/test_spec3f_home.py -q -p no:cacheprovider --junitxml=/tmp/j10.xml`
Expected: junit `failures="0" errors="0"` with `tests` > 3.

- [ ] **Step 7: Prove the ranking test can fail.** Back up `manager.py`; in `get_process_findings` change the `ORDER BY` to `ORDER BY annual_volume DESC`: `test_the_list_is_ranked_and_a_row_opens_the_model` must FAIL (SMALL, with 9,000 volume and no gain, jumps ahead of BIG). Restore from the backup.

- [ ] **Step 8: Commit**

```bash
git add src/laser_trim_analyzer/gui/v6/pages/findings_page.py src/laser_trim_analyzer/gui/v6/sidebar.py src/laser_trim_analyzer/gui/v6/app.py tests/test_findings_page.py tests/test_spec3a_shell.py tests/test_spec3f_home.py
git commit -m "feat: a Findings page -- every model's findings, ranked"
```

---

### Task 11: Guard it in the sweep, make the gate a tool, and tell James

**Files:**
- Modify: `scripts/app_qa_sweep.py` — add `check_findings_engine`, call it from `main()`
- Create: `scripts/run_test_gate.py`
- Modify: `TRACKER.md`, `BRING_TO_WORK.md`, `docs/superpowers/specs/2026-09-17-process-recommendations-design.md`

**Interfaces:**
- Consumes: everything above; the sweep's `check(name, ok, detail)`, `warn(name, detail)`, `sqlalchemy_text`, `Path` (all already defined at the top of `scripts/app_qa_sweep.py`).

- [ ] **Step 1: Add the sweep check.** In `scripts/app_qa_sweep.py`, add this function next to the other `check_*` functions:

```python
def check_findings_engine(db) -> None:
    """Process findings: real-cut counting, grading fidelity, lever safety, cache round trip.

    Falsify before trusting (2026-09-20): make findings/data.py `_is_real_cut`
    return True and the real-cut check must go FAIL with ("B", 3) in its detail.
    """
    import shutil
    import tempfile
    from laser_trim_analyzer.core.processor import Processor
    from laser_trim_analyzer.database import manager as _mgr
    import laser_trim_analyzer.database as _dbpkg
    from laser_trim_analyzer.findings.data import load_model_tracks, yardstick_fidelity
    from laser_trim_analyzer.findings.engine import refresh_findings
    from laser_trim_analyzer.findings.model import Finding, LEVERS

    fixtures = sorted(Path("tests/fixtures/trim").glob("*.xls"))
    check("findings: the four trim fixtures are present", len(fixtures) == 4,
          f"{[f.name for f in fixtures]}")
    if len(fixtures) == 4:
        saved = (_mgr._db_manager, _dbpkg._db_manager)
        tmp = Path(tempfile.mkdtemp(prefix="findings_sweep_"))
        try:
            fdb = _mgr.DatabaseManager(tmp / "f.db")
            _mgr._db_manager = fdb                 # BOTH globals: a Processor must never
            _dbpkg._db_manager = fdb               # reach the configured database.
            proc = Processor(use_ml=False)
            for f in fixtures:
                fdb.save_analysis(proc.process_file(f))
            tracks = load_model_tracks(fdb, "8232-1")
            cuts = sorted((t.system, len(t.passes)) for t in tracks)
            check("findings: real cuts are counted, the duplicate Lin Error row is not",
                  cuts == [("A", 2), ("A", 3), ("B", 2), ("B", 2)], f"{cuts}")
            y = yardstick_fidelity(tracks)
            check("findings: the yardstick reproduces the app's verdict on the fixtures",
                  y["n"] == 4 and y["agreement"] == 1.0, f"{y}")
            stored = refresh_findings(fdb, ["8232-1"])
            facts = fdb.get_process_facts("8232-1")
            check("findings: four tracks find nothing, and the facts are cached anyway",
                  stored == 0 and facts is not None and facts.get("tracks") == 4,
                  f"stored={stored} tracks={None if facts is None else facts.get('tracks')}")
        except Exception as e:                      # an exception is a FAIL, never a skip
            check("findings: the engine runs on the fixtures", False, f"{type(e).__name__}: {e}")
        finally:
            _mgr._db_manager, _dbpkg._db_manager = saved
            shutil.rmtree(tmp, ignore_errors=True)

    try:
        Finding(model="m", analyzer="a", category="c", lever="atp_spec", title="t", summary="s",
                systems=("A",), n_units=1, strength_name="n", strength_value=1.0)
        check("findings: the ATP spec can never be named as a lever", False,
              "Finding accepted lever='atp_spec'")
    except ValueError:
        check("findings: the ATP spec can never be named as a lever", True, f"levers={sorted(LEVERS)}")

    # On the database under test: wherever passes were captured, the yardstick must hold.
    with db.session() as s:
        models = [r[0] for r in s.execute(sqlalchemy_text(
            "SELECT a.model FROM trim_passes p JOIN track_results t ON t.id = p.track_result_id "
            "JOIN analysis_results a ON a.id = t.analysis_id "
            "GROUP BY a.model ORDER BY COUNT(*) DESC LIMIT 5"))]
    if not models:
        warn("findings: this database predates the trim-pass capture",
             "no rows in trim_passes (rebuild pending) — the engine was checked on the fixtures only")
        return
    for m in models:
        y = yardstick_fidelity(load_model_tracks(db, m))
        check(f"findings: yardstick fidelity on {m}",
              y["n"] > 0 and y["agreement"] is not None and y["agreement"] >= 0.99, f"{y}")

```

and in `main()`, directly after the line `check_ft_regrade_dry_run(db)`, add `check_findings_engine(db)`.

- [ ] **Step 2: Make it fail first.** Back up `src/laser_trim_analyzer/findings/data.py`; make `_is_real_cut` `return True`; then:

```bash
find src tests scripts -name __pycache__ -type d -prune -exec rm -rf {} +
cp data/analysis.db /tmp/qa_copy.db
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B scripts/app_qa_sweep.py /tmp/qa_copy.db > /tmp/sweep_broken.log 2>&1
grep "findings:" /tmp/sweep_broken.log
```

The line `findings: real cuts are counted, the duplicate Lin Error row is not` must read **FAIL** with `('B', 3)` in its detail. Quote it in the report. Restore `data.py` from the backup (`cmp` to confirm), run the sweep again, and the same line must read PASS. Then `rm -f /tmp/qa_copy.db*`.

- [ ] **Step 3: Create the gate tool** — `scripts/run_test_gate.py`

```python
"""Run test files one at a time and report REAL counts from junit.

    python scripts/run_test_gate.py tests/test_a.py tests/test_b.py ...

Why this exists (2026-09-20, three mistakes in one day): a hand-picked list
missed the test that mattered; a zsh variable that did not word-split made
pytest collect ZERO tests while a wrapper printed GREEN; and this repo's
double-quiet pytest prints no "N passed" line, so a text-scraping counter read
every file as "0 passed". So: one file per process with a hard time limit (one
hanging file cannot freeze the gate), counts from the junit <testsuite>
element, and a file that collects nothing is a FAILURE. Exit code = files not OK.
"""
import os
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

LIMIT_S = 600


def main() -> int:
    files = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not files:
        print(__doc__)
        return 2
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    bad = total = 0
    for f in files:
        fd, jx = tempfile.mkstemp(suffix=".xml")
        os.close(fd)
        os.remove(jx)
        t = time.time()
        try:
            r = subprocess.run([sys.executable, "-B", "-m", "pytest", f, "-q", "-p", "no:cacheprovider",
                                f"--junitxml={jx}"], capture_output=True, text=True, timeout=LIMIT_S, env=env)
        except subprocess.TimeoutExpired:
            bad += 1
            print(f"HANG {time.time()-t:5.0f}s  killed after {LIMIT_S} s  {f}")
            continue
        if not os.path.exists(jx):
            bad += 1
            print(f"FAIL {time.time()-t:5.0f}s  no junit written (exit {r.returncode})  {f}")
            continue
        root = ET.parse(jx).getroot()
        os.remove(jx)
        suite = root if root.tag == "testsuite" else root[0]
        n, fl, er, sk = (int(suite.attrib.get(k, 0)) for k in ("tests", "failures", "errors", "skipped"))
        ok = n > 0 and fl == 0 and er == 0 and r.returncode == 0
        bad += not ok
        total += n
        print(f"{'OK  ' if ok else 'FAIL'} {time.time()-t:5.0f}s tests={n:<4} fail={fl} err={er} skip={sk:<3} {f}")
    print(f"\n{len(files) - bad} of {len(files)} files OK · {total:,} tests run · "
          f"{'GREEN' if bad == 0 else 'RED'}")
    return bad


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the gate** — selected by REFERENCE, plus the core gates a reference search cannot see:

```bash
find src tests scripts -name __pycache__ -type d -prune -exec rm -rf {} +
refs=($(grep -lE "laser_trim_analyzer\.findings|findings_helpers|findings_tab|findings_page|process_findings|_post_batch|Sidebar\.ITEMS|ingest_run" tests/test_*.py | sort))
core=(tests/test_parse_all_models.py tests/test_trim_capture_noop.py tests/test_trim_capture_db.py tests/test_spec3c_model.py)
.venv/bin/python scripts/run_test_gate.py "${refs[@]}" "${core[@]}"
```

(`refs=(...)` and `"${refs[@]}"` are zsh/bash ARRAYS on purpose: an unquoted `$VAR` does not word-split in zsh, pytest then collects nothing, and an empty run is how a broken test reached main on 2026-09-20.) Expected: last line `... GREEN`, and `tests/test_parse_all_models.py` shows `tests=652`.

- [ ] **Step 5: Run both QA harnesses** on a copy (the Model page's data loader changed, so `CLAUDE.md` requires both), then delete the copy:

```bash
cp data/analysis.db /tmp/qa_copy.db
.venv/bin/python scripts/app_qa_sweep.py /tmp/qa_copy.db > /tmp/sweep.log 2>&1; echo "sweep exit $?"
.venv/bin/python scripts/chart_qa_render_all.py qa_output /tmp/qa_copy.db > /tmp/charts.log 2>&1; echo "charts exit $?"
rm -f /tmp/qa_copy.db*
grep -E "APP QA SWEEP|^FAIL" /tmp/sweep.log
```

Expected: the sweep's only FAILs are the two known ones (the NaN-bearing fail-point count and the DRAWN marker count — both pick "the newest 4,000 rows" by id). Any third FAIL is yours. On the production copy the findings check ends with its WARN (`this database predates the trim-pass capture`) — that is correct until the rebuild. Open two or three PNGs in `qa_output/` and look at them; `git status --porcelain` must show no modified tracked file.

- [ ] **Step 6: Tell James.** In `TRACKER.md` tick **B5** and add under it what shipped (engine, three analyzers, Model-page tab, Findings page, `scripts/refresh_findings.py`) and that B6 (the remaining analyzers) and B7 (the cut-length model) are next; add the day's line to "Done recently". In `BRING_TO_WORK.md`, add to the top of the 2026-09-18 section's "The sequence" a sixth step: *"6. Open **Findings** in the sidebar. The ranked list fills in by itself as each trim folder finishes."* In the spec, change the status line to `Status: capture built (2026-09-18); engine + first three analyzers planned in docs/superpowers/plans/2026-09-20-process-findings-engine.md`.

- [ ] **Step 7: Commit**

```bash
git add scripts/app_qa_sweep.py scripts/run_test_gate.py TRACKER.md BRING_TO_WORK.md docs/superpowers/specs/2026-09-17-process-recommendations-design.md
git commit -m "test: guard the findings engine in the sweep; the test gate becomes a tool"
```

---

## Self-review (done while writing)

- **Spec coverage.** Finding shape (model, machines, category, lever, lead time, expected gain, strength, units, evidence) → Task 2. Ranking = gain × annual volume, no-gain findings last by volume → Task 2 + the SQL order in Task 7. Two surfaces, one engine → Tasks 9 and 10. "It must be able to say nothing" → a negative test per analyzer (Tasks 4–6) plus the unfaithful-yardstick silence (Task 7). "Computed after an ingest… and cached" → Tasks 7–8. Evidence tiers: every finding carries its table, sample size and strength; the recipe history is cached for every model. **Not in this plan, deliberately:** evidence *charts* (text tables only in v1), a configurable volume threshold for the "full pack" tier, dismiss/snooze (the spec excludes it), and analyzers 2, 3, 5, 7, 8 and 10 of the catalogue — they are TRACKER item B6, one analyzer at a time on this frame.
- **Placeholders.** None: every code step carries its code, every fail-first step names its break and the test that must fail, and each break was run against the prototype on 2026-09-20.
- **Type consistency.** `analyze(model, tracks, laser_label)` for all three analyzers; `recipe_change` returns `(history, findings)`, `trim_effort` returns `(facts, findings)`, `ink_target` returns `findings` — the engine in Task 7 unpacks exactly those. `Finding.to_dict()` keys are the ones the tab (Task 9), the page (Task 10) and the manager (Task 7) read: `model, analyzer, category, lever, lever_label, lead_time, title, summary, expected_gain_points, units_per_year, annual_volume, n_units, strength_name, strength_value, evidence`.

## After this plan

Run `scripts/refresh_findings.py` on the home slice and read the list with James. Then TRACKER **B6** — the remaining analyzers, each a new file under `findings/analyzers/` with its negative test — and **B7**, the cut-length model, which gets its own design: James's stated goal is *the least trim that lands in spec*, and the recipe history this plan caches (8232-1's fixed 0.75 → 0.88 recipe; the 2019 change that took laser 2 from 84% to 32% and back) is its starting evidence.
