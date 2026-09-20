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


def table(n_points: int = 12, half: float = BAND, span=(-10.0, 10.0), end_half=None):
    """A limit table as (positions, upper, lower): `n_points` evenly spaced across `span`, half-width
    `half` everywhere (or `end_half` on the first and last point -- a bowtie's wide ends)."""
    lo, hi = span
    pos = tuple(lo + (hi - lo) * i / (n_points - 1) for i in range(n_points))
    halves = [half] * n_points
    if end_half is not None:
        halves[0] = halves[-1] = end_half
    return pos, tuple(halves), tuple(-h for h in halves)


def on_table(track: TrackView, tab, name: str = "Track A") -> TrackView:
    """The same track, graded against `tab` (from `table()`), on track `name`."""
    from dataclasses import replace
    pos, up, lo = tab
    return replace(track, final_positions=pos, final_upper=up, final_lower=lo, track_name=name)


def table_era(first_id: int, start: datetime, n: int, tab, good_share: float, *, system: str = "B",
              name: str = "Track A", step_days: float = 1.0, cuts=(1.0,), r_in: float = 4500.0):
    """`n` tracks graded against `tab`, one every `step_days`, `good_share` of them passing."""
    out = []
    for k, d in enumerate(days(start, n, step_days)):
        good = (k % 100) < good_share * 100
        ps = tuple((0.8 if (good and j == len(cuts) - 1) else 1.5, c) for j, c in enumerate(cuts))
        out.append(on_table(make_track(first_id + k, date=d, passes=ps, system=system, r_in=r_in), tab, name))
    return out
