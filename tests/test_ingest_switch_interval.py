"""The ingest holds a fast GIL switch interval — and gives it back.

James's new laptop "kept freezing" during its first full ingest. The ingest
itself was never stuck: `Processor._process_parallel` runs up to four CPU-bound
Excel parsers, and at CPython's 5 ms default switch interval the Tk thread lost
nearly every GIL race to them. Measured over a real 1,308-file run with the V6
window open: 283 UI stalls over 60 ms, 48% of the run frozen. At 0.5 ms: 77
stalls, 12% frozen, for 2 seconds of extra wall clock in 68.

`sys.setswitchinterval` is PROCESS-WIDE, which is the whole risk of the fix, so
what these tests pin is the scope rather than the speed: it is 0.5 ms only
while an ingest is actually running, the caller's value comes back on every
exit path (success, failure result, exception), and the nested
`run_folders -> run_folder` call does not hand it back between folders while
the run is still going.

Tk-free and fast on purpose: empty tmp folders, no database, no widgets.
"""
import sys
import threading

import pytest

from laser_trim_analyzer.core import ingest_run
from laser_trim_analyzer.core.ingest_run import (
    INGEST_SWITCH_INTERVAL,
    run_folder,
    run_folders,
)

# Nothing else in the process uses this value: not the 0.005 interpreter
# default, not the ingest's 0.0005. "Restored" has to mean the caller's
# interval, not a plausible-looking constant.
CALLER_INTERVAL = 0.02


@pytest.fixture(autouse=True)
def caller_interval():
    """Stand in for an app that set its own interval, and clean up after.

    Autouse so a failing test cannot leak a fast interval into the rest of the
    suite. The double set is not superstition: CPython stores the interval
    truncated to whole microseconds, so a value that is not exactly on a
    microsecond drifts by one on every save/restore round trip (0.017 comes
    back as 0.016999). Settling on the stored value first means these tests
    fail for a lost interval, never for a rounding artefact.
    """
    before = sys.getswitchinterval()
    sys.setswitchinterval(CALLER_INTERVAL)
    sys.setswitchinterval(sys.getswitchinterval())
    try:
        yield sys.getswitchinterval()
    finally:
        sys.setswitchinterval(before)


def test_the_ingest_interval_is_fast_enough_to_matter():
    """0.5 ms — an order of magnitude under the 5 ms default it replaces."""
    assert INGEST_SWITCH_INTERVAL == pytest.approx(0.0005)
    assert INGEST_SWITCH_INTERVAL <= 0.001


def test_a_folder_run_holds_the_fast_interval_and_then_gives_it_back(
        tmp_path, caller_interval):
    seen = []
    res = run_folder(str(tmp_path), db=None, config=None,
                     on_phase=lambda text: seen.append(sys.getswitchinterval()))
    assert res.ok is True
    assert seen, "on_phase never fired — the observation window was empty"
    assert all(v == pytest.approx(INGEST_SWITCH_INTERVAL) for v in seen), seen
    assert sys.getswitchinterval() == caller_interval


def test_the_interval_comes_back_when_the_folder_is_unusable(tmp_path, caller_interval):
    """An offline share returns a failure RESULT, not an exception — and that
    early return is the easiest path to leak a process-wide setting down."""
    res = run_folder(str(tmp_path / "gone"), db=None, config=None)
    assert res.ok is False
    assert sys.getswitchinterval() == caller_interval


def test_the_interval_comes_back_when_the_batch_explodes(tmp_path, monkeypatch,
                                                         caller_interval):
    (tmp_path / "a.xls").write_bytes(b"junk")

    class _Boom:
        last_scan_stats = {}

        def __init__(self, *a, **k):
            pass

        def process_batch(self, *a, **k):
            raise RuntimeError("disk went away")
            yield  # pragma: no cover  (makes this a generator function)

    monkeypatch.setattr(ingest_run, "Processor", _Boom)
    res = run_folder(str(tmp_path), db=None, config=None)
    assert res.ok is False
    assert sys.getswitchinterval() == caller_interval


def test_the_interval_comes_back_when_the_run_raises(tmp_path, monkeypatch,
                                                     caller_interval):
    """run_folder is written never to raise, but `finally` is what makes that
    a guarantee instead of a hope."""
    def boom(folder):
        raise OSError("share vanished mid-walk")

    monkeypatch.setattr(ingest_run, "discover_excel_files", boom)
    with pytest.raises(OSError):
        run_folder(str(tmp_path), db=None, config=None)
    assert sys.getswitchinterval() == caller_interval


def test_a_multi_folder_run_stays_fast_between_folders(tmp_path, caller_interval):
    """The nested case: run_folders -> run_folder. If the inner call restored
    on its way out, the gap between folders — post-batch work, the callbacks,
    the next folder's walk — would run at the caller's slow interval again."""
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    between = []
    report = run_folders([str(a), str(b)], db=None, config=None,
                         on_folder_done=lambda r: between.append(sys.getswitchinterval()))
    assert report.ok is True
    assert len(between) == 2
    assert all(v == pytest.approx(INGEST_SWITCH_INTERVAL) for v in between), between
    assert sys.getswitchinterval() == caller_interval


def test_two_overlapping_ingests_do_not_leak_the_fast_interval(tmp_path, caller_interval):
    """Home's "process everything new" and the Process page's folder run are
    two buttons on two pages, and each disables only its own — so two ingests
    CAN be in flight on two worker threads. A guard that saved "the caller's
    interval" per run would save the OTHER run's 0.5 ms and hand it back as if
    it were the app's, leaving the whole process fast forever after the ingest
    ended. Here the first run finishes while the second is still going.
    """
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    first_inside = threading.Event()
    second_inside = threading.Event()
    first_finished = threading.Event()
    seen_by_second, waits = [], []

    def first_phase(_text):
        if not first_inside.is_set():
            first_inside.set()
            waits.append(second_inside.wait(5))

    def second_phase(_text):
        if not second_inside.is_set():
            second_inside.set()
            waits.append(first_finished.wait(5))
            seen_by_second.append(sys.getswitchinterval())

    def run(folder, on_phase):
        run_folder(str(folder), db=None, config=None, on_phase=on_phase)

    t1 = threading.Thread(target=run, args=(a, first_phase), daemon=True)
    t2 = threading.Thread(target=run, args=(b, second_phase), daemon=True)
    t1.start()
    assert first_inside.wait(5), "the first run never reached a callback"
    t2.start()
    t1.join(10)
    assert not t1.is_alive(), "the first run never finished"
    first_finished.set()
    t2.join(10)
    assert not t2.is_alive(), "the second run never finished"

    assert all(waits) and len(waits) == 2, "the two runs did not actually overlap"
    # The second run was still going when the first one ended: still fast.
    assert seen_by_second == [pytest.approx(INGEST_SWITCH_INTERVAL)]
    # And once BOTH are done, the app's own interval is back — not 0.5 ms.
    assert sys.getswitchinterval() == caller_interval


def test_an_empty_folder_list_still_gives_the_interval_back(caller_interval):
    report = run_folders([], db=None, config=None)
    assert report.ok is True
    assert sys.getswitchinterval() == caller_interval
