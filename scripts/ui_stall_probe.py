"""UI-thread stall probe: what the user's hands feel, measured.

    python scripts/ui_stall_probe.py /path/to/COPY_of_analysis.db

Drives the real V6App programmatically (no clicks needed — Tk on macOS
ignores synthetic background clicks anyway) and runs a 20 ms heartbeat on
the Tk thread. Any gap between heartbeats longer than STALL_MS is a moment
the app could not respond to a click. A watchdog thread samples the MAIN
thread's Python stack during each stall, so the report says WHERE the UI
thread was, not just that it froze.

Written 2026-09-02 when James reported "the whole thing feels sluggish".
Every component measured in isolation was fast; only this probe showed the
truth — a 1.6 s single stall on model select and 1.1 s on every window
resize, both inside CustomTkinter's scrollbar redraw cascade plus
synchronous matplotlib draws. Keep using it: a page that feels slow gets
probed, not argued about.

Refuses the production database like the other harnesses (the app opens it
read-write at startup).
"""
import collections
import sys
import threading
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

STALL_MS = 60          # a gap this long is a missed frame the user can feel
HEARTBEAT_MS = 20


def _refuse_production(db_path: Path) -> None:
    if db_path.resolve() == (REPO / "data" / "analysis.db").resolve():
        raise SystemExit(
            f"FATAL | {db_path} is the PRODUCTION database; the app opens it "
            f"read-write.\n      |     cp data/analysis.db /tmp/qa_copy.db\n"
            f"      |     python scripts/ui_stall_probe.py /tmp/qa_copy.db")


def main(db_path: Path) -> int:
    _refuse_production(db_path)
    if not db_path.exists():
        print(f"FATAL | no database at {db_path}")
        return 1

    from laser_trim_analyzer.config import get_config
    from laser_trim_analyzer.database.manager import DatabaseManager
    from laser_trim_analyzer.gui.v6.app import V6App

    cfg = get_config()
    db = DatabaseManager(db_path)
    app = V6App(cfg, db=db, auto_train_on_first_run=False)
    app.geometry("1440x900+60+60")
    app.deiconify()
    app.update()

    def pump(seconds: float) -> None:
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            app.update()
            time.sleep(0.005)

    pump(3.0)                                    # let HOME settle

    main_ident = threading.main_thread().ident
    last = [time.perf_counter()]
    gaps: list = []
    stacks: collections.Counter = collections.Counter()
    armed = [False]

    def tick() -> None:
        now = time.perf_counter()
        gap_ms = (now - last[0]) * 1000
        if armed[0] and gap_ms > STALL_MS:
            gaps.append(round(gap_ms))
        last[0] = now
        app.after(HEARTBEAT_MS, tick)

    def watchdog() -> None:
        while True:
            time.sleep(0.05)
            if armed[0] and time.perf_counter() - last[0] > 0.15:
                frame = sys._current_frames().get(main_ident)
                if frame is not None:
                    tail = traceback.extract_stack(frame)[-5:]
                    stacks[" <- ".join(f"{Path(f.filename).name}:{f.name}"
                                       for f in tail)] += 1

    app.after(HEARTBEAT_MS, tick)
    threading.Thread(target=watchdog, daemon=True).start()

    worst_overall = 0

    def scenario(label: str, action, seconds: float) -> None:
        nonlocal worst_overall
        gaps.clear(); stacks.clear()
        last[0] = time.perf_counter(); armed[0] = True
        action()
        pump(seconds)
        armed[0] = False
        worst = max(gaps) if gaps else 0
        worst_overall = max(worst_overall, worst)
        print(f"{label:36s} stalls>{STALL_MS}ms: {len(gaps):3d}   "
              f"worst: {worst:5d} ms   total: {sum(gaps):5d} ms")
        for key, n in stacks.most_common(2):
            print(f"    x{n:<3d} {key[-150:]}")

    page = app.page_container.get_page("model")
    scenario("idle", lambda: None, 2.0)
    scenario("HOME refresh (on_show)", lambda: app.show_page("home"), 3.0)
    scenario("Investigate: select 8232-1",
             lambda: (app.show_page("model"), page._on_model_selected("8232-1")), 6.0)
    scenario("Investigate: select 6607", lambda: page._on_model_selected("6607"), 6.0)
    scenario("resize on Investigate", lambda: app.geometry("1500x940+60+60"), 1.5)
    app.show_page("home"); pump(1.0)
    scenario("resize on HOME", lambda: app.geometry("1380x860+60+60"), 1.5)
    app.destroy()

    print(f"\nWORST STALL: {worst_overall} ms   "
          f"(a click during a stall waits that long)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python scripts/ui_stall_probe.py /path/to/COPY_of_analysis.db")
        raise SystemExit(1)
    raise SystemExit(main(Path(sys.argv[1])))
