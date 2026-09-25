"""Spec 3d — Database Cleanup: scan health + category purge + reset skipped (real V5 port).

Destructive operations go through the verified db.preview_cleanup / db.execute_cleanup
methods (never re-implemented here) and are gated behind a confirm dialog.
"""
import threading
from datetime import datetime
from typing import Optional

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager
from laser_trim_analyzer.gui.v6.ui_dispatch import post_ui
from laser_trim_analyzer.gui.v6.widgets import blocks

# (UI key, checkbox label, execute_cleanup kwarg)
_CATEGORIES = [
    ("non_mps", "Non-MPS models (not in your MPS list)", "delete_non_mps"),
    ("suspect", "Suspect quality (flagged dirty)", "delete_suspect_quality"),
    ("unknown", "Unknown system/model", "delete_unknown"),
    ("error", "Error status", "delete_error_status"),
    ("no_tracks", "No tracks", "delete_no_tracks"),
    ("misclassified_ft", "Misclassified Final-Test", "delete_misclassified_ft"),
]


def build_cleanup_options(*, non_mps, before_date_enabled, date_str, suspect, unknown,
                          error, no_tracks, misclassified_ft, mps_models) -> Optional[dict]:
    """Build preview/execute_cleanup kwargs from UI state (pure port of V5
    _get_cleanup_options). Returns None when nothing is selected, when 'non-MPS'
    is checked but no MPS list is configured, or when the date is invalid."""
    opts = {
        "delete_non_mps": bool(non_mps),
        "mps_models": None,
        "delete_before_date": None,
        "delete_suspect_quality": bool(suspect),
        "delete_unknown": bool(unknown),
        "delete_error_status": bool(error),
        "delete_no_tracks": bool(no_tracks),
        "delete_misclassified_ft": bool(misclassified_ft),
    }
    if non_mps:
        opts["mps_models"] = list(mps_models or [])
        if not opts["mps_models"]:
            return None
    if before_date_enabled:
        try:
            opts["delete_before_date"] = datetime.strptime((date_str or "").strip(), "%Y-%m-%d")
        except ValueError:
            return None
    if not any([opts["delete_non_mps"], opts["delete_before_date"], opts["delete_suspect_quality"],
                opts["delete_unknown"], opts["delete_error_status"], opts["delete_no_tracks"],
                opts["delete_misclassified_ft"]]):
        return None
    return opts


def build_database_cleanup_section(parent, theme: ThemeManager, app) -> None:
    t = theme
    db = app.db

    ctk.CTkLabel(parent, justify="left", wraplength=640, anchor="w", font=t.font(t.SIZE_BODY),
                 text_color=t.TEXT_SECONDARY,
                 text=("Scan for dirty/contaminated records, then optionally purge by category. "
                       "Deletes are permanent and run through the same filters as the preview."))\
        .pack(side="top", fill="x", anchor="w", pady=(0, t.SPACE_MD))

    # wraplength: every job in this section reports here, and the longest reports are the ones
    # about what went WRONG (model names and all). A label that cannot wrap runs off the window.
    status = ctk.CTkLabel(parent, text="", justify="left", font=t.font(t.SIZE_CAPTION),
                          text_color=t.TEXT_SECONDARY, anchor="w", wraplength=640)
    status.pack(side="top", fill="x")

    def _async(work_fn):
        """Run `work_fn` off the Tk thread and show what it returned.

        Returns the thread, so a long job (the re-grade) can be registered
        with the app and stopped when the window closes.
        """
        def runner():
            try:
                msg = work_fn()
            except Exception as exc:
                msg = f"Error: {exc}"
            post_ui(app, lambda: status.winfo_exists() and status.configure(text=msg))
        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        return thread

    def _scan():
        status.configure(text="Scanning…")

        def work():
            health = db.scan_database_health()
            lines = [f"{health.get('total_dirty_records', 0)} records with issues"]
            for info in (health.get("issues") or {}).values():
                lines.append(f"  • {info['label']}: {info['count']}")
            return "\n".join(lines) if len(lines) > 1 else "Database is clean."
        _async(work)

    blocks.link_button(parent, t, "Scan database", _scan)\
        .pack(side="top", anchor="w", pady=(0, t.SPACE_MD))

    # Category checkboxes.
    cvars = {}
    for key, label, _kw in _CATEGORIES:
        var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(parent, text=label, variable=var, font=t.font(t.SIZE_BODY),
                        text_color=t.TEXT_PRIMARY, fg_color=t.ACCENT,
                        hover_color=t.ACCENT_HOVER, checkmark_color=t.TEXT_INVERSE
                        ).pack(side="top", anchor="w", pady=1)
        cvars[key] = var

    date_row = ctk.CTkFrame(parent, fg_color="transparent")
    date_row.pack(side="top", fill="x", pady=t.SPACE_XS)
    date_var = ctk.BooleanVar(value=False)
    ctk.CTkCheckBox(date_row, text="Before date (YYYY-MM-DD):", variable=date_var,
                    font=t.font(t.SIZE_BODY), text_color=t.TEXT_PRIMARY, fg_color=t.ACCENT,
                    hover_color=t.ACCENT_HOVER, checkmark_color=t.TEXT_INVERSE).pack(side="left")
    date_entry = ctk.CTkEntry(date_row, width=120, fg_color=t.SURFACE, border_color=t.BORDER,
                              text_color=t.TEXT_PRIMARY)
    date_entry.pack(side="left", padx=(t.SPACE_SM, 0))

    def _current_options():
        return build_cleanup_options(
            non_mps=cvars["non_mps"].get(), before_date_enabled=date_var.get(),
            date_str=date_entry.get(), suspect=cvars["suspect"].get(),
            unknown=cvars["unknown"].get(), error=cvars["error"].get(),
            no_tracks=cvars["no_tracks"].get(), misclassified_ft=cvars["misclassified_ft"].get(),
            mps_models=getattr(app.config.active_models, "mps_models", []))

    def _preview():
        opts = _current_options()
        if opts is None:
            status.configure(text="Select at least one category (non-MPS needs an MPS list; "
                                  "before-date needs a valid YYYY-MM-DD).")
            return
        status.configure(text="Previewing…")
        _async(lambda: (lambda p: f"Would delete {p['records_to_delete']} of {p['total_records']} "
                        f"records.")(db.preview_cleanup(**opts)))

    def _execute():
        opts = _current_options()
        if opts is None:
            status.configure(text="Select at least one category first.")
            return
        from tkinter import messagebox
        if not messagebox.askyesno("Confirm cleanup",
                                   "Permanently delete the selected records? This cannot be undone."):
            return
        status.configure(text="Deleting…")
        _async(lambda: (lambda r: f"Deleted {sum(r.values()) if isinstance(r, dict) else r} records.")
               (db.execute_cleanup(**opts)))

    btns = ctk.CTkFrame(parent, fg_color="transparent")
    btns.pack(side="top", fill="x", pady=(t.SPACE_SM, 0))
    blocks.link_button(btns, t, "Preview", _preview).pack(side="left")
    # Destructive: the CONFIRMATION dialog above (messagebox.askyesno, unchanged) is what
    # actually gates the delete. Still a link (ruling 3: actions use primary_button/link_button;
    # Settings has no teal-filled button at all), but in the CHECK coral, so the danger cue
    # survives beside the harmless teal "Preview" (final review, 2026-09-24).
    blocks.link_button(btns, t, "Clear selected", _execute, tone="check"
                       ).pack(side="left", padx=(t.SPACE_SM, 0))

    def _reset_skipped():
        from tkinter import messagebox

        # Count off-thread, then confirm on the Tk thread, then reset off-thread.
        def runner():
            try:
                count = db.count_skipped_files()
            except Exception as exc:
                post_ui(app, lambda: status.winfo_exists() and status.configure(text=f"Error: {exc}"))
                return
            def confirm_and_run():
                if not status.winfo_exists():
                    return
                if count == 0:
                    status.configure(text="No skipped files to reset.")
                    return
                if not messagebox.askyesno("Reset skipped files",
                                           f"Reset {count} skipped files so they get reprocessed next run?"):
                    return
                _async(lambda: f"Reset {db.reset_skipped_files()} skipped files.")
            post_ui(app, confirm_and_run)
        threading.Thread(target=runner, daemon=True).start()

    blocks.link_button(parent, t, "Reset skipped files", _reset_skipped)\
        .pack(side="top", anchor="w", pady=(t.SPACE_MD, 0))

    def _retry_unreadable():
        """Offer the files that FAILED TO READ again (2026-09-17).

        The narrow sibling of "Reset skipped files" above: that one re-offers
        every skipped file, including the 8,114 the app correctly decided are
        not test data. This one clears only the markers written because a file
        could not be parsed — what a parser upgrade needs, and the way back
        from the line HOME shows.

        Count off-thread → confirm on the Tk thread → clear off-thread, the
        same shape as every other button in this section.
        """
        from tkinter import messagebox

        def runner():
            try:
                count = db.count_failed_file_markers()
            except Exception as exc:
                post_ui(app, lambda: status.winfo_exists() and status.configure(text=f"Error: {exc}"))
                return

            def confirm_and_run():
                if not status.winfo_exists():
                    return
                if count == 0:
                    status.configure(text="No files are being skipped for being unreadable.")
                    return
                if not messagebox.askyesno(
                        "Retry unreadable files",
                        f"Offer {count:,} file(s) again that could not be read on an "
                        f"earlier run?\n\n"
                        f"They are re-parsed on the next \"Process everything new\". "
                        f"Any that still cannot be read are recorded again, so this "
                        f"is worth pressing after a parser upgrade and not before.\n\n"
                        f"Files skipped for NOT being test data, and duplicates "
                        f"already stored under another path, are left alone."):
                    return
                _async(lambda: f"{db.reset_failed_file_markers():,} file(s) will be "
                               f"read again on the next run.")
            post_ui(app, confirm_and_run)
        threading.Thread(target=runner, daemon=True).start()

    blocks.link_button(parent, t, "Retry unreadable files", _retry_unreadable)\
        .pack(side="top", anchor="w", pady=(t.SPACE_SM, 0))

    def _refresh_findings():
        """Recompute cached process findings for every model (Task 10b, 2026-09-20).

        The gap this closes: the post-ingest hook (Task 8) only computes
        findings for the models in each NEW batch it ingests. After the
        owner rebuilds his database from `main`, before this engine is
        pushed, most models would otherwise never get findings without a
        command line.

        This only recomputes derived, cached findings -- it never touches a
        stored measurement -- so unlike the destructive buttons above there
        is no confirmation dialog.

        One long job at a time (2026-09-14): a findings refresh reads the
        whole database while an ingest or re-grade might be writing it, so
        refuse BEFORE touching the database -- the same guard
        `_regrade_final_tests` above uses.
        """
        from laser_trim_analyzer.findings import engine as _findings

        busy = getattr(app, "active_run_name", lambda: None)()
        if busy:
            status.configure(
                text=f"{busy} is running — let it finish first. Findings "
                     f"read the same database it is writing.")
            return

        status.configure(text="Working out process findings for every "
                              "model… this can take a few minutes.")

        # A registry KEY only. refresh_findings takes no cancel token, so this run cannot be
        # stopped part-way; it is registered so that OTHER long runs refuse to start while it
        # reads the database.
        cancel = threading.Event()

        def work():
            try:
                report: dict = {}
                stored = _findings.refresh_findings(db, None, report)
                failed = report.get("failed_models") or {}
                partial = report.get("analyzer_errors") or {}
                msg = f"Process findings refreshed: {stored:,} findings"
                if report.get("models") is not None:
                    msg += f" across {report['models']:,} models"
                if not failed and not partial:
                    return msg + ". Open Findings in the sidebar."
                names = sorted(set(failed) | set(partial))
                shown = ", ".join(names[:8]) + (" …" if len(names) > 8 else "")
                return (f"{msg} — but {len(failed):,} model(s) could not be worked out at all and "
                        f"{len(partial):,} had an analyzer fail ({shown}). The log has the details; "
                        f"each model's Findings tab names what failed.")
            finally:
                # Posted, not called: the run registry is Tk-thread state
                # and this `finally` runs on the worker -- mirrors the
                # re-grade's own unregister below.
                drop = getattr(app, "unregister_ingest", None)
                if drop is not None:
                    post_ui(app, lambda: drop(cancel))

        thread = _async(work)
        register = getattr(app, "register_ingest", None)
        if register is not None and thread is not None:
            register(cancel, thread, "A findings refresh")

    blocks.link_button(parent, t, "Refresh process findings", _refresh_findings)\
        .pack(side="top", anchor="w", pady=(t.SPACE_SM, 0))

    def _recompute_statuses():
        """Re-grade Pass/Warning/Fail from stored track flags (M4, 2026-07-07).

        Preview (dry-run) off-thread → confirm with transition counts on the
        Tk thread → execute off-thread. Rows with NULL pass flags are never
        touched (they need Fix Missing Tracks first).
        """
        from tkinter import messagebox
        status.configure(text="Previewing status recompute…")

        def runner():
            try:
                preview = db.recompute_overall_statuses(dry_run=True)
            except Exception as exc:
                post_ui(app, lambda: status.winfo_exists() and status.configure(
                    text=f"Error: {exc}"))
                return

            def confirm_and_run():
                if not status.winfo_exists():
                    return
                if not preview["changed"]:
                    status.configure(text=(
                        f"Statuses already consistent — {preview['examined']} checked, "
                        f"{preview['skipped_null_flags']} skipped (missing pass flags)."))
                    return
                trans = ", ".join(f"{k}: {v}" for k, v in
                                  sorted(preview["transitions"].items()))
                if not messagebox.askyesno(
                        "Recompute unit statuses",
                        f"Re-grade {preview['changed']} of {preview['examined']} units "
                        f"from their stored linearity/sigma results?\n\n"
                        f"Transitions — {trans}\n\n"
                        f"{preview['skipped_null_flags']} units skipped (missing pass "
                        f"flags — run Fix Missing Tracks first).\n\n"
                        f"Linearity is zero-tolerance: linearity-FAIL units currently "
                        f"labeled Warning become FAIL. This rewrites overall_status "
                        f"only; measurements are untouched."):
                    return
                def do_execute():
                    res = db.recompute_overall_statuses(dry_run=False)
                    return (f"Re-graded {res['changed']} units "
                            f"({', '.join(f'{k}: {v}' for k, v in sorted(res['transitions'].items()))}). "
                            f"Refresh Dashboard/Triage to see updated yields.")
                _async(do_execute)
            post_ui(app, confirm_and_run)
        threading.Thread(target=runner, daemon=True).start()

    blocks.link_button(parent, t, "Recompute unit statuses", _recompute_statuses)\
        .pack(side="top", anchor="w", pady=(t.SPACE_SM, 0))

    def _fix_missing_tracks():
        """Re-parse records whose track measurements are missing.

        The prerequisite for 'Recompute unit statuses' above, which skips rows
        with NULL pass flags. Until now this tool only existed on the V5
        Compare page, so V6 told users to run something V6 did not have; the
        loop itself lives in core.track_repair and is shared by both.

        Count off-thread → confirm on the Tk thread → repair off-thread. The
        source workbooks live on the plant share, so off the work network
        every record reports 'source unreachable' rather than failing.
        """
        from tkinter import messagebox
        from laser_trim_analyzer.core.track_repair import repair_missing_tracks

        status.configure(text="Finding records with missing tracks…")

        def runner():
            try:
                n_ft = len(db.get_final_tests_missing_tracks())
                n_trim = len(db.get_trim_records_missing_tracks(linked_only=False))
            except Exception as exc:
                post_ui(app, lambda: status.winfo_exists() and status.configure(
                    text=f"Error: {exc}"))
                return

            def confirm_and_run():
                if not status.winfo_exists():
                    return
                if not (n_ft or n_trim):
                    status.configure(text="No records need fixing.")
                    return
                if not messagebox.askyesno(
                        "Fix missing tracks",
                        f"Re-parse {n_ft + n_trim} record(s) with missing track "
                        f"measurements ({n_ft} Final Test, {n_trim} Trim)?\n\n"
                        f"Each one is read again from its source workbook on the "
                        f"plant share and its tracks are rewritten. Off the work "
                        f"network the sources are unreachable and nothing is "
                        f"changed.\n\nThis can take several minutes."):
                    return

                def on_progress(done, total, phase):
                    post_ui(app, lambda: status.winfo_exists() and status.configure(
                        text=f"Re-parsing {phase} {done}/{total}…"))

                _async(lambda: repair_missing_tracks(
                    db=db, progress=on_progress, linked_only=False).summary())
            post_ui(app, confirm_and_run)
        threading.Thread(target=runner, daemon=True).start()

    blocks.link_button(parent, t, "Fix missing tracks", _fix_missing_tracks)\
        .pack(side="top", anchor="w", pady=(t.SPACE_SM, 0))

    # ---- Re-grade final tests (2026-09-13) --------------------------------
    regrade_cancel = {"event": None}

    def _regrade_final_tests():
        """Re-grade stored final tests against the window the sheet grades.

        Until 2026-09-13 the app graded every row of a final-test sweep,
        including the lead-in the station leaves ungraded — on 8232-1 that is
        a phantom 0.047 error against a ±0.010 band, and it failed 98.5% of
        the model's files at a station that passed them. The parser and the
        grader are fixed, so new files are right; the rows already stored are
        not, and this re-parses and re-grades them.

        Count off-thread → confirm on the Tk thread → re-grade off-thread, the
        same shape as Fix missing tracks above, with a Stop button because
        this one can run over 150,000 records. The loop lives in
        core.ft_regrade and is the SAME code path the processor grades with,
        so a re-graded row is indistinguishable from a freshly ingested one.
        """
        from tkinter import messagebox
        from laser_trim_analyzer.core.ft_regrade import (
            format_regrade_line, regrade_final_tests)

        # One long job at a time (2026-09-14) — the mirror of HOME's refusal.
        # Checked BEFORE the count, because the count itself is a query
        # against the database an ingest is busy writing.
        busy = getattr(app, "active_run_name", lambda: None)()
        if busy:
            status.configure(
                text=f"{busy} is running — stop it first, then start the "
                     f"re-grade. Two jobs over the plant share make each "
                     f"other slower than either one alone.")
            return

        status.configure(text="Counting final-test records to re-grade…")

        def runner():
            try:
                n_legacy = db.count_legacy_ft_verdicts()
            except Exception as exc:
                post_ui(app, lambda: status.winfo_exists() and status.configure(
                    text=f"Error: {exc}"))
                return

            def confirm_and_run():
                if not status.winfo_exists():
                    return
                if not n_legacy:
                    status.configure(
                        text="Every final-test record is already graded against "
                             "the station's window.")
                    return
                if not messagebox.askyesno(
                        "Re-grade final tests",
                        f"Re-grade {n_legacy:,} final-test record(s) that were "
                        f"graded before the ignore-window fix?\n\n"
                        f"Each one is read again from its source workbook on the "
                        f"plant share and re-graded on the rows the sheet itself "
                        f"grades. Off the work network the sources are "
                        f"unreachable and nothing is changed.\n\n"
                        f"RUN \"Process everything new\" FIRST, and do not start "
                        f"one while this is going. They share the database and "
                        f"the plant share, and together they make each other "
                        f"slower than either one alone.\n\n"
                        f"It works through the last two years' FAILURES first, "
                        f"so a run you stop early has still done the rows the "
                        f"screens read. Stop is safe — what is written stays "
                        f"written and starting again continues where it left "
                        f"off.\n\n"
                        f"EVERY final-test number moves afterwards: fail rates, "
                        f"escapes and overkills, the FOCUS list, and ML labels "
                        f"trained on final test."):
                    return

                cancel = threading.Event()
                regrade_cancel["event"] = cancel
                stop_btn.configure(state="normal")

                def on_progress(done, total, name, rate, eta):
                    # "12,480 of 151,375 · 2.6 files/s · about 14 h 50 min
                    # left" — the same words the command-line script prints,
                    # from the same formatter, because a run that takes hours
                    # is a run someone will check from both. "· Stop" names
                    # the button sitting next to this label.
                    line = format_regrade_line(done, total, rate, eta)
                    post_ui(app, lambda: status.winfo_exists() and status.configure(
                        text=f"{line} · Stop\n{name}"))

                def work():
                    try:
                        return regrade_final_tests(
                            db, progress=on_progress, cancel=cancel,
                            only_legacy=True, apply=True).summary()
                    finally:
                        regrade_cancel["event"] = None
                        post_ui(app, lambda: stop_btn.winfo_exists()
                                and stop_btn.configure(state="disabled"))
                        # Posted, not called: `_ingest_runs` is app state the
                        # Tk thread owns, and this `finally` is on the worker.
                        drop = getattr(app, "unregister_ingest", None)
                        if drop is not None:
                            post_ui(app, lambda: drop(cancel))

                # Registered with the app like an ingest is, for two reasons:
                # closing the window now stops a re-grade instead of killing
                # Tk out from under a batch mid-write, and HOME can see it and
                # refuse to start an ingest on top of it. Same registry, one
                # `active_run_name`, no second notion of "busy".
                thread = _async(work)
                register = getattr(app, "register_ingest", None)
                if register is not None and thread is not None:
                    register(cancel, thread, "A re-grade")
            post_ui(app, confirm_and_run)
        threading.Thread(target=runner, daemon=True).start()

    def _stop_regrade():
        event = regrade_cancel.get("event")
        if event is not None:
            event.set()
            status.configure(text="Stopping after the records already in flight…")

    regrade_row = ctk.CTkFrame(parent, fg_color="transparent")
    regrade_row.pack(side="top", fill="x", anchor="w", pady=(t.SPACE_SM, 0))
    blocks.link_button(regrade_row, t, "Re-grade final tests", _regrade_final_tests)\
        .pack(side="left", anchor="w")
    stop_btn = blocks.link_button(regrade_row, t, "Stop", _stop_regrade)
    stop_btn.configure(state="disabled")
    stop_btn.pack(side="left", padx=(t.SPACE_SM, 0))
