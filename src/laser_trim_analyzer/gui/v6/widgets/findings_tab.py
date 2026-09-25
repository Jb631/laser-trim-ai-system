"""Model page tab: what the process data says about THIS model, and what to do about it.

Facts first (always shown -- they are measurements), then findings (only where
there is something a person could act on), then the recipe history and the rest of
what the analyzers measured -- the station limits, the laser comparison, the rework
count and, last, where the loss is made, directly above the Model page's Predictor
panel: spec ruling 2 (2026-09-24) sets loss_origin's AUC beside the predictor's, and
the predictor's stored AUC is quoted with it. A model with no findings reads as
"nothing to act on", never as a gap. Text only: no chart in v1, so nothing here
touches matplotlib or the chart QA harness.

The section texts are built by the module's `*_lines` functions -- pure, so every
rule they follow is tested without a window.

Facts cached by an older version (controller ruling, 2026-09-25). At work every
model's facts were written by the code before this pull, and only the models an
ingest touches are refreshed, so this tab WILL meet the keys it reads absent (the
work database's 319 cached rows hold none of the five new ones) or in an older shape
(57fdcdb's: machine_compare keyed by table with no window, rework_load flat with its
lasers pooled, loss_origin pooling limit tables, station_setup without the sampled
laser). Such a key never crashes the tab and never shows a number: it reads "Not
worked out yet by this version", naming what -- `not_worked_out`. The other keys the
tab reads kept their shape through the pull.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

import customtkinter as ctk

from laser_trim_analyzer.core.models import laser_label
from laser_trim_analyzer.gui.v6.theme import ThemeManager
from laser_trim_analyzer.gui.v6.widgets import blocks
from laser_trim_analyzer.gui.v6.widgets.findings_view import FindingsView


def _pct(v) -> str:
    return "—" if v is None else f"{v:.0f}%"


def _num(v) -> str:
    """A whole number for display. A value that is not there reads "—", never 0: nothing on
    this tab may turn "not recorded" into a measurement. (dict.get(key, 0) does not do this --
    its default only applies when the key is ABSENT, not when it is present and None.)"""
    return "—" if v is None else f"{v:,.0f}"


def _txt(v) -> str:
    return "—" if v in (None, "") else str(v)


def _auc(v) -> str:
    return "—" if v is None else f"{v:.2f}"


def _share(v) -> str:
    """A ratio as a percentage (0.41 -> "41%"); "—" when absent."""
    return "—" if v is None else f"{v:.0%}"


def _mon(month) -> str:
    try:
        return datetime.strptime(str(month)[:7], "%Y-%m").strftime("%b %Y")
    except (TypeError, ValueError):
        return "—"


def _span(months) -> str:
    months = [m for m in (months or []) if m]
    if not months:
        return "—"
    first, last = _mon(months[0]), _mon(months[-1])
    return first if first == last else f"{first} – {last}"


# What the engine calls each analyzer -> what a person would call it.
_ANALYZER_NAMES = {"recipe_change": "the recipe history", "ink_target": "the ink target",
                   "trim_effort": "what each cut buys", "limit_tables": "the limit tables",
                   "cut_setting": "the cut settings",
                   "pass_burden": "the multi-pass burden",
                   "setup_change": "the setting changes",
                   "machine_compare": "the laser comparison",
                   "loss_origin": "where the loss is made",
                   "station_setup": "the station limits comparison",
                   "rework_load": "the rework count"}

# NOT COMPUTED, in one set of words: this tab and the model page's "Worth changing" section both
# say it (final review, 2026-09-24 -- the section used to read "nothing worth changing" instead).
NOT_COMPUTED_TEXT = ("No process findings have been computed for this model yet. They are worked "
                     "out after each ingest; Settings can also refresh them.")


def analyzer_name(key: str) -> str:
    """An analyzer's name as a person would say it ("the cut settings"); unknown keys as-is."""
    return _ANALYZER_NAMES.get(key, key)


def failed_text(error: str) -> str:
    """FAILED, in one set of words: the load itself crashed (facelift F4). Never NOT_COMPUTED_TEXT
    -- "not worked out yet" over a crash is a failure looking like a result."""
    return (f"The process findings for this model could not be loaded ({error}). This is an "
            f"error, not a result — the log has the details.")


def _current_station(v) -> bool:
    return isinstance(v, dict) and (not v or "sampled_lasers" in v)


def _current_machine(v) -> bool:
    return isinstance(v, dict) and (not v or ("comparisons" in v and "window" in v))


def _current_rework(v) -> bool:
    return isinstance(v, dict) and (not v or "by_laser" in v)


def _current_loss(v) -> bool:
    return isinstance(v, dict) and all(isinstance(x, dict) and "limit_table" in x
                                       for x in v.values())


# The facts keys this tab reads whose shape changed in this version, each with the test of its
# CURRENT shape ({} -- computed, nothing to show -- reads the same in every version).
_SHAPES = {"station_setup": _current_station, "machine_compare": _current_machine,
           "rework_load": _current_rework, "loss_origin": _current_loss}


def not_worked_out(facts: Dict[str, Any]) -> List[str]:
    """The keys this tab reads that this version has not worked out for the model: absent (the
    cache predates the analyzer), None, or in an older shape. A key whose analyzer crashed is
    named with its error instead; a model with no trim tracks has nothing to work out."""
    if not facts.get("tracks"):
        return []
    errors = facts.get("errors") or {}
    return [key for key, current in _SHAPES.items()
            if key not in errors and not (facts.get(key) is not None and current(facts[key]))]


def not_worked_out_line(keys: List[str]) -> str:
    names = [analyzer_name(k) for k in keys]
    listed = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    return (f"Not worked out yet by this version — {listed}. Refresh findings (Settings) to see "
            f"{'it' if len(names) == 1 else 'them'}.")


def station_lines(ss: Dict[str, Any]) -> List[str]:
    """station_setup: its own note, and whose limits it compared."""
    lasers = ", ".join(ss.get("sampled_lasers") or []) or "The laser"
    return [f"{lasers} against final test: {_txt(ss.get('note'))}.",
            f"{_num(ss.get('matched_positions'))} positions measured by both stations, on the "
            "newest units linked to a final test."]


def machine_lines(mc: Dict[str, Any]) -> List[str]:
    """machine_compare: each comparable (limit table, shared months) period, its per-laser rates --
    a period before the window dated as such, never a recommendation."""
    window = mc.get("window") or {}
    lines = []
    for c in mc.get("comparisons") or []:
        parts = [_span(c.get("months"))]
        if c.get("graded_points") is not None:
            parts.append(f"{_num(c.get('graded_points'))}-point limit table")
        by_laser = c.get("by_laser") or {}
        parts += [f"{laser} {_pct((by_laser[laser] or {}).get('pass_pct'))} of "
                  f"{_num((by_laser[laser] or {}).get('n'))}" for laser in sorted(by_laser)]
        if c.get("in_window") is False:
            parts.append(f"before the {window.get('months', 24)} months to {_mon(window.get('last'))}, "
                         "so not a finding")
        lines.append(" · ".join(parts))
    return lines


def rework_lines(rw: Dict[str, Any]) -> List[str]:
    """rework_load: the model's counts, then each laser's verdict -- confirmed, or why not -- with
    both sizes, the p-value and the effect the verdict rests on."""
    lines = [f"{_num(rw.get('linked'))} final tests linked to this model's trim files in the last "
             f"year · {_num(rw.get('rework_unit_days'))} unit-days failed at the laser and passed "
             "final test"]
    if not rw.get("linked"):
        if rw.get("note"):
            lines.append(f"Nothing to test: {rw['note']}.")
        return lines
    per = rw.get("by_laser") or {}
    for laser in sorted(per):
        f = per[laser] or {}
        verdict = ("confirmed as hand trim" if f.get("confirmed")
                   else f"not confirmed: {_txt(f.get('note'))}")
        p = f.get("p_value")
        lines.append(
            f"{laser} · {verdict} · {_num(f.get('rework_ratio_n'))} reworked unit-days against the "
            f"{_num(f.get('control_top_third_n'))} untouched units with the largest laser errors "
            f"(of {_num(f.get('control_n'))}) · median final-test/laser error "
            f"{_share(f.get('median_ratio_rework'))} against "
            f"{_share(f.get('median_ratio_control_top_third'))} · effect "
            f"{_auc(f.get('effect_ratio'))} · p = {'—' if p is None else f'{p:.2g}'}")
    return lines


def loss_origin_lines(lo: Dict[str, Any]) -> List[str]:
    """loss_origin: one line per laser -- tracks, fails, both AUCs ("—" when uncomputable, never
    0) and the one limit table it was scored on."""
    lines = []
    for laser in sorted(lo):
        f = lo[laser] or {}
        line = (f"{laser} · {_num(f.get('n'))} tracks, {_num(f.get('fails'))} failed at the laser · "
                f"AUC {_auc(f.get('auc_error'))} from incoming linearity, "
                f"{_auc(f.get('auc_resistance'))} from incoming resistance")
        table = f.get("limit_table") or {}
        if table:
            line += f" · on the {_num(table.get('graded_points'))}-point limit table"
            if f.get("other_tables_n"):
                line += f" ({_num(f['other_tables_n'])} tracks on other tables left out)"
        lines.append(line)
    return lines


def predictor_line(data: Dict[str, Any]) -> Optional[str]:
    """The final-test predictor's own AUC as its last training stored it (the Model page reads
    it), set beside loss_origin's -- or why it cannot be. None when the caller supplied neither."""
    if data.get("predictor_auc_error"):
        return (f"The final-test predictor's own AUC could not be read "
                f"({_txt(data['predictor_auc_error'])}).")
    if "predictor_auc" not in data:
        return None
    auc = data["predictor_auc"]
    if auc is None:
        return ("No final-test predictor is trained for this model, so there is no predictor AUC "
                "to set beside these.")
    return (f"The final-test predictor's own AUC, for comparison: {auc:.2f} -- how well it ranks "
            "units at final test (the Predictor panel below), not where the loss starts.")


def _loss_reading() -> str:
    from laser_trim_analyzer.findings.analyzers import loss_origin as lo
    return (f"An AUC of 0.5 is no better than chance and 1.0 a perfect split. A finding needs "
            f"{lo.STRONG_AUC:.2f} over {lo.MIN_TRACKS} tracks with {lo.MIN_PER_OUTCOME} of each "
            "outcome; below that it is a fact, not a call.")


class FindingsTab(ctk.CTkFrame):
    def __init__(self, master, theme: ThemeManager, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.theme = theme
        self._body = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self._body.pack(fill="both", expand=True)
        self.set_data(None)

    # ---- public ----
    def set_data(self, data: Optional[Dict[str, Any]], *, failed: Optional[str] = None,
                 inactive: Optional[Dict[str, Any]] = None) -> None:
        """Three states, never two (facelift F4; the Model page's "Worth changing" section has had
        them since the final review):
          * FAILED -- the load itself crashed; `failed` names the error ("ExcType: message"). A
            check-tone banner and nothing else: whatever this tab drew before is not this load's.
          * NOT COMPUTED -- no facts cached (`data` None, or no "facts" in it).
          * computed -- the facts, then the findings, or "Nothing to act on" (the EMPTY state).

        `inactive` = {model: newest trim file} when the model is inactive (the Model page's own
        load, core/activity): its rows carry the quiet "Inactive · last trimmed" tag (F5).
        """
        t = self.theme
        for child in self._body.winfo_children():
            child.destroy()
        if failed:
            # Wraps to a frame rebuilt with it, never to _body, which lives as long as the tab
            # (blocks.wrap_to_width's rule).
            holder = ctk.CTkFrame(self._body, fg_color="transparent")
            holder.pack(fill="x")
            blocks.banner(holder, t, failed_text(failed), wrap_to=holder
                          ).pack(fill="x", pady=(0, t.SPACE_SM))
            return
        facts = (data or {}).get("facts")
        findings: List[Dict[str, Any]] = (data or {}).get("findings") or []
        if not facts:
            self._line(NOT_COMPUTED_TEXT, muted=True)
            return
        self._heading("What was measured")
        self._facts(facts)
        self._heading("What to do about it")
        if findings:
            # Same widget the Findings page uses (Task 7): no Open button -- we are already
            # on this model -- and empty groups hidden, since one model rarely has all four.
            view = FindingsView(self._body, t, on_open=None, include_empty=False)
            view.pack(fill="x")
            view.set_findings(findings, inactive=inactive)
        else:
            self._line("Nothing to act on. No analyzer found a lever worth pulling on this model — "
                       "that is a result, not a gap.", muted=True)
        tables = facts.get("limit_tables") or []
        if len(tables) > 1:             # one table is the unremarkable case; two is something to look at
            self._group_heading("Limit tables this model has been graded against", len(tables))
            for tab in tables:
                self._line(f"{laser_label(tab.get('system'))} · {_txt(tab.get('track'))} · "
                           f"{_num(tab.get('graded'))} graded points of {_num(tab.get('rows'))} rows · "
                           f"{_num(tab.get('n'))} tracks · {_txt(tab.get('first'))} → {_txt(tab.get('last'))} · "
                           f"{_pct(tab.get('trim_pass_pct'))} left the laser inside limits", muted=True)
        burden = facts.get("pass_burden") or {}
        if burden:
            self._group_heading("Cuts the recipe did not ask for (last year)", len(burden))
            for group, g in sorted(burden.items()):
                self._line(f"{group} · {_num(g.get('n'))} tracks · the recipe's normal is "
                           f"{_num(g.get('normal_cuts'))} cut(s) · "
                           f"{_pct(g.get('share_over_recipe'))} took more · "
                           f"{_num(g.get('unplanned_passes'))} unplanned passes "
                           f"({_num(g.get('unplanned_passes_per_100_tracks'))} per 100 tracks)",
                           muted=True)
        cuts = facts.get("cut_setting") or {}
        if cuts:
            self._group_heading("Cut settings this model has been run at", len(cuts))
            for group, g in sorted(cuts.items()):
                current = g.get("current_setting")
                mixed = g.get("days_with_more_than_one_setting_pct")
                self._line(f"{group} · {_num(g.get('n'))} tracks · {_txt(g.get('window'))}"
                           + (f" · {_pct(mixed)} of days ran more than one setting" if mixed is not None else ""),
                           muted=True)
                for s_ in g.get("settings") or []:
                    mark = "  ← running now" if current is not None and s_.get("setting") == current else ""
                    self._line(f"      cut {_txt(s_.get('setting'))} · {_num(s_.get('n'))} tracks · "
                               f"{_pct(s_.get('pass_pct'))} left the laser inside limits · "
                               f"median incoming {_num(s_.get('median_incoming_resistance'))} Ω · "
                               f"{_txt(s_.get('window'))}{mark}", muted=True)
        history = facts.get("recipe_history") or []
        if history:
            self._group_heading("Recipe history", len(history))
            for run in history:
                self._line(f"{laser_label(run.get('system'))} · {_txt(run.get('first'))} → "
                           f"{_txt(run.get('last'))} · {_txt(run.get('recipe'))} · {_num(run.get('n'))} tracks · "
                           f"{_pct(run.get('trim_pass_pct'))} left the laser inside limits · "
                           f"median incoming {_num(run.get('median_incoming_r'))} Ω", muted=True)
        # A key this version has not worked out (absent, None, or an older shape -- named at the
        # top by not_worked_out) or whose analyzer crashed (named with its error) draws no
        # section; {} is "computed, nothing comparable" -- no section either, the way the
        # sections above behave.
        stale = set(not_worked_out(facts)) | set(facts.get("errors") or {})

        def fresh(key):
            return None if key in stale else facts.get(key)

        self._section("Laser and final test limits", station_lines, fresh("station_setup"))
        self._section("Two lasers on the same test", machine_lines, fresh("machine_compare"))
        self._section("Hand trim after a laser fail (last year)", rework_lines, fresh("rework_load"))
        loss = fresh("loss_origin") or {}
        said = predictor_line(data or {})
        if loss:
            # Last: directly above the Model page's Predictor panel (spec ruling 2).
            lines = loss_origin_lines(loss)
            self._group_heading("Where the loss is made (last year)", len(lines))
            for line in lines:
                self._line(line, muted=True)
            self._line(_loss_reading(), muted=True)
            if said:
                self._line(said, muted=True)
        elif said and (data or {}).get("predictor_auc_error"):
            # A failed read is named with no loss section too (re-review, 2026-09-25: it was only
            # ever drawn inside it, so it vanished with the section). Same place, last. The AUC and
            # "no predictor" lines stay with the section: they are a comparison with it.
            self._line(said)

    # ---- pieces ----
    def _section(self, title: str, build, facts) -> None:
        """One fact section: `build(facts)` -> its lines, under a counted heading."""
        if not facts:
            return
        lines = build(facts)
        if not lines:
            return
        self._group_heading(title, len(lines))
        for line in lines:
            self._line(line, muted=True)

    def _heading(self, text: str) -> None:
        # A SECTION of the tab ("What to do about it") holds the FindingsView's group headers
        # (SIZE_HEADING, bold), so it can be no smaller than they are -- a caption-sized section
        # title above a heading-sized group title read upside down (final review, 2026-09-24).
        # More space above than below, so each section starts visibly.
        t = self.theme
        ctk.CTkLabel(self._body, text=text, font=t.font(t.SIZE_HEADING, "bold"),
                     text_color=t.TEXT_PRIMARY, anchor="w").pack(fill="x", pady=(t.SPACE_XL, t.SPACE_SM))

    def _group_heading(self, title: str, count: int) -> None:
        """A FACT section's own heading (step 1's spec, design doc §1 item 6): the same
        `blocks.group_header` the Findings page and this tab's own FindingsView use for a
        group of rows, so a count of "how many" reads the same everywhere in the app --
        instead of the bare `_heading()` label these four sections used to draw. `count` is
        always the number of rows the section lists right below it (one table, one recipe
        run, one cut-setting group, one pass-burden group), never a separately-computed
        number that could drift from what is actually drawn under it."""
        t = self.theme
        blocks.group_header(self._body, t, title, count
                            ).pack(fill="x", pady=(t.SPACE_XL, t.SPACE_SM))

    def _line(self, text: str, *, muted: bool = False) -> None:
        t = self.theme
        ctk.CTkLabel(self._body, text=text, font=t.font(t.SIZE_BODY), anchor="w", justify="left",
                     wraplength=900, text_color=t.TEXT_SECONDARY if muted else t.TEXT_PRIMARY
                     ).pack(fill="x", pady=(0, 2))

    def _facts(self, facts: Dict[str, Any]) -> None:
        # An analyzer that CRASHED is named, first. On this tab silence is a result ("nothing to
        # act on"), so a failure must never be allowed to look like one.
        errors = facts.get("errors") or {}
        for name in sorted(errors):
            self._line(f"Could not be worked out this time — {analyzer_name(name)} "
                       f"({_txt(errors[name])}). The rest of this tab is unaffected; the log has the details.")
        stale = not_worked_out(facts)
        if stale:
            self._line(not_worked_out_line(stale), muted=True)
        if not facts.get("tracks"):
            self._line("No laser trim tracks are stored for this model, so there is nothing to measure.",
                       muted=True)
            return
        effort = facts.get("trim_effort")
        if effort is None:
            if "trim_effort" in errors:
                return                  # said above -- do not blame the yardstick for a crash
            y = facts.get("yardstick") or {}
            agreement = y.get("agreement")
            need = y.get("min_agreement")
            self._line("Intermediate passes are not graded for this model. The grading yardstick is only "
                       f"trusted where it reproduces the app's own verdict on at least "
                       f"{_pct(None if need is None else need * 100)} of at least {_num(y.get('min_n'))} tracks; "
                       f"here it could be checked on {_num(y.get('n'))} tracks and agreed on "
                       f"{_pct(None if agreement is None else agreement * 100)}. Anything built on it "
                       "would be a guess.", muted=True)
            return
        for system, f in effort.items():
            cuts = " · ".join(f"{k} cut{'s' if k != '1' else ''}: {_num(v)}"
                              for k, v in sorted((f.get("cuts") or {}).items()))
            self._line(f"{laser_label(system)} — {_num(f.get('tracks_cut'))} tracks cut"
                       + (f"  ({cuts})" if cuts else ""))
            self._line(f"    arrive already inside linearity limits: {_pct(f.get('arrive_in_spec_pct'))} "
                       f"of {_num(f.get('graded_untrimmed_n'))}    ·    inside limits after the first cut: "
                       f"{_pct(f.get('in_limits_after_cut1_pct'))}", muted=True)
            if f.get("multi_cut_n"):
                self._line(f"    tracks given more than one cut ({_num(f['multi_cut_n'])}): "
                           f"{_pct(f.get('multi_in_limits_after_first_pct'))} inside limits after the first → "
                           f"{_pct(f.get('multi_in_limits_after_last_pct'))} after the last", muted=True)
