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


# What the engine calls each analyzer -> what a person would call it.
_ANALYZER_NAMES = {"recipe_change": "the recipe history", "ink_target": "the ink target",
                   "trim_effort": "what each cut buys", "limit_tables": "the limit tables",
                   "cut_setting": "the cut settings",
                   "pass_burden": "the multi-pass burden"}

# NOT COMPUTED, in one set of words: this tab and the model page's "Worth changing" section both
# say it (final review, 2026-09-24 -- the section used to read "nothing worth changing" instead).
NOT_COMPUTED_TEXT = ("No process findings have been computed for this model yet. They are worked "
                     "out after each ingest; Settings can also refresh them.")


def analyzer_name(key: str) -> str:
    """An analyzer's name as a person would say it ("the cut settings"); unknown keys as-is."""
    return _ANALYZER_NAMES.get(key, key)


class FindingsTab(ctk.CTkFrame):
    def __init__(self, master, theme: ThemeManager, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.theme = theme
        self._body = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self._body.pack(fill="both", expand=True)
        self.set_data(None)

    # ---- public ----
    def set_data(self, data: Optional[Dict[str, Any]]) -> None:
        t = self.theme
        for child in self._body.winfo_children():
            child.destroy()
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
            view.set_findings(findings)
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

    # ---- pieces ----
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
