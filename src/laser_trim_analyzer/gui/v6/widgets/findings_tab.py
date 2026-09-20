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


def _num(v) -> str:
    """A whole number for display. A value that is not there reads "—", never 0: nothing on
    this tab may turn "not recorded" into a measurement. (dict.get(key, 0) does not do this --
    its default only applies when the key is ABSENT, not when it is present and None.)"""
    return "—" if v is None else f"{v:,.0f}"


def _txt(v) -> str:
    return "—" if v in (None, "") else str(v)


# What the engine calls each analyzer -> what a person would call it.
_ANALYZER_NAMES = {"recipe_change": "the recipe history", "ink_target": "the ink target",
                   "trim_effort": "what each cut buys", "limit_tables": "the limit tables"}


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
        tables = facts.get("limit_tables") or []
        if len(tables) > 1:             # one table is the unremarkable case; two is something to look at
            self._heading("LIMIT TABLES THIS MODEL HAS BEEN GRADED AGAINST")
            for tab in tables:
                self._line(f"{laser_label(tab.get('system'))} · {_txt(tab.get('track'))} · "
                           f"{_num(tab.get('graded'))} graded points of {_num(tab.get('rows'))} rows · "
                           f"{_num(tab.get('n'))} tracks · {_txt(tab.get('first'))} → {_txt(tab.get('last'))} · "
                           f"{_pct(tab.get('trim_pass_pct'))} left the laser inside limits", muted=True)
        history = facts.get("recipe_history") or []
        if history:
            self._heading("RECIPE HISTORY")
            for run in history:
                self._line(f"{laser_label(run.get('system'))} · {_txt(run.get('first'))} → "
                           f"{_txt(run.get('last'))} · {_txt(run.get('recipe'))} · {_num(run.get('n'))} tracks · "
                           f"{_pct(run.get('trim_pass_pct'))} left the laser inside limits · "
                           f"median incoming {_num(run.get('median_incoming_r'))} Ω", muted=True)

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
        # An analyzer that CRASHED is named, first. On this tab silence is a result ("nothing to
        # act on"), so a failure must never be allowed to look like one.
        errors = facts.get("errors") or {}
        for name in sorted(errors):
            self._line(f"Could not be worked out this time — {_ANALYZER_NAMES.get(name, name)} "
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

    def _card(self, f: Dict[str, Any]) -> None:
        t = self.theme
        card = ctk.CTkFrame(self._body, fg_color=t.CARD, corner_radius=8)
        card.pack(fill="x", pady=(0, t.SPACE_SM))
        ctk.CTkLabel(card, text=f.get("title", ""), font=t.font(t.SIZE_BODY, "bold"), anchor="w",
                     justify="left", wraplength=880, text_color=t.TEXT_PRIMARY
                     ).pack(fill="x", padx=t.SPACE_MD, pady=(t.SPACE_SM, 0))
        upy, points = f.get("units_per_year"), f.get("expected_gain_points")
        gain = (f"{points:+.1f} yield points ≈ {upy:,.0f} units a year"
                if upy is not None and points is not None else "no gain claimed")
        ctk.CTkLabel(card, text=f"{f.get('category', '')}  ·  lever: {f.get('lever_label', '')} "
                                f"({f.get('lead_time', '')})  ·  {gain}",
                     font=t.font(t.SIZE_CAPTION), anchor="w", text_color=t.ACCENT
                     ).pack(fill="x", padx=t.SPACE_MD)
        ctk.CTkLabel(card, text=f.get("summary", ""), font=t.font(t.SIZE_BODY), anchor="w", justify="left",
                     wraplength=880, text_color=t.TEXT_PRIMARY).pack(fill="x", padx=t.SPACE_MD, pady=(2, 0))
        strength = f.get("strength_value")
        ctk.CTkLabel(card, text=f"{f.get('strength_name', '')}: "
                                f"{'—' if strength is None else format(strength, '.2f')}  ·  "
                                f"rests on {_num(f.get('n_units'))} tracks",
                     font=t.font(t.SIZE_CAPTION), anchor="w", text_color=t.TEXT_SECONDARY
                     ).pack(fill="x", padx=t.SPACE_MD, pady=(0, t.SPACE_SM))
