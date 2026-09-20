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
