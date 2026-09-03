"""Model page — 'Trim vs Final Test' tab.

Test-pass vs trim-pass, escapes (passed trim but failed final test — a bad unit that got
through), overkills (failed trim but passed final test — unnecessarily rejected), agreement,
and the trim-pass-count distribution ('how many trim passes'). This is the Job-2 "are we
catching the problem at the right station?" view. Fed from db.get_model_trim_ft_agreement.
"""
from typing import Dict, List

import customtkinter as ctk

from laser_trim_analyzer.gui.v6.retire import retire
from laser_trim_analyzer.gui.v6.theme import ThemeManager


class TrimFtTab(ctk.CTkScrollableFrame):
    def __init__(self, master, theme: ThemeManager, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.theme = theme
        self._rendered: List[ctk.CTkBaseClass] = []

    def set_data(self, d: Dict) -> None:
        # Unmapped now, destroyed on idle time — the model switch must not wait
        # for the previous model's rows (see `gui/v6/retire.py`).
        retire(*self._rendered)
        self._rendered = []
        t = self.theme
        if not d:
            self._line("No trim / final-test data for this model.", t.TEXT_SECONDARY)
            return

        def pct(v):
            return "—" if v is None else f"{v:.1f}%"

        # --- Pass rates: trim vs final test ---
        self._header("Pass rates")
        self._kv("Trim linearity", f"{pct(d['trim_pass_rate'])}   ({d['trim_pass']}/{d['trim_total']})")
        self._kv("Final test", f"{pct(d['ft_pass_rate'])}   ({d['ft_pass']}/{d['ft_total']})")

        # --- Escapes / overkills (catch-at-the-right-station) ---
        self._header("Trim vs final-test agreement")
        if not d["linked"]:
            self._line("No units matched across both stations (need linked trim + final-test "
                       "records with a confident match).", t.TEXT_SECONDARY)
        else:
            self._kv("Units tested at both stations", str(d["linked"]))
            self._kv("Escapes — passed trim but FAILED final test", str(d["escapes"]),
                     color=t.TIER_OOC if d["escapes"] else None)
            self._kv("Overkills — failed trim but passed final test", str(d["overkills"]),
                     color=t.TIER_WARNING if d["overkills"] else None)
            self._kv("Agreement", f"{pct(d['agreement_rate'])}   ({d['agreements']}/{d['linked']})")
            def _serials(units, cap=25):
                # Numeric-aware sort (string sort gave '1, 10, 102, 13, 130' —
                # live-walk finding, 2026-07-08) and an EXPLICIT '+N more'
                # instead of a truncated list trailing off mid-comma.
                uniq = sorted(set(str(u) for u in units),
                              key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else 0, x))
                shown = ", ".join(uniq[:cap])
                if len(uniq) > cap:
                    shown += f"  (+{len(uniq) - cap} more — see the evidence pack export)"
                return shown
            if d["escape_units"]:
                self._line("Escaped serials (passed trim, failed FT): "
                           f"{_serials(d['escape_units'])}", t.TIER_OOC)
            if d["overkill_units"]:
                self._line("Overkill serials (failed trim, passed FT): "
                           f"{_serials(d['overkill_units'])}", t.TIER_WARNING)

        # --- Trim attempts (how many trim passes) ---
        self._header("Trim attempts (how many trim passes per unit)")
        avg = d.get("trim_pass_count_avg")
        self._kv("Average attempts per unit", "—" if avg is None else f"{avg:.2f}")
        dist = d.get("trim_pass_count_dist") or {}
        if dist:
            self._line("   ".join(f"{k}×: {v}" for k, v in sorted(dist.items())), t.TEXT_PRIMARY)

    # ---- render helpers ----
    def _header(self, text):
        t = self.theme
        lbl = ctk.CTkLabel(self, text=text, font=t.font(t.SIZE_BODY, "bold"),
                           text_color=t.TEXT_PRIMARY, anchor="w")
        lbl.pack(side="top", fill="x", pady=(t.SPACE_MD, t.SPACE_XS))
        self._rendered.append(lbl)

    def _kv(self, key, value, color=None):
        t = self.theme
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(side="top", fill="x", pady=1)
        ctk.CTkLabel(row, text=key, width=360, anchor="w", font=t.font(t.SIZE_BODY),
                     text_color=t.TEXT_SECONDARY).pack(side="left")
        ctk.CTkLabel(row, text=value, anchor="w", font=t.font(t.SIZE_BODY, "bold"),
                     text_color=color or t.TEXT_PRIMARY).pack(side="left")
        self._rendered.append(row)

    def _line(self, text, color):
        t = self.theme
        lbl = ctk.CTkLabel(self, text=text, justify="left", wraplength=720, anchor="w",
                           font=t.font(t.SIZE_CAPTION), text_color=color)
        lbl.pack(side="top", fill="x", pady=1)
        self._rendered.append(lbl)
