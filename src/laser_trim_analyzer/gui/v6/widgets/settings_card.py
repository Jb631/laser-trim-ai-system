"""Spec 3d — SettingsCard: collapsible section host."""
import customtkinter as ctk

from laser_trim_analyzer.gui.v6.theme import ThemeManager


class SettingsCard(ctk.CTkFrame):
    def __init__(self, master, theme: ThemeManager, title: str, expanded: bool = False, **kwargs):
        super().__init__(master, fg_color=theme.CARD, corner_radius=theme.RADIUS_MD, **kwargs)
        self.theme = theme
        self._expanded = False
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(side="top", fill="x")
        self._title = ctk.CTkLabel(header, text=title, font=theme.font(theme.SIZE_HEADING, "bold"),
                                   text_color=theme.TEXT_PRIMARY, anchor="w")
        self._title.pack(side="left", fill="x", expand=True, padx=theme.SPACE_MD, pady=theme.SPACE_SM)
        self._chevron = ctk.CTkButton(header, text="▾", width=32, fg_color="transparent",
                                      hover_color=theme.ELEVATED, text_color=theme.TEXT_SECONDARY,
                                      font=theme.font(theme.SIZE_HEADING), command=self.toggle,
                                      corner_radius=theme.RADIUS_SM)
        self._chevron.pack(side="right", padx=theme.SPACE_MD, pady=theme.SPACE_XS)
        for w in (header, self._title):
            w.bind("<Button-1>", lambda e: self.toggle())
        self._body = ctk.CTkFrame(self, fg_color="transparent")
        if expanded:
            self.toggle()

    def body_frame(self) -> ctk.CTkFrame:
        return self._body

    def toggle(self) -> None:
        if self._expanded:
            self._body.pack_forget()
            self._chevron.configure(text="▾")
        else:
            self._body.pack(side="top", fill="x", padx=self.theme.SPACE_MD,
                            pady=(0, self.theme.SPACE_SM))
            self._chevron.configure(text="▴")
        self._expanded = not self._expanded
