"""The status bar -- one quiet line at the foot of every page (option B, 2026-10-04).

James, with Task Manager TMOG open: "this is an example of a finished peice of solftware". Left to
right:

    ● Database OK   333 models   Newest file 29 Sep 2026   Drift watch current   70 files skipped
                                                          Processing new files · 34 of 120  (right)

  * the database's parts come from `status_data.load_status(db)`, on a worker. A part that failed is
    NAMED in the check colour where its number would be ("Models: could not count
    (OperationalError)"); a database that could not be read at all says so on its own, with the dot
    in the check colour. Nothing is ever drawn as a zero it was not;
  * the drift watch is V6App's (`drift_watch()`): "updating…" while the startup rebuild and catch-up
    run, "current" once they have, and what failed when they did not;
  * the run is V6App's run observer: the progress the Process page's two runs already receive,
    reported app-wide (`V6App.report_run_progress`) -- or, for a run that reports none (a re-grade,
    a findings refresh), its name. Nothing on the right when no run is in flight.

The database's parts are read again at startup, when a run ends, after the drift rebuild, and when
the Overview is shown (it reloads then) -- V6App asks; `refresh()` coalesces what arrives during a
load into one more load.

Thread discipline (CLAUDE.md rule 5): the worker only loads, and posts the result through the app's
UI dispatcher; every widget call is on the Tk thread.
"""
import logging
import threading
import time
from typing import Callable, Optional

import customtkinter as ctk

from laser_trim_analyzer.gui.v6 import status_data as sd
from laser_trim_analyzer.gui.v6.theme import ThemeManager

logger = logging.getLogger(__name__)

HEIGHT = 28                 # the bar, its top line included
DOT = 8                     # the health dot's diameter
# While a run is in flight the bar looks again this often: a run's thread can end without anyone
# saying so (NewFilesRun's crash path posts no unregister), and a run that reports no progress is
# named once QUIET_START has passed -- by then the Process page's first report (a quarter second)
# has landed, so its name never flashes in front of its numbers.
POLL_MS = 500
QUIET_START = 0.5
_clock = time.monotonic


class StatusBar(ctk.CTkFrame):
    def __init__(self, master, *, app, theme: ThemeManager,
                 loader: Optional[Callable] = None, **kwargs):
        super().__init__(master, height=HEIGHT, fg_color=theme.SURFACE, corner_radius=0, **kwargs)
        t = self.theme = theme
        self._app = app
        self._loader = loader or sd.load_status
        self._status: Optional[sd.Status] = None       # None until the first load lands
        self._loading = False
        self._again = False
        self._in_flight = False                        # a run was in flight at the last look
        self._poll_id = None
        self.pack_propagate(False)

        ctk.CTkFrame(self, height=1, fg_color=t.DIVIDER, corner_radius=0)\
            .pack(side="top", fill="x")
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(side="top", fill="both", expand=True, padx=t.SPACE_LG)
        # The database's words and the drift watch, packed first so they keep their room; the run
        # takes what is left on the right.
        self._left = ctk.CTkFrame(row, fg_color="transparent")
        self._left.pack(side="left", fill="y")
        self._dot = ctk.CTkFrame(self._left, width=DOT, height=DOT, corner_radius=DOT // 2,
                                 fg_color=t.TEXT_DISABLED)
        self._database = self._word(self._left)
        self._models = self._word(self._left)
        self._newest = self._word(self._left)
        self._drift = self._word(self._left)
        self._skipped = self._word(self._left)
        self._run = self._word(row)

        add = getattr(app, "add_run_listener", None)
        if add is not None:
            add(self._render_run)
        self._render_database()
        self.show_drift()
        self._render_run()

    def _word(self, parent) -> ctk.CTkLabel:
        t = self.theme
        # height=0: the words' own line height, inside the bar's 27 px -- CustomTkinter's default
        # 28 would ask for more than the bar has, and be cut.
        return ctk.CTkLabel(parent, text="", height=0, font=t.font(t.SIZE_CAPTION),
                            text_color=t.TEXT_SECONDARY, anchor="w")

    def _colour(self, tone: str) -> str:
        return self.theme.CHECK if tone == sd.CHECK else self.theme.TEXT_SECONDARY

    # ---- the database's parts --------------------------------------------------------------------
    def refresh(self) -> None:
        """Read the database's parts again, on a worker (Tk thread). One load at a time: refreshes
        asked for while one is in flight become ONE more load when it lands."""
        ui = getattr(self._app, "ui", None)
        if ui is None:                    # no dispatcher to hand a result back through: load here
            self.refresh_now()
            return
        if self._loading:
            self._again = True
            return
        self._loading, self._again = True, False
        db, loader = self._app.db, self._loader            # read here, handed to the worker

        def landed(status):
            def guarded():                                  # Tk thread, via the dispatcher
                try:
                    alive = self.winfo_exists()
                except Exception:
                    return                                  # the app is being torn down
                if alive:
                    self._landed(status)
            return guarded

        def work():
            try:
                status = loader(db)                         # load_status never raises...
            except Exception as exc:                        # ...a stand-in might: named, and
                logger.exception("Status bar: the load failed")     # the bar is never stuck
                status = sd.Status(failed={sd.PART_DATABASE: type(exc).__name__})
            ui.post(landed(status))
        threading.Thread(target=work, daemon=True).start()

    def refresh_now(self) -> None:
        """The same load on this thread, then drawn -- the test path."""
        self.show_status(self._loader(self._app.db))

    def _landed(self, status: sd.Status) -> None:
        self._loading = False
        self.show_status(status)
        if self._again:
            self.refresh()

    def show_status(self, status: sd.Status) -> None:
        """Draw one load (Tk thread)."""
        self._status = status
        self._render_database()

    def _render_database(self) -> None:
        t = self.theme
        s = self._status
        text, tone = sd.database_words(s)
        self._dot.configure(fg_color=(t.PASS_FG if tone == sd.OK else
                                      t.CHECK if tone == sd.CHECK else t.TEXT_DISABLED))
        self._database.configure(text=text, text_color=self._colour(tone))
        for label, words in ((self._models, sd.models_words(s)), (self._newest, sd.newest_words(s)),
                             (self._skipped, sd.skipped_words(s))):
            label.configure(text=words[0], text_color=self._colour(words[1]))
        self._place_left()

    # ---- the drift watch ----------------------------------------------------------------------
    def show_drift(self) -> None:
        """Draw the drift watch as V6App holds it (Tk thread)."""
        state, error = getattr(self._app, "drift_watch", lambda: (sd.DRIFT_CURRENT, None))()
        text, tone = sd.drift_words(state, error)
        self._drift.configure(text=text, text_color=self._colour(tone))
        self._place_left()

    def _place_left(self) -> None:
        """The left words in their fixed order, each only while it has something to say."""
        t = self.theme
        words = (self._database, self._models, self._newest, self._drift, self._skipped)
        for w in (self._dot,) + words:
            w.pack_forget()
        self._dot.pack(side="left", padx=(0, t.SPACE_SM))
        for w in words:
            if w.cget("text"):
                w.pack(side="left", padx=(0, t.SPACE_LG))

    # ---- the run in flight --------------------------------------------------------------------
    def _render_run(self) -> None:
        """V6App's run observer, and the bar's own look while a run is in flight (Tk thread): the
        run's progress on the right, or nothing. When the run has ended, the database's parts are
        read again -- the run changed them."""
        run = getattr(self._app, "run_progress", lambda: None)()
        was, self._in_flight = self._in_flight, run is not None
        if run is not None and not run.label and _clock() - run.started < QUIET_START:
            text = ""                       # its progress may be a quarter second away -- wait
        else:
            text = sd.run_words(run)
        if text:
            self._run.configure(text=text)
            if self._run.winfo_manager() == "":
                self._run.pack(side="right")
        elif self._run.winfo_manager():
            self._run.pack_forget()
            self._run.configure(text="")
        if run is not None:
            self._look_again()
        if was and run is None:
            self.refresh()

    def _look_again(self) -> None:
        if self._poll_id is None:
            self._poll_id = self.after(POLL_MS, self._tick)

    def _tick(self) -> None:
        self._poll_id = None
        try:
            self._render_run()
        except Exception:
            logger.exception("Status bar: the run line could not be drawn")

    def destroy(self):
        if self._poll_id is not None:
            try:
                self.after_cancel(self._poll_id)
            except Exception:
                pass
            self._poll_id = None
        super().destroy()
