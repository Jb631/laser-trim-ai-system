"""Final-test <-> trim matching: the two matchers, side by side, and their repair/rematch
helpers.

There are TWO independent matching implementations here, on purpose, not by oversight:
`_find_matching_trim` runs per-record at Final Test save time; `rematch_unlinked_final_tests`
runs in bulk after a batch, for FT records a same-batch or earlier trim couldn't have matched yet.
The review that ordered this move noted they "have already drifted apart twice" -- this step
moves them next to each other, unchanged, and does NOT unify them. `rematch_final_tests` (a full
re-evaluation of every FT record's link) and `backfill_trim_file_times` (recovers trim clock-times
so same-day re-trim attempts order correctly) are the repair/rematch helpers named alongside them.

Moved out of `database/manager.py` (2026-09-25, C2 Task 5, step 4) -- a pure move, byte-identical
method bodies (see the AST proof in this step's commit). `DatabaseManager` inherits this mixin
too, alongside `MigrationsMixin` and `SpecsMixin`, so every existing call site
(`db.rematch_final_tests()`, `db._find_matching_trim(...)` from `_save_final_test_in`, which
STAYS in manager.py as an ingest-speed Task 7 save body, etc.) is unchanged -- `self.` dispatch
resolves through the instance's MRO regardless of which file defines the method.
"""
from datetime import datetime, time, timedelta
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from laser_trim_analyzer.database.manager import logger
from laser_trim_analyzer.database.models import AnalysisResult as DBAnalysisResult


class FtMatchingMixin:
    """The final-test <-> trim matchers (`_find_matching_trim`, `rematch_unlinked_final_tests`)
    and their repair/rematch helpers (`backfill_trim_file_times`, `rematch_final_tests`), plus the
    normalization/confidence helpers only they use.

    Inherited by `DatabaseManager`. `self.session()` and other `self.`-dispatched calls resolve
    through the instance, defined elsewhere on `DatabaseManager` (or its other mixins) -- not
    through this module's own globals.
    """

    @staticmethod
    def _normalize_serial(serial: str) -> str:
        """
        Normalize a serial number for fuzzy matching (selective).

        Handles common formatting differences between trim and FT files:
        - Strip leading zeros (007 -> 7)
        - Lowercase
        - Strip whitespace
        - Remove common prefixes (sn, s/n, #)
        - Strip known track-position suffixes only (A/B for dual-track,
          P/R for primary/redundant, T for test)
        - Do NOT strip other letters (25D, 31L stay as-is since they may
          be meaningful serial identifiers)
        """
        import re
        s = serial.lower().strip()
        s = re.sub(r'^(sn|s/n|s\.n\.|#)\s*', '', s)
        # Strip only known track-indicator suffixes
        s = re.sub(r'^(\d+)[abprt]$', r'\1', s)
        s = s.lstrip('0') or '0'
        return s

    @staticmethod
    def _normalize_serial_aggressive(serial: str) -> str:
        """
        Aggressively normalize a serial number — strips ALL trailing letters.

        Used as a fallback when selective normalization fails to find a match.
        May produce false matches (e.g. 25D matches 25E) but increases recall.
        """
        import re
        s = serial.lower().strip()
        s = re.sub(r'^(sn|s/n|s\.n\.|#)\s*', '', s)
        s = re.sub(r'^(\d+)[a-z]$', r'\1', s)
        s = s.lstrip('0') or '0'
        return s

    @staticmethod
    def _normalize_model(model: str) -> str:
        """
        Normalize a model number to its base form for variant matching.

        Strips trailing letter suffixes that indicate product variants:
        - 8275A, 8275B, 8275C → 8275
        - 8508-A, 8508-B → 8508
        - 7280-1-CT, 7280-1-AB → 7280-1

        Strips leading zeros in hyphenated suffixes:
        - 2475-08 → 2475-8
        - 8867-01 → 8867-1

        Does NOT strip numeric suffixes (8340-1 stays 8340-1) since
        those are distinct model configurations.
        """
        import re
        if not model:
            return model
        # Strip leading zeros in hyphenated numeric suffixes: "2475-08" → "2475-8"
        s = re.sub(r'-0+(\d)', r'-\1', model)
        # Strip trailing letter-only variant: "8275A" → "8275"
        s = re.sub(r'^(\d+)[A-Za-z]$', r'\1', s)
        # Strip trailing hyphen + letter(s) variant: "8508-A" → "8508", "7280-1-CT" → "7280-1"
        s = re.sub(r'^(\d+(?:-\d+)*)-[A-Za-z]+$', r'\1', s)
        # Strip trailing letters glued to a hyphenated numeric suffix:
        # "7953-1A" → "7953-1" (2026-07-13: FT files say "7953-1", trim files
        # say "7953-1A"/"7953-1B" — 197 recent FT records unlinkable without this)
        s = re.sub(r'^(\d+(?:-\d+)+)[A-Za-z]+$', r'\1', s)
        return s

    def _find_matching_trim(
        self,
        session: Session,
        model: Optional[str],
        serial: Optional[str],
        test_date: Optional[datetime]
    ) -> Tuple[Optional[int], Optional[float], Optional[int], Optional[str]]:
        """
        Find the matching trim result for a final test.

        Logic:
        1. Exact model + exact serial match (case-insensitive) — highest confidence
        2. Exact model + fuzzy serial match (strip zeros, prefixes, track suffixes)
        3. Normalized model + fuzzy serial match (8275A trim matches 8275 FT)

        Among candidates the LATEST trim wins, and "latest" is resolved down to
        the clock time: a unit that fails linearity is re-trimmed until it
        passes, so several attempts share one calendar date and only the last
        one is the disposition the unit carried to final test. Linking an
        earlier failing attempt is what inflated the "overkill" metric.

        The window is bounded by calendar DATE, not by the raw timestamps —
        final-test records are stored at midnight, so a same-day trim at 14:30
        must still match (7,263 real linked rows are same-day).

        Returns:
            Tuple of (trim_id, confidence, days_since_trim, match_method)
        """
        from laser_trim_analyzer.utils.constants import FINAL_TEST_MAX_DAYS_FROM_TRIM

        if not model or not serial or not test_date:
            return None, None, None, None

        serial_clean = serial.lower().strip()
        test_day = test_date.replace(hour=0, minute=0, second=0, microsecond=0)
        # Half-open [cutoff, next midnight): keeps the file_date index usable.
        cutoff_date = test_day - timedelta(days=FINAL_TEST_MAX_DAYS_FROM_TRIM)
        before_date = test_day + timedelta(days=1)

        def _days_since(trim_date: datetime) -> int:
            """Trim→FT age in whole days, immune to the trim's clock time."""
            return (test_day - trim_date.replace(
                hour=0, minute=0, second=0, microsecond=0)).days

        # Attempt 1: Exact model + exact serial match (case-insensitive)
        candidates = (
            session.query(DBAnalysisResult)
            .filter(
                DBAnalysisResult.model == model,
                func.lower(DBAnalysisResult.serial) == serial_clean,
                DBAnalysisResult.file_date.isnot(None),
                DBAnalysisResult.file_date < before_date,
                DBAnalysisResult.file_date >= cutoff_date,
            )
            .order_by(desc(DBAnalysisResult.file_date), desc(DBAnalysisResult.id))
            .limit(5)
            .all()
        )

        if candidates:
            match = candidates[0]
            days_diff = _days_since(match.file_date)
            confidence = self._calculate_match_confidence(days_diff, exact_serial=True)
            return match.id, confidence, days_diff, "exact"

        # Attempt 2: Exact model + fuzzy serial match
        ft_serial_norm = self._normalize_serial(serial)

        model_trims = (
            session.query(DBAnalysisResult.id, DBAnalysisResult.serial, DBAnalysisResult.file_date)
            .filter(
                DBAnalysisResult.model == model,
                DBAnalysisResult.file_date.isnot(None),
                DBAnalysisResult.file_date < before_date,
                DBAnalysisResult.file_date >= cutoff_date,
            )
            .order_by(desc(DBAnalysisResult.file_date), desc(DBAnalysisResult.id))
            .all()
        )

        for trim_id, trim_serial, trim_date in model_trims:
            if trim_serial and self._normalize_serial(trim_serial) == ft_serial_norm:
                days_diff = _days_since(trim_date)
                confidence = self._calculate_match_confidence(days_diff, exact_serial=False)
                logger.debug(
                    f"Fuzzy match: FT serial '{serial}' → trim serial '{trim_serial}' "
                    f"(normalized: '{ft_serial_norm}'), {days_diff} days"
                )
                return trim_id, confidence, days_diff, "fuzzy_serial"

        # Attempt 2b: Exact model + aggressively normalized serial (strips all trailing letters)
        ft_serial_aggressive = self._normalize_serial_aggressive(serial)
        if ft_serial_aggressive != ft_serial_norm:
            for trim_id, trim_serial, trim_date in model_trims:
                if trim_serial and self._normalize_serial_aggressive(trim_serial) == ft_serial_aggressive:
                    days_diff = _days_since(trim_date)
                    confidence = self._calculate_match_confidence(days_diff, exact_serial=False) * 0.90
                    logger.debug(
                        f"Aggressive fuzzy match: FT serial '{serial}' -> trim serial '{trim_serial}' "
                        f"(aggressive norm: '{ft_serial_aggressive}'), {days_diff} days"
                    )
                    return trim_id, confidence, days_diff, "fuzzy_serial_aggressive"

        # Attempt 3: Model variant matching — normalize model on both sides
        # This handles cases like FT model "8275" matching trim model "8275A"
        # or FT model "8508" matching trim model "8508-A"
        ft_model_norm = self._normalize_model(model)

        if ft_model_norm != model:
            # FT model itself has a suffix — try base model in trim
            variant_trims = (
                session.query(DBAnalysisResult.id, DBAnalysisResult.serial,
                              DBAnalysisResult.file_date, DBAnalysisResult.model)
                .filter(
                    DBAnalysisResult.model == ft_model_norm,
                    DBAnalysisResult.file_date.isnot(None),
                    DBAnalysisResult.file_date < before_date,
                    DBAnalysisResult.file_date >= cutoff_date,
                )
                .order_by(desc(DBAnalysisResult.file_date), desc(DBAnalysisResult.id))
                .all()
            )
            for trim_id, trim_serial, trim_date, trim_model in variant_trims:
                if trim_serial and self._normalize_serial(trim_serial) == ft_serial_norm:
                    days_diff = _days_since(trim_date)
                    confidence = self._calculate_match_confidence(days_diff, exact_serial=False, model_variant=True)
                    logger.debug(
                        f"Model variant match: FT {model}/{serial} → trim {trim_model}/{trim_serial} "
                        f"(normalized model: '{ft_model_norm}'), {days_diff} days"
                    )
                    return trim_id, confidence, days_diff, "model_variant"

        # Try reverse: trim has variant suffixes, FT has base model
        # Find all trim models that normalize to our FT model
        # Use LIKE to find variants efficiently (e.g. "8275%" for FT model "8275")
        variant_trims = (
            session.query(DBAnalysisResult.id, DBAnalysisResult.serial,
                          DBAnalysisResult.file_date, DBAnalysisResult.model)
            .filter(
                DBAnalysisResult.model.like(f"{model}%"),
                DBAnalysisResult.model != model,  # Skip exact (already tried)
                DBAnalysisResult.file_date.isnot(None),
                DBAnalysisResult.file_date < before_date,
                DBAnalysisResult.file_date >= cutoff_date,
            )
            .order_by(desc(DBAnalysisResult.file_date), desc(DBAnalysisResult.id))
            .all()
        )

        for trim_id, trim_serial, trim_date, trim_model in variant_trims:
            # Verify this is actually a variant (normalizes to same base)
            if self._normalize_model(trim_model) != ft_model_norm:
                continue
            if trim_serial and self._normalize_serial(trim_serial) == ft_serial_norm:
                days_diff = _days_since(trim_date)
                confidence = self._calculate_match_confidence(days_diff, exact_serial=False, model_variant=True)
                logger.debug(
                    f"Model variant match: FT {model}/{serial} → trim {trim_model}/{trim_serial} "
                    f"(base model: '{ft_model_norm}'), {days_diff} days"
                )
                return trim_id, confidence, days_diff, "model_variant"

        return None, None, None, None

    @staticmethod
    def _calculate_match_confidence(days_diff: int, exact_serial: bool = True,
                                     model_variant: bool = False) -> float:
        """Calculate match confidence based on time proximity and match quality.

        Confidence bands:
        - Exact model + exact serial, same week: 0.93-1.00
        - Exact model + fuzzy serial, same week: 0.84-0.90
        - Model variant + fuzzy serial, same week: 0.71-0.77
        - Any match beyond 30 days: drops significantly
        """
        # Time-based confidence. Decay beyond 30 days is 0.002/day so the
        # scale spans the full 180-day match window (0.40 ≈ 180d); the old
        # 0.007/day rate hit the 0.40 floor by day ~73 and couldn't tell a
        # 75-day link from a 175-day one.
        if days_diff <= 7:
            time_conf = 1.0 - (days_diff * 0.01)
        elif days_diff <= 30:
            time_conf = 0.9 - ((days_diff - 7) * 0.01)
        else:
            time_conf = 0.7 - ((days_diff - 30) * 0.002)

        # Match quality penalties (applied multiplicatively)
        if not exact_serial:
            time_conf *= 0.90  # was 0.95 — fuzzy serial is less certain

        if model_variant:
            time_conf *= 0.85  # model variant adds uncertainty

        return max(0.40, time_conf)

    def backfill_trim_file_times(self, chunk_size: int = 5000) -> Dict[str, int]:
        """Recover the clock time for trim rows stored before the parser kept it.

        Until 2026-08-30 `_extract_date_from_filename` parsed only the calendar
        date and threw away the time sitting in the same filename, so every
        same-day re-trim attempt tied on `file_date`. `_find_matching_trim` then
        linked an arbitrary attempt — in practice the earliest-ingested — and a
        unit that was re-trimmed into spec before final test scored as a trim
        "overkill". This re-reads the time out of `filename` for rows still at
        midnight and rewrites `file_date` in place.

        Idempotent: rows already carrying a time, and rows whose filename has no
        time to recover, are left alone. Callers that care about link accuracy
        should follow this with `rematch_final_tests()`.

        Returns counts: scanned, updated, no_time_in_filename.
        """
        from laser_trim_analyzer.core.parser import ExcelParser

        parser = ExcelParser()
        stats = {"scanned": 0, "updated": 0, "no_time_in_filename": 0}

        with self._write_lock:
            with self.session() as session:
                # Only rows sitting exactly at midnight can be missing a time.
                base = (session.query(DBAnalysisResult.id, DBAnalysisResult.filename,
                                      DBAnalysisResult.file_date)
                        .filter(DBAnalysisResult.file_date.isnot(None),
                                func.strftime('%H:%M:%S', DBAnalysisResult.file_date)
                                == '00:00:00')
                        .order_by(DBAnalysisResult.id))
                offset = 0
                while True:
                    rows = base.limit(chunk_size).offset(offset).all()
                    if not rows:
                        break
                    updates = []
                    for row_id, filename, file_date in rows:
                        stats["scanned"] += 1
                        stamp = parser._extract_date_from_filename(filename or "")
                        # Trust only the TIME: the filename date can disagree with
                        # a date taken from inside the workbook, and re-dating rows
                        # is not this method's job.
                        if stamp is None or stamp.time() == time(0, 0):
                            stats["no_time_in_filename"] += 1
                            continue
                        updates.append({"id": row_id, "file_date": file_date.replace(
                            hour=stamp.hour, minute=stamp.minute, second=stamp.second)})
                    if updates:
                        session.bulk_update_mappings(DBAnalysisResult, updates)
                        stats["updated"] += len(updates)
                    session.commit()
                    # Updated rows drop out of the midnight filter, so only the
                    # skipped ones remain ahead of the cursor.
                    offset += len(rows) - len(updates)

        logger.info("Trim time backfill: %d scanned, %d updated, %d had no time in filename",
                    stats["scanned"], stats["updated"], stats["no_time_in_filename"])
        return stats

    def rematch_final_tests(self) -> Dict[str, int]:
        """
        Re-run matching for all Final Test records against current trim data.

        This is useful when trim files are imported after Final Test files,
        or when trim data has been updated.

        Returns:
            Dict with counts: new_matches, updated_matches, unchanged, total
        """
        from laser_trim_analyzer.database.models import (
            FinalTestResult as DBFinalTestResult,
        )

        stats = {"new_matches": 0, "updated_matches": 0, "unchanged": 0, "total": 0}

        with self._write_lock:
            with self.session() as session:
                # Get all Final Test records
                final_tests = session.query(DBFinalTestResult).all()
                stats["total"] = len(final_tests)

                for ft in final_tests:
                    # Get test date (prefer file_date, fall back to test_date)
                    test_date = ft.file_date or ft.test_date

                    # Find matching trim
                    new_trim_id, new_confidence, new_days, new_method = self._find_matching_trim(
                        session, ft.model, ft.serial, test_date
                    )

                    # Check if match changed
                    if new_trim_id != ft.linked_trim_id:
                        if ft.linked_trim_id is None and new_trim_id is not None:
                            stats["new_matches"] += 1
                        elif ft.linked_trim_id is not None and new_trim_id is not None:
                            stats["updated_matches"] += 1

                        # Update the record
                        ft.linked_trim_id = new_trim_id
                        ft.match_confidence = new_confidence
                        ft.days_since_trim = new_days
                        ft.match_method = new_method
                    else:
                        stats["unchanged"] += 1

                session.commit()
                logger.info(
                    f"Rematch complete: {stats['new_matches']} new, "
                    f"{stats['updated_matches']} updated, {stats['unchanged']} unchanged"
                )

        return stats

    def rematch_unlinked_final_tests(self, models=None) -> Dict[str, int]:
        """Link-only rematch pass over FT records that have NO trim link yet.

        Why this exists (2026-07-13): matching runs at FT save time, so an FT
        file processed in the same batch as — or any batch before — its trim
        file finds nothing and stays NULL forever. Nothing ever retried. This
        runs automatically after every processing batch.

        `models`: restrict the pass to FT records for these models (the models
        whose trims were just saved). A late trim can only create links for
        the models in that batch, so re-attempting the other ~100k
        permanently-unmatchable FT records every batch is pure waste — the
        work log read "Unlinked-FT rematch: 0 of 101,605 linked" after every
        single batch. Scoping is by NORMALIZED model family, not by exact
        name, so the model-variant stage ("8275" FT ↔ "8275A" trim) still
        works. None = every unlinked record (full pass).

        Bulk strategy: instead of the per-record query cascade (minutes for
        30k+ records), load every candidate trim ONCE, build serial-form
        indexes in memory, and answer each FT record with bisect lookups.
        Match semantics mirror _find_matching_trim exactly: newest trim at or
        before the FT date within FINAL_TEST_MAX_DAYS_FROM_TRIM, staged
        exact serial → fuzzy → aggressive → model-variant.

        Existing links are never touched — rematch_final_tests() re-evaluates
        everything if that is ever needed.
        """
        import bisect
        import time as _time
        from collections import defaultdict
        from sqlalchemy import update as sa_update
        from laser_trim_analyzer.database.models import (
            FinalTestResult as DBFinalTestResult,
        )
        from laser_trim_analyzer.utils.constants import FINAL_TEST_MAX_DAYS_FROM_TRIM

        t0 = _time.time()
        stats = {"unlinked": 0, "new_matches": 0, "still_unmatched": 0,
                 "seconds": 0.0, "models": []}
        window = timedelta(days=FINAL_TEST_MAX_DAYS_FROM_TRIM)
        families = None
        if models is not None:
            families = {self._normalize_model(m) for m in models if m}
            if not families:
                return stats

        with self._write_lock:
            with self.session() as session:
                # Narrow column query, not full ORM entities: the loop reads
                # five fields, and materializing 100k+ FinalTestResult objects
                # (with their identity map) was minutes of the post-batch cost.
                pending = (
                    session.query(
                        DBFinalTestResult.id, DBFinalTestResult.model,
                        DBFinalTestResult.serial, DBFinalTestResult.file_date,
                        DBFinalTestResult.test_date,
                    )
                    .filter(DBFinalTestResult.linked_trim_id.is_(None))
                    .all()
                )
                if families is not None:
                    # Normalization is Python-side, so the family filter can't
                    # be pushed into SQL — but the rows are narrow tuples.
                    pending = [ft for ft in pending if ft.model
                               and self._normalize_model(ft.model) in families]
                stats["unlinked"] = len(pending)
                if not pending:
                    return stats

                # One pass over all trims → per-serial-form indexes of
                # (file_date, trim_id) sorted ascending, plus a per-family
                # index for variant matches ("8275" FT ↔ "8275A" trim).
                exact_ix: dict = defaultdict(list)    # (model, serial_lower) → [(date, id)]
                fuzzy_ix: dict = defaultdict(list)    # (model, norm_serial)  → [(date, id)]
                aggr_ix: dict = defaultdict(list)     # (model, aggr_serial)  → [(date, id)]
                variant_ix: dict = defaultdict(list)  # (norm_model, norm_serial) → [(date, id, model)]
                rows = session.query(
                    DBAnalysisResult.id, DBAnalysisResult.model,
                    DBAnalysisResult.serial, DBAnalysisResult.file_date,
                ).filter(
                    DBAnalysisResult.file_date.isnot(None),
                    DBAnalysisResult.serial.isnot(None),
                ).order_by(DBAnalysisResult.file_date).all()
                for tid, tmodel, tserial, tdate in rows:
                    if families is not None and self._normalize_model(tmodel) not in families:
                        continue    # can't match any in-scope FT record
                    s_low = tserial.lower().strip()
                    s_norm = self._normalize_serial(tserial)
                    exact_ix[(tmodel, s_low)].append((tdate, tid))
                    fuzzy_ix[(tmodel, s_norm)].append((tdate, tid))
                    aggr_ix[(tmodel, self._normalize_serial_aggressive(tserial))].append((tdate, tid))
                    variant_ix[(self._normalize_model(tmodel), s_norm)].append((tdate, tid, tmodel))

                def newest_in_window(entries, test_date, cutoff, model_ok=None):
                    """Rightmost entry with cutoff <= date <= test_date whose
                    model passes model_ok (None = any)."""
                    i = bisect.bisect_right(entries, (test_date, float("inf"))) - 1
                    while i >= 0:
                        e = entries[i]
                        if e[0] < cutoff:
                            return None
                        if model_ok is None or model_ok(e[2]):
                            return e
                        i -= 1
                    return None

                affected_models = set()
                link_updates: List[Dict[str, Any]] = []
                for ft in pending:
                    test_date = ft.file_date or ft.test_date
                    if not ft.model or not ft.serial or not test_date:
                        stats["still_unmatched"] += 1
                        continue
                    cutoff = test_date - window
                    s_low = ft.serial.lower().strip()
                    s_norm = self._normalize_serial(ft.serial)
                    hit = method = None
                    exact_serial = True
                    variant = False
                    penalty = 1.0
                    e = newest_in_window(exact_ix.get((ft.model, s_low), ()), test_date, cutoff)
                    if e:
                        hit, method = e, "exact"
                    if hit is None:
                        e = newest_in_window(fuzzy_ix.get((ft.model, s_norm), ()), test_date, cutoff)
                        if e:
                            hit, method, exact_serial = e, "fuzzy_serial", False
                    if hit is None:
                        # Parity with _find_matching_trim attempt 2b: the
                        # aggressive stage only runs when the FT serial itself
                        # changes under aggressive normalization (code-review
                        # finding #2, 2026-07-13 — the bulk path previously
                        # probed unconditionally and could link serial "123"
                        # to trim "123X", which save-time matching never does).
                        s_aggr = self._normalize_serial_aggressive(ft.serial)
                        if s_aggr != s_norm:
                            e = newest_in_window(
                                aggr_ix.get((ft.model, s_aggr), ()), test_date, cutoff)
                            if e:
                                hit, method, exact_serial, penalty = (
                                    e, "fuzzy_serial_aggressive", False, 0.90)
                    if hit is None:
                        # Variant stage — one side MUST be the base form
                        # (code-review finding #1, 2026-07-13 BLOCKER: keying
                        # only on the normalized family let FT "7953-1A" link
                        # to a "7953-1B" trim — SIBLING variants, a match class
                        # _find_matching_trim never makes. Attempt 3a matches
                        # trims whose model IS the base; attempt 3b matches
                        # variant trims only when the FT model is the base).
                        ft_norm = self._normalize_model(ft.model)
                        ft_is_base = (ft.model == ft_norm)
                        e = newest_in_window(
                            variant_ix.get((ft_norm, s_norm), ()),
                            test_date, cutoff,
                            model_ok=lambda m, _fn=ft_norm, _fm=ft.model, _fb=ft_is_base:
                                m != _fm and (_fb or m == _fn))
                        if e:
                            hit, method, exact_serial, variant = e, "model_variant", False, True
                    if hit is None:
                        stats["still_unmatched"] += 1
                        continue
                    days_diff = (test_date - hit[0]).days
                    link_updates.append({
                        "id": ft.id,
                        "linked_trim_id": hit[1],
                        "match_confidence": self._calculate_match_confidence(
                            days_diff, exact_serial=exact_serial,
                            model_variant=variant) * penalty,
                        "days_since_trim": days_diff,
                        "match_method": method,
                    })
                    affected_models.add(ft.model)
                    stats["new_matches"] += 1
                stats["models"] = sorted(affected_models)

                if link_updates:
                    # Bulk UPDATE ... WHERE id = :id (one statement, no ORM
                    # objects) — same four columns the loop used to assign.
                    session.execute(sa_update(DBFinalTestResult), link_updates)
                session.commit()

        stats["seconds"] = round(_time.time() - t0, 1)
        logger.info(
            "Unlinked-FT rematch (%s): %d of %d linked in %.1fs (%d still unmatched)",
            "all models" if families is None
            else f"{len(families)} model(s) from this batch",
            stats["new_matches"], stats["unlinked"], stats["seconds"],
            stats["still_unmatched"],
        )
        return stats
