"""Start-up schema migrations and the unit_id backfill.

Moved out of `database/manager.py` (2026-09-25, C2 Task 4 -- see
docs/CODE_REVIEW_2026-09-20.md §7 and .superpowers/sdd/C2-refactors/task-4-brief.md).
This is a PURE MOVE: `MigrationsMixin`'s method bodies are byte-for-byte what
`DatabaseManager._run_migrations` / `_backfill_unit_ids` / `_meta_get` /
`_meta_set` used to be -- same behaviour, same order, same logging, same
`except` paths. Nothing about what a migration does changed, only which file
it lives in.

`DatabaseManager` inherits this mixin (`class DatabaseManager(MigrationsMixin):`
in manager.py), so every existing call site is unchanged: `self._run_migrations()`
in `DatabaseManager.__init__`, and tests calling `db._backfill_unit_ids(session)`
directly, both keep working exactly as before.

Why this module imports FROM manager.py, and not the other way only: two
names these migrations need (`compute_unit_id`, and the two `app_meta` key
constants) are used elsewhere in manager.py too, outside the migrations, so
they stayed put there rather than moving here with everything else. Importing
them back here would normally be a circular import (manager.py must also
import `MigrationsMixin` from this module, to use it as a base class) --
it works because manager.py's import of `MigrationsMixin` is placed AFTER
`compute_unit_id` and the two key constants are already defined in manager.py
(see the comment at that import site). Python resolves the two modules'
imports of each other in that order every time, because
`database/__init__.py` always imports `manager` first, before anything can
reach this module directly.
"""

import re
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import func, text
from sqlalchemy.exc import OperationalError

from laser_trim_analyzer.database.manager import (
    INCREMENT_VOLTS_SINCE_KEY,
    UNKNOWN_REPARSE_COUNT_KEY,
    compute_unit_id,
    logger,
)


class MigrationsMixin:
    """Start-up schema migrations (`_run_migrations`) and the unit_id
    backfill (`_backfill_unit_ids`), plus the two `app_meta` helpers used
    only by `_run_migrations`.

    Inherited by `DatabaseManager`. Every method below runs with `self` bound
    to a `DatabaseManager` instance, and reaches back into manager.py through
    `self`: `self.session()`, `self._engine`, `self._reparse_filename(...)`,
    `self.rematch_final_tests()` are all defined there, not here.
    """

    @staticmethod
    def _meta_get(session, key: str) -> Optional[str]:
        """Read an app_meta value, or None if unset.

        Never raises. It is called from inside migrations, and a database
        opened before app_meta existed (or one where the CREATE failed) must
        degrade to "no memory of a previous run" rather than take the whole
        migration pass down with it.
        """
        try:
            row = session.execute(
                text("SELECT value FROM app_meta WHERE key = :k"), {"k": key}).fetchone()
            return row[0] if row else None
        except Exception:
            session.rollback()
            return None

    @staticmethod
    def _meta_set(session, key: str, value: str) -> None:
        """Write an app_meta value, committing it. Never raises (see _meta_get).

        A failure here costs only the skip — the next launch redoes the work
        and tries to record it again — so it must not abort the migration that
        just succeeded.
        """
        try:
            session.execute(text(
                "INSERT INTO app_meta (key, value, updated_at) VALUES (:k, :v, :t) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
                "updated_at = excluded.updated_at"),
                {"k": key, "v": value,
                 "t": datetime.now().isoformat(sep=" ", timespec="seconds")})
            session.commit()
        except Exception:
            session.rollback()
            logger.warning(f"could not record app_meta[{key}]", exc_info=True)

    def _run_migrations(self) -> None:
        """Run database migrations for schema updates."""
        # Baseline requalification audit table (2026-07-13: per-model manual
        # baseline reset on design change — AS9100 traceability).
        try:
            with self.session() as _s:
                _s.execute(text(
                    "CREATE TABLE IF NOT EXISTS baseline_requalifications ("
                    "id INTEGER PRIMARY KEY AUTOINCREMENT, model TEXT NOT NULL, "
                    "effective_date TEXT NOT NULL, note TEXT, set_at TEXT NOT NULL)"))
                _s.execute(text(
                    "CREATE INDEX IF NOT EXISTS idx_baseline_requal_model "
                    "ON baseline_requalifications(model)"))
                _s.commit()
        except Exception:
            logger.exception("baseline_requalifications migration failed")

        # App-level key/value state (2026-08-31). Every other table in this
        # schema is domain data and there was no meta/settings table, so
        # migrations had nowhere to record what they had already tried — see
        # the Unknown-model re-parse below, its first and so far only user.
        # Deliberately dumb: TEXT values, one row per key, raw SQL and no ORM
        # model, matching baseline_requalifications directly above.
        try:
            with self.session() as _s:
                _s.execute(text(
                    "CREATE TABLE IF NOT EXISTS app_meta ("
                    "key TEXT PRIMARY KEY, value TEXT NOT NULL, "
                    "updated_at TEXT NOT NULL)"))
                _s.commit()
        except Exception:
            logger.exception("app_meta migration failed")

        needs_rematch = False

        with self.session() as session:
            # Migration: Add is_anomaly and anomaly_reason columns to track_results
            try:
                # Check if columns exist by attempting a query
                session.execute(text("SELECT is_anomaly FROM track_results LIMIT 1"))
            except OperationalError:
                session.rollback()  # Clear error state from failed probe
                # Columns don't exist, add them
                logger.info("Running migration: Adding is_anomaly and anomaly_reason columns")
                try:
                    session.execute(text("ALTER TABLE track_results ADD COLUMN is_anomaly BOOLEAN DEFAULT 0"))
                    session.execute(text("ALTER TABLE track_results ADD COLUMN anomaly_reason TEXT"))
                    session.commit()
                    logger.info("Migration completed: Added anomaly detection columns")
                except Exception as e:
                    logger.warning(f"Migration warning (may already exist): {e}")

            # Migration: Add drift_baseline_cutoff_date column to model_ml_state
            try:
                session.execute(text("SELECT drift_baseline_cutoff_date FROM model_ml_state LIMIT 1"))
            except OperationalError:
                session.rollback()  # Clear error state from failed probe
                logger.info("Running migration: Adding drift_baseline_cutoff_date column")
                try:
                    session.execute(text("ALTER TABLE model_ml_state ADD COLUMN drift_baseline_cutoff_date DATETIME"))
                    session.commit()
                    logger.info("Migration completed: Added drift_baseline_cutoff_date column")
                except Exception as e:
                    logger.warning(f"Migration warning (may already exist): {e}")

            # Migration: Add peak_cusum column to model_ml_state
            try:
                session.execute(text("SELECT peak_cusum FROM model_ml_state LIMIT 1"))
            except OperationalError:
                session.rollback()  # Clear error state from failed probe
                logger.info("Running migration: Adding peak_cusum column")
                try:
                    session.execute(text("ALTER TABLE model_ml_state ADD COLUMN peak_cusum FLOAT DEFAULT 0"))
                    session.commit()
                    logger.info("Migration completed: Added peak_cusum column")
                except Exception as e:
                    logger.warning(f"Migration warning (may already exist): {e}")

            # Migration: Normalize status values from 'Pass' to 'PASS' format
            # SQLAlchemy stores enum NAME (PASS), not value (Pass)
            # This fixes data corrupted by bulk SQL that used .value instead of .name
            try:
                # Check if there are any title-case values that need fixing
                result = session.execute(text(
                    "SELECT COUNT(*) FROM analysis_results WHERE overall_status IN ('Pass', 'Fail', 'Warning', 'Error')"
                )).scalar()
                if result and result > 0:
                    logger.info(f"Running migration: Normalizing {result} status values to uppercase")
                    # Fix analysis_results
                    session.execute(text("UPDATE analysis_results SET overall_status = 'PASS' WHERE overall_status = 'Pass'"))
                    session.execute(text("UPDATE analysis_results SET overall_status = 'FAIL' WHERE overall_status = 'Fail'"))
                    session.execute(text("UPDATE analysis_results SET overall_status = 'WARNING' WHERE overall_status = 'Warning'"))
                    session.execute(text("UPDATE analysis_results SET overall_status = 'ERROR' WHERE overall_status = 'Error'"))
                    # Fix track_results
                    session.execute(text("UPDATE track_results SET status = 'PASS' WHERE status = 'Pass'"))
                    session.execute(text("UPDATE track_results SET status = 'FAIL' WHERE status = 'Fail'"))
                    session.execute(text("UPDATE track_results SET status = 'WARNING' WHERE status = 'Warning'"))
                    session.execute(text("UPDATE track_results SET status = 'ERROR' WHERE status = 'Error'"))
                    # Fix final_test_results / smoothness_results too (the original
                    # migration missed these tables). Idempotent if already uppercase.
                    for _tbl in ("final_test_results", "smoothness_results"):
                        for _old, _new in (("Pass", "PASS"), ("Fail", "FAIL"),
                                           ("Warning", "WARNING"), ("Error", "ERROR"),
                                           ("Untrimmed", "UNTRIMMED")):
                            try:
                                session.execute(text(
                                    f"UPDATE {_tbl} SET overall_status = '{_new}' "
                                    f"WHERE overall_status = '{_old}'"))
                            except Exception:
                                pass  # table/column may not exist on older schemas
                    session.commit()
                    logger.info("Migration completed: Status values normalized")
            except Exception as e:
                logger.warning(f"Status normalization warning: {e}")

            # Migration: Clean up "-shop" model name parsing artifacts
            # Files like "8444-shop0_date.xlsx" were incorrectly parsed with
            # model="8444-shop0" instead of model="8444", serial="shop0"
            try:
                shop_count = session.execute(text(
                    "SELECT COUNT(*) FROM analysis_results WHERE LOWER(model) LIKE '%-shop%'"
                )).scalar()
                if shop_count and shop_count > 0:
                    logger.info(f"Running migration: Cleaning up {shop_count} shop model name records")

                    # Fix analysis_results: split "8444-shop0" into model="8444", serial="shop0"
                    shop_records = session.execute(text(
                        "SELECT id, model FROM analysis_results WHERE LOWER(model) LIKE '%-shop%'"
                    )).fetchall()

                    for row in shop_records:
                        old_model = row[1]
                        # Find the "-shop" split point (case-insensitive)
                        lower = old_model.lower()
                        shop_idx = lower.find('-shop')
                        if shop_idx > 0:
                            base_model = old_model[:shop_idx]
                            new_serial = old_model[shop_idx + 1:]  # "shop0", "shop101", etc.
                            session.execute(text(
                                "UPDATE analysis_results SET model = :model, serial = :serial WHERE id = :id"
                            ), {"model": base_model, "serial": new_serial, "id": row[0]})

                    # Delete model_ml_state entries for fake shop model names
                    ml_deleted = session.execute(text(
                        "DELETE FROM model_ml_state WHERE LOWER(model) LIKE '%-shop%'"
                    )).rowcount
                    logger.info(f"Deleted {ml_deleted} fake ML state entries for shop models")

                    session.commit()
                    needs_rematch = True
                    logger.info(f"Migration completed: Cleaned up {shop_count} shop model name records")
            except Exception as e:
                logger.warning(f"Shop model cleanup warning: {e}")

            # Migration: Re-parse "Unknown" model records with improved parser logic
            # Handles: multi-hyphen models (7280-1-CT), -sn serial indicators,
            # concatenated sn patterns, "final NNN" serials, etc.
            #
            # Runs only when the Unknown population has CHANGED since the last
            # attempt (2026-08-31). On the work database it re-parsed the same
            # 381 filenames, fixed 0 of them and printed two INFO lines on
            # every single launch for months. A migration that has provably
            # finished should cost nothing and say nothing; otherwise the
            # startup log stops being something anyone reads, and the next real
            # message hides in the noise. The parser is what decides these
            # filenames are unreadable, and the parser does not change between
            # two launches — only the rows do. So the count of Unknown rows is
            # the whole trigger: a different count means rows this has never
            # tried (new Unknown rows arrived, or some were fixed or deleted),
            # and it runs again and reports at INFO. Same count, same verdict,
            # no work. The fix logic below is untouched.
            try:
                unknown_count = session.execute(text(
                    "SELECT COUNT(*) FROM analysis_results WHERE model = 'Unknown'"
                )).scalar() or 0
                attempted = self._meta_get(session, UNKNOWN_REPARSE_COUNT_KEY)
                if not unknown_count:
                    pass  # nothing to re-parse, and nothing worth saying
                elif attempted == str(unknown_count):
                    logger.debug(
                        f"Unknown model re-parse: skipped, {unknown_count} records "
                        f"unchanged since the last attempt")
                else:
                    unknown_records = session.execute(text(
                        "SELECT id, filename FROM analysis_results WHERE model = 'Unknown'"
                    )).fetchall()
                    logger.info(f"Running migration: Re-parsing {len(unknown_records)} Unknown model records")
                    fixed = 0
                    for row in unknown_records:
                        rec_id, filename = row[0], row[1]
                        model, serial = self._reparse_filename(filename)
                        if model != "Unknown":
                            session.execute(text(
                                "UPDATE analysis_results SET model = :model, serial = :serial WHERE id = :id"
                            ), {"model": model, "serial": serial, "id": rec_id})
                            fixed += 1
                    if fixed > 0:
                        session.commit()
                        needs_rematch = True
                        logger.info(f"Migration completed: Fixed {fixed} of {len(unknown_records)} Unknown model records")
                    else:
                        logger.info("Migration: No Unknown records could be re-parsed")
                    # The REMAINDER, not what we started with: storing the
                    # pre-fix count on a run that fixed something would leave a
                    # marker no future launch can ever match, and this would
                    # re-run forever on exactly the databases it had improved.
                    self._meta_set(session, UNKNOWN_REPARSE_COUNT_KEY,
                                   str(unknown_count - fixed))
            except Exception as e:
                logger.warning(f"Unknown model re-parse warning: {e}")

            # Migration: Ensure all performance indexes exist
            # create_all() only creates indexes for NEW tables. Existing databases
            # may be missing indexes that were added to models.py later.
            # CREATE INDEX IF NOT EXISTS is idempotent — safe to run every startup.
            try:
                index_statements = [
                    # analysis_results indexes
                    "CREATE INDEX IF NOT EXISTS idx_filename_date ON analysis_results(filename, file_date)",
                    "CREATE INDEX IF NOT EXISTS idx_file_date ON analysis_results(file_date)",
                    "CREATE INDEX IF NOT EXISTS idx_model_serial ON analysis_results(model, serial)",
                    "CREATE INDEX IF NOT EXISTS idx_model_serial_date ON analysis_results(model, serial, file_date)",
                    "CREATE INDEX IF NOT EXISTS idx_timestamp ON analysis_results(timestamp)",
                    "CREATE INDEX IF NOT EXISTS idx_status ON analysis_results(overall_status)",
                    "CREATE INDEX IF NOT EXISTS idx_system ON analysis_results(system)",
                    "CREATE INDEX IF NOT EXISTS idx_status_timestamp ON analysis_results(overall_status, timestamp)",
                    "CREATE INDEX IF NOT EXISTS idx_model_status ON analysis_results(model, overall_status)",
                    # track_results indexes
                    "CREATE INDEX IF NOT EXISTS idx_track_analysis ON track_results(analysis_id, track_id)",
                    "CREATE INDEX IF NOT EXISTS idx_track_sigma_gradient ON track_results(sigma_gradient)",
                    "CREATE INDEX IF NOT EXISTS idx_track_sigma_pass ON track_results(sigma_pass)",
                    "CREATE INDEX IF NOT EXISTS idx_track_linearity_pass ON track_results(linearity_pass)",
                    "CREATE INDEX IF NOT EXISTS idx_track_risk_category ON track_results(risk_category)",
                    "CREATE INDEX IF NOT EXISTS idx_track_failure_probability ON track_results(failure_probability)",
                    "CREATE INDEX IF NOT EXISTS idx_track_status ON track_results(status)",
                    "CREATE INDEX IF NOT EXISTS idx_track_analysis_prob ON track_results(analysis_id, failure_probability)",
                    # final_test_results indexes
                    "CREATE INDEX IF NOT EXISTS idx_ft_filename_date ON final_test_results(filename, file_date)",
                    "CREATE INDEX IF NOT EXISTS idx_ft_model_serial ON final_test_results(model, serial)",
                    "CREATE INDEX IF NOT EXISTS idx_ft_model_serial_date ON final_test_results(model, serial, file_date)",
                    "CREATE INDEX IF NOT EXISTS idx_ft_timestamp ON final_test_results(timestamp)",
                    "CREATE INDEX IF NOT EXISTS idx_ft_status ON final_test_results(overall_status)",
                    "CREATE INDEX IF NOT EXISTS idx_ft_linked_trim ON final_test_results(linked_trim_id)",
                    "CREATE INDEX IF NOT EXISTS idx_ft_test_date ON final_test_results(test_date)",
                    # Standalone file_date index - Compare/Final Test page does
                    # ORDER BY file_date DESC LIMIT 500 with no leading filter,
                    # which the composite indexes don't satisfy.
                    "CREATE INDEX IF NOT EXISTS idx_ft_file_date ON final_test_results(file_date)",
                    # Placed last so the bulk loop on pre-Spec-1 DBs re-confirms all
                    # existing indexes before failing on this one new-column entry.
                    # The Spec 1 column migration immediately after (untrimmed_sigma_gradient
                    # block) creates both the column and this index on first upgrade.
                    "CREATE INDEX IF NOT EXISTS idx_track_untrimmed_sigma_gradient ON track_results(untrimmed_sigma_gradient)",
                    # file_hash lookups (ingest-speed spec 3.7, ruling 11): every
                    # final-test and smoothness save checks "is this content already
                    # on record?" by file_hash before deciding insert vs. duplicate/
                    # upsert (save_final_test, save_smoothness_result), and so do
                    # is_file_processed and the stat-heal pass. Unindexed, F9 measured
                    # that SCANning the whole final_test_results table (151,793 rows)
                    # was 20 of the FT save's 22 ms; indexed, 3.4-3.7 batched.
                    "CREATE INDEX IF NOT EXISTS idx_ft_file_hash ON final_test_results(file_hash)",
                    "CREATE INDEX IF NOT EXISTS idx_smoothness_file_hash ON smoothness_results(file_hash)",
                ]
                created = 0
                for stmt in index_statements:
                    session.execute(text(stmt))
                    created += 1
                session.commit()
                logger.info(f"Index migration: ensured {created} indexes exist")
            except Exception as e:
                session.rollback()  # Clear error state from the failed statement (e.g. a
                                     # read-only database, James's first launch on a pre-
                                     # Task-3 file before these two are no-ops) -- matches
                                     # every sibling migration's idiom in this method.
                logger.warning(f"Index migration warning: {e}")

            # Migration: Add failure margin columns to track_results
            try:
                session.execute(text("SELECT max_violation FROM track_results LIMIT 1"))
            except OperationalError:
                session.rollback()  # Clear error state from failed probe
                logger.info("Running migration: Adding failure margin columns")
                try:
                    session.execute(text("ALTER TABLE track_results ADD COLUMN max_violation FLOAT"))
                    session.execute(text("ALTER TABLE track_results ADD COLUMN avg_violation FLOAT"))
                    session.execute(text("ALTER TABLE track_results ADD COLUMN margin_to_spec FLOAT"))
                    session.commit()
                    logger.info("Migration completed: Added failure margin columns")
                except Exception as e:
                    logger.warning(f"Failure margin migration warning (may already exist): {e}")

            # Migration: Add linearity_spec_warning column to track_results
            try:
                session.execute(text("SELECT linearity_spec_warning FROM track_results LIMIT 1"))
            except OperationalError:
                session.rollback()  # Clear error state from failed probe
                logger.info("Running migration: Adding linearity_spec_warning column")
                try:
                    session.execute(text("ALTER TABLE track_results ADD COLUMN linearity_spec_warning TEXT"))
                    session.commit()
                    logger.info("Migration completed: Added linearity_spec_warning column")
                except Exception as e:
                    logger.warning(f"linearity_spec_warning migration warning (may already exist): {e}")

            # Migration: Add error_reason column to analysis_results (2026-09-23).
            # Why overall_status is ERROR, for the 237 rows on the rebuild that
            # couldn't say -- see core/processor.py's error_reason_of(). Not
            # back-filled: the Model page COALESCEs onto track_results'
            # linearity_spec_warning/anomaly_reason for rows written before
            # this column existed.
            try:
                session.execute(text("SELECT error_reason FROM analysis_results LIMIT 1"))
            except OperationalError:
                session.rollback()  # Clear error state from failed probe
                logger.info("Running migration: Adding error_reason column")
                try:
                    session.execute(text("ALTER TABLE analysis_results ADD COLUMN error_reason TEXT"))
                    session.commit()
                    logger.info("Migration completed: Added error_reason column")
                except Exception as e:
                    logger.warning(f"error_reason migration warning (may already exist): {e}")

            # Migration: laser 1's TrimVolts capture on trim_passes (2026-09-24;
            # see TrimPass.increment_volts). Metadata-only ADD COLUMNs, so a
            # 6 GB database is not rewritten. Rows already stored read NULL --
            # "not captured", which is exactly what they are -- until the
            # back-fill (design doc ruling 4c) or a reprocess fills them. No
            # DEFAULT, deliberately: a default would erase the record of which
            # rows predate the capture.
            increment_volts_columns = {
                "increment_volts": "JSON",
                "increment_volts_first_row": "INTEGER",
                "increment_volts_truncated": "BOOLEAN",
            }
            for col_name, col_type in increment_volts_columns.items():
                try:
                    session.execute(text(
                        f"ALTER TABLE trim_passes ADD COLUMN {col_name} {col_type}"))
                    session.commit()
                    logger.info(f"Migration: Added {col_name} column to trim_passes")
                except Exception as e:
                    if ("duplicate column" not in str(e).lower()
                            and "already exists" not in str(e).lower()):
                        logger.warning(f"trim_passes.{col_name} migration warning: {e}")
                    session.rollback()
            # ...and WHEN this database started capturing (INCREMENT_VOLTS_SINCE_KEY):
            # recorded once, by the first start-up that finds the column in place
            # with nothing recorded -- the one whose ALTER above just added it, or a
            # new database's first start-up (create_all made the column). A database
            # migrated by the first version of this code, which recorded nothing, gets
            # its record at its next start-up; the passes written in between carry
            # their curves anyway. Never recorded while the column is missing.
            try:
                has_column = any(
                    r[1] == "increment_volts" for r in session.execute(
                        text("PRAGMA table_info(trim_passes)")).fetchall())
            except Exception:
                session.rollback()
                has_column = False
            if has_column and self._meta_get(session, INCREMENT_VOLTS_SINCE_KEY) is None:
                self._meta_set(session, INCREMENT_VOLTS_SINCE_KEY,
                               datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f"))

            # Migration: initial_trim_value gets its own column on trim_passes
            # (2026-09-24; see TrimPass.initial_trim_value). Metadata-only ADD
            # COLUMN -- no data is moved. The ~83,000 existing laser-2/3 pass
            # rows keep the value inside `recipe`, where it has always been; a
            # heavy UPDATE across them at start-up is the shape of the
            # 2026-09-14 night. They read back the same value as a new row
            # through trim_passes.initial_trim_values(row.initial_trim_value,
            # row.recipe) -- no "since" record is needed the way increment_volts
            # has one, because that helper's recipe fallback works forever, not
            # just until a back-fill catches up. No DEFAULT: a default would
            # make an old row indistinguishable from a laser-1 row that never
            # had the value at all.
            try:
                session.execute(text("SELECT initial_trim_value FROM trim_passes LIMIT 1"))
            except OperationalError:
                session.rollback()  # Clear error state from failed probe
                logger.info("Running migration: Adding initial_trim_value column")
                try:
                    session.execute(text(
                        "ALTER TABLE trim_passes ADD COLUMN initial_trim_value JSON"))
                    session.commit()
                    logger.info("Migration completed: Added initial_trim_value column")
                except Exception as e:
                    if ("duplicate column" not in str(e).lower()
                            and "already exists" not in str(e).lower()):
                        logger.warning(f"initial_trim_value migration warning: {e}")
                    session.rollback()

            # Migration: track2_parameters gets its own column on trim_setup
            # (2026-09-24; see TrimSetup.track2_parameters). Metadata-only ADD
            # COLUMN -- no data is moved, and there is nothing to move: this
            # column never existed under another name. `trim_setup` stays one
            # row per analysis (analysis_id is UNIQUE) -- dropping that would
            # need a table rebuild, the same shape as the sigma-nullable
            # migration above, for a column most files never populate. No
            # back-fill either: two-track files that already exist in this
            # database are re-read on reprocess, not updated here -- the
            # column is simply NULL on every row until then.
            try:
                session.execute(text("SELECT track2_parameters FROM trim_setup LIMIT 1"))
            except OperationalError:
                session.rollback()  # Clear error state from failed probe
                logger.info("Running migration: Adding track2_parameters column")
                try:
                    session.execute(text(
                        "ALTER TABLE trim_setup ADD COLUMN track2_parameters JSON"))
                    session.commit()
                    logger.info("Migration completed: Added track2_parameters column")
                except Exception as e:
                    if ("duplicate column" not in str(e).lower()
                            and "already exists" not in str(e).lower()):
                        logger.warning(f"track2_parameters migration warning: {e}")
                    session.rollback()

            # Migration: Add measured_electrical_angle column to track_results
            try:
                session.execute(text("SELECT measured_electrical_angle FROM track_results LIMIT 1"))
            except OperationalError:
                session.rollback()  # Clear error state from failed probe
                logger.info("Running migration: Adding measured_electrical_angle column")
                try:
                    session.execute(text("ALTER TABLE track_results ADD COLUMN measured_electrical_angle FLOAT"))
                    session.commit()
                    logger.info("Migration completed: Added measured_electrical_angle column")
                except Exception as e:
                    logger.warning(f"measured_electrical_angle migration warning (may already exist): {e}")

            # Migration: Add max deviation columns to track_results
            try:
                session.execute(text("SELECT max_deviation FROM track_results LIMIT 1"))
            except OperationalError:
                session.rollback()  # Clear error state from failed probe
                logger.info("Running migration: Adding max deviation columns")
                try:
                    session.execute(text("ALTER TABLE track_results ADD COLUMN max_deviation FLOAT"))
                    session.execute(text("ALTER TABLE track_results ADD COLUMN max_deviation_position FLOAT"))
                    session.execute(text("ALTER TABLE track_results ADD COLUMN deviation_uniformity FLOAT"))
                    session.commit()
                    logger.info("Migration completed: Added max deviation columns")
                except Exception as e:
                    logger.warning(f"Max deviation migration warning (may already exist): {e}")

            # Migration: Add untrimmed_sigma_gradient column to track_results.
            # Spec 1 (2026-05-30): upstream element-quality signal independent
            # of post-trim sigma_gradient.  Backfilled by natural reprocess flow.
            try:
                session.execute(
                    text("SELECT untrimmed_sigma_gradient FROM track_results LIMIT 1")
                )
            except OperationalError:
                session.rollback()  # Clear error state from failed probe
                logger.info(
                    "Running migration: Adding untrimmed_sigma_gradient column"
                )
                try:
                    session.execute(text(
                        "ALTER TABLE track_results "
                        "ADD COLUMN untrimmed_sigma_gradient FLOAT"
                    ))
                    session.execute(text(
                        "CREATE INDEX IF NOT EXISTS "
                        "idx_track_untrimmed_sigma_gradient "
                        "ON track_results (untrimmed_sigma_gradient)"
                    ))
                    session.commit()
                    logger.info(
                        "Migration completed: Added untrimmed_sigma_gradient"
                    )
                except Exception as e:
                    logger.warning(
                        f"Migration warning (may already exist): {e}"
                    )

            # Migration: Add untrimmed_error_max column to track_results.
            # Spec 2 (2026-06-02): worst-case linearity error across untrimmed
            # data points; complements untrimmed_sigma_gradient as an element-
            # quality signal.  Backfilled by natural reprocess flow.
            try:
                session.execute(
                    text("SELECT untrimmed_error_max FROM track_results LIMIT 1")
                )
            except OperationalError:
                session.rollback()  # Clear error state from failed probe
                logger.info(
                    "Running migration: Adding untrimmed_error_max column"
                )
                try:
                    session.execute(text(
                        "ALTER TABLE track_results "
                        "ADD COLUMN untrimmed_error_max FLOAT"
                    ))
                    session.execute(text(
                        "CREATE INDEX IF NOT EXISTS "
                        "idx_track_untrimmed_error_max "
                        "ON track_results (untrimmed_error_max)"
                    ))
                    session.commit()
                    logger.info(
                        "Migration completed: Added untrimmed_error_max"
                    )
                except Exception as e:
                    logger.warning(
                        f"Migration warning (may already exist): {e}"
                    )

            # Migration: Add composite_trim_risk_score column to track_results.
            try:
                session.execute(
                    text("SELECT composite_trim_risk_score FROM track_results LIMIT 1")
                )
            except OperationalError:
                session.rollback()
                logger.info("Running migration: Adding composite_trim_risk_score column")
                try:
                    session.execute(text(
                        "ALTER TABLE track_results "
                        "ADD COLUMN composite_trim_risk_score FLOAT"
                    ))
                    session.execute(text(
                        "CREATE INDEX IF NOT EXISTS idx_track_composite_trim_risk_score "
                        "ON track_results (composite_trim_risk_score)"
                    ))
                    session.commit()
                    logger.info("Migration completed: Added composite_trim_risk_score")
                except Exception as e:
                    logger.warning(f"Migration warning (may already exist): {e}")

            # Migration: Create model_metric_state table for Spec 2.
            # This is a CREATE TABLE rather than ALTER TABLE because the
            # table is entirely new in V6.  Use Base.metadata.create_all
            # with checkfirst=True for idempotency.
            try:
                from laser_trim_analyzer.database.models import ModelMetricState
                ModelMetricState.__table__.create(bind=self._engine, checkfirst=True)
                session.commit()
            except Exception as e:
                session.rollback()
                logger.warning(
                    f"Migration warning for model_metric_state (may already exist): {e}"
                )

            # Migration: Retag LTS3 (System C) rows (2026-07-06). The third
            # trim system writes files format-identical to an existing system,
            # so anything processed before path-based detection landed was
            # stored as A or B. Identity marker = an 'LTS3*' DIRECTORY in the
            # path (a separator must follow, so filenames starting with LTS3
            # don't match). Idempotent: already-C rows aren't selected.
            try:
                total_retagged = 0
                for table in ("analysis_results", "final_test_results"):
                    res = session.execute(text(
                        f"UPDATE {table} SET system = 'C' "
                        f"WHERE system != 'C' AND ("
                        f"file_path LIKE '%/LTS3%/%' OR file_path LIKE '%\\LTS3%\\%'"
                        f")"
                    ))
                    total_retagged += res.rowcount or 0
                session.commit()
                if total_retagged:
                    logger.info(f"Migration: retagged {total_retagged} LTS3 rows as System C")
            except Exception as e:
                session.rollback()
                logger.warning(f"LTS3 retag migration warning: {e}")

            # Migration: Add last_row_id watermark + recent_window to
            # model_metric_state (2026-07-06). last_row_id lets
            # advance_drift_state consume same-day samples the date-only
            # filter skipped forever; recent_window makes the step-change
            # check live across restarts.
            for col, ddl in (("last_row_id", "INTEGER"), ("recent_window", "TEXT")):
                try:
                    session.execute(text(
                        f"SELECT {col} FROM model_metric_state LIMIT 1"))
                except OperationalError:
                    session.rollback()
                    logger.info(f"Running migration: Adding model_metric_state.{col}")
                    try:
                        session.execute(text(
                            f"ALTER TABLE model_metric_state ADD COLUMN {col} {ddl}"
                        ))
                        session.commit()
                        logger.info(f"Migration completed: Added {col}")
                    except Exception as e:
                        logger.warning(f"{col} migration warning (may already exist): {e}")

            # Migration: Add data_quality columns to analysis_results
            try:
                session.execute(text("SELECT data_quality FROM analysis_results LIMIT 1"))
            except OperationalError:
                session.rollback()  # Clear error state from failed probe
                logger.info("Running migration: Adding data_quality columns")
                try:
                    session.execute(text(
                        "ALTER TABLE analysis_results ADD COLUMN data_quality VARCHAR(20) DEFAULT 'good'"
                    ))
                    session.execute(text(
                        "ALTER TABLE analysis_results ADD COLUMN data_quality_issues TEXT"
                    ))
                    session.commit()
                    logger.info("Migration completed: Added data_quality columns")
                except Exception as e:
                    logger.warning(f"Data quality migration warning (may already exist): {e}")

            # Migration: Add Phase 2 spec-aware optimization columns to track_results.
            # Each column gets its own try/commit so a duplicate-column error on
            # one ALTER does not roll back columns added earlier in the same
            # session — that was the prior bug (single rollback at end of loop
            # discarded successful ALTERs in the SQLAlchemy unit of work).
            phase2_columns = {
                "optimal_slope": "FLOAT DEFAULT 0.0",
                "station_compensation": "FLOAT",
                "linearity_type": "VARCHAR(30)",
                "raw_linearity_error": "FLOAT",
                "optimized_linearity_error": "FLOAT",
                "raw_fail_points": "INTEGER",
            }
            for col_name, col_type in phase2_columns.items():
                try:
                    session.execute(text(
                        f"ALTER TABLE track_results ADD COLUMN {col_name} {col_type}"
                    ))
                    session.commit()
                except Exception as e:
                    session.rollback()
                    if "duplicate column" not in str(e).lower() and "already exists" not in str(e).lower():
                        logger.warning(f"Migration error adding {col_name}: {e}")
            logger.info("Phase 2 migration: ensured spec-aware columns exist")

            # Migration: Add match_method column to final_test_results
            try:
                session.execute(text("SELECT match_method FROM final_test_results LIMIT 1"))
            except OperationalError:
                session.rollback()  # Clear error state from failed probe
                try:
                    session.execute(text("ALTER TABLE final_test_results ADD COLUMN match_method VARCHAR(30)"))
                    session.commit()
                    logger.info("Migration: Added match_method column to final_test_results")
                except Exception:
                    pass

            # Migration: Add trim_pass_count column to track_results.
            # Counts how many laser-trim passes the equipment ran per track,
            # surfaced from the file's "Trim N" / "TRK<n> M" sheet layout.
            # Used as a quality indicator (1 = clean, 2+ = retrim needed).
            try:
                session.execute(text("SELECT trim_pass_count FROM track_results LIMIT 1"))
            except OperationalError:
                session.rollback()
                try:
                    session.execute(text("ALTER TABLE track_results ADD COLUMN trim_pass_count INTEGER"))
                    session.commit()
                    logger.info("Migration: Added trim_pass_count column to track_results")
                except Exception:
                    pass

            # Migration: Add unit_id column to analysis_results.
            # Canonical unit identifier "<model>/<shop>/<date>", used by the
            # unit-level yield feature (Trends chart + Excel "Yield by Unit"
            # sheet). Nullable so the app remains functional during/after
            # partial backfill; the backfill itself happens in a separate
            # migration step below so it can be retried independently.
            try:
                session.execute(text("SELECT unit_id FROM analysis_results LIMIT 1"))
            except OperationalError:
                session.rollback()
                try:
                    session.execute(text(
                        "ALTER TABLE analysis_results ADD COLUMN unit_id VARCHAR(80)"
                    ))
                    session.commit()
                    logger.info("Migration: Added unit_id column to analysis_results")
                except Exception as e:
                    session.rollback()
                    logger.warning(f"Migration error adding unit_id: {e}")
            # Ensure the index exists on both new and migrated DBs. Project
            # convention uses idx_* naming (not SQLAlchemy's auto ix_*),
            # so we manage it explicitly here rather than via index=True
            # on the column declaration.
            try:
                session.execute(text(
                    "CREATE INDEX IF NOT EXISTS idx_analysis_unit_id "
                    "ON analysis_results(unit_id)"
                ))
                session.commit()
            except Exception as e:
                session.rollback()
                logger.warning(f"Migration error creating unit_id index: {e}")

            # Migration: Backfill unit_id for existing rows. Idempotent —
            # only updates rows where unit_id IS NULL, so re-running picks up
            # where it left off (e.g. after a crash or interrupted startup).
            # Done in Python because the shop-number extraction regex lives
            # there; iterates in batches of 1000 to keep memory bounded.
            #
            # Guarded like every sibling migration (2026-09-24): it WRITES, so
            # on a database it cannot write -- a read-only file, a full disk, a
            # locked share -- it raised straight out of _init_database and the
            # V6 app exited with "Fatal error" before showing a single screen.
            # The work database has 1,512 rows no backfill can fill (a junk
            # serial or no date), so this runs, and writes, at EVERY start-up.
            # A refused backfill costs only the unit-level yield of the rows it
            # did not reach, and it is retried at the next start-up.
            try:
                self._backfill_unit_ids(session)
            except Exception as e:
                session.rollback()
                logger.warning(
                    f"unit_id backfill migration refused, skipped until the next "
                    f"start-up (the database opens without it): {e}")

            # Migration: Add aliases column to model_specs.
            # Stores pipe-separated alternate model numbers so a single spec
            # row covers cases like 1621501 and 2001621501 being the same part.
            try:
                session.execute(text("SELECT aliases FROM model_specs LIMIT 1"))
            except OperationalError:
                session.rollback()  # Clear error state from failed probe
                try:
                    session.execute(text("ALTER TABLE model_specs ADD COLUMN aliases TEXT"))
                    session.commit()
                    logger.info("Migration: Added aliases column to model_specs")
                except Exception as e:
                    if "duplicate column" not in str(e).lower():
                        logger.warning(f"aliases migration warning: {e}")
                    session.rollback()

            # Migration: Add exclude_points column to model_specs
            try:
                session.execute(text("SELECT exclude_points FROM model_specs LIMIT 1"))
            except OperationalError:
                session.rollback()  # Clear error state from failed probe
                try:
                    session.execute(text("ALTER TABLE model_specs ADD COLUMN exclude_points TEXT"))
                    session.commit()
                    logger.info("Migration: Added exclude_points column to model_specs")
                except Exception as e:
                    if "duplicate column" not in str(e).lower():
                        logger.warning(f"exclude_points migration warning: {e}")
                    session.rollback()

            # Migration: Add exclude_points_ft column to model_specs
            # FT files have different data point counts than trim files,
            # so they need separate exclude ranges.
            try:
                session.execute(text("SELECT exclude_points_ft FROM model_specs LIMIT 1"))
            except OperationalError:
                session.rollback()  # Clear error state from failed probe
                try:
                    session.execute(text("ALTER TABLE model_specs ADD COLUMN exclude_points_ft TEXT"))
                    session.commit()
                    logger.info("Migration: Added exclude_points_ft column to model_specs")
                except Exception as e:
                    if "duplicate column" not in str(e).lower():
                        logger.warning(f"exclude_points_ft migration warning: {e}")
                    session.rollback()

            # Migration: Add open_closed column to model_specs and backfill from
            # circuit_type. The original importer wrote the Excel "Open/Closed"
            # column into circuit_type, which is misleading — Open vs Closed
            # refers to whether the resistive element is visible, not the
            # electrical circuit type. Keep circuit_type for backward compat
            # but add a correctly-named column and sync values across.
            try:
                session.execute(text("SELECT open_closed FROM model_specs LIMIT 1"))
            except OperationalError:
                session.rollback()  # Clear error state from failed probe
                try:
                    session.execute(text(
                        "ALTER TABLE model_specs ADD COLUMN open_closed VARCHAR(10)"
                    ))
                    # Backfill from circuit_type for existing rows
                    session.execute(text(
                        "UPDATE model_specs SET open_closed = circuit_type "
                        "WHERE open_closed IS NULL AND circuit_type IS NOT NULL"
                    ))
                    session.commit()
                    logger.info(
                        "Migration: Added open_closed column to model_specs "
                        "and backfilled from circuit_type"
                    )
                except Exception as e:
                    if "duplicate column" not in str(e).lower():
                        logger.warning(f"open_closed migration warning: {e}")
                    session.rollback()

            # Migration: Add spec-aware columns to final_test_tracks so the FT
            # analyzer's optimal_slope/offset/linearity_type are persisted (not
            # just held in memory for the current Process Files screen).
            ft_phase2_columns = {
                "optimal_offset": "FLOAT",
                "optimal_slope": "FLOAT DEFAULT 0.0",
                "linearity_type": "VARCHAR(30)",
            }
            for col_name, col_type in ft_phase2_columns.items():
                try:
                    session.execute(text(
                        f"ALTER TABLE final_test_tracks ADD COLUMN {col_name} {col_type}"
                    ))
                except Exception as e:
                    if "duplicate column" not in str(e).lower() and "already exists" not in str(e).lower():
                        logger.warning(f"FT migration warning adding {col_name}: {e}")
                    session.rollback()
            try:
                session.commit()
                logger.info("FT phase2 migration: ensured spec-aware columns exist on final_test_tracks")
            except Exception:
                pass

            # Migration: Add the STATION-REFERENCE columns (2026-09-13).
            #
            # The app's own corrected grade stays in linearity_pass — these
            # record what the SHEET said and which rows it graded, so a
            # disagreement is visible instead of invisible. graded_window_source
            # NULL is load-bearing: it marks a row graded before the window fix
            # and is what count_legacy_ft_verdicts() counts and what the
            # re-grade pass selects by default. Never backfill it with a
            # default — a default would erase the record of what needs redoing.
            ft_station_columns = {
                "final_test_results": {
                    "station_linearity_pass": "BOOLEAN",
                    "station_cell_flag_conflict": "BOOLEAN",
                    "graded_window_source": "VARCHAR(16)",
                },
                "final_test_tracks": {
                    "station_flags": "TEXT",
                    "station_fail_points": "INTEGER",
                    "graded_start": "INTEGER",
                    "graded_end": "INTEGER",
                    "ignore_start": "INTEGER",
                    "ignore_end": "INTEGER",
                },
            }
            for _table, _columns in ft_station_columns.items():
                for col_name, col_type in _columns.items():
                    try:
                        session.execute(text(
                            f"ALTER TABLE {_table} ADD COLUMN {col_name} {col_type}"
                        ))
                        session.commit()
                    except Exception as e:
                        if ("duplicate column" not in str(e).lower()
                                and "already exists" not in str(e).lower()):
                            logger.warning(
                                f"FT station-reference migration warning "
                                f"adding {_table}.{col_name}: {e}")
                        session.rollback()
            try:
                session.execute(text(
                    "CREATE INDEX IF NOT EXISTS idx_ft_graded_window_source "
                    "ON final_test_results (graded_window_source)"
                ))
                session.commit()
            except Exception as e:
                logger.warning(f"graded_window_source index warning: {e}")
                session.rollback()

            # Migration: Add consecutive_recovered column to model_ml_state
            try:
                session.execute(text("ALTER TABLE model_ml_state ADD COLUMN consecutive_recovered INTEGER DEFAULT 0"))
                session.commit()
            except Exception as e:
                if "duplicate column" not in str(e).lower() and "already exists" not in str(e).lower():
                    logger.warning(f"consecutive_recovered migration warning: {e}")
                session.rollback()

            # Migration: Add electrical_angle_tol_type to model_specs so the
            # angle-parser qualifier ('symmetric', 'min', 'max', 'range',
            # 'bilateral') is preserved. The slope-correction rule depends on
            # this to know whether a tolerance is one-sided or two-sided.
            try:
                session.execute(text(
                    "ALTER TABLE model_specs ADD COLUMN electrical_angle_tol_type VARCHAR(12)"
                ))
                session.commit()
                logger.info("Migration: Added electrical_angle_tol_type column to model_specs")
            except Exception as e:
                if "duplicate column" not in str(e).lower() and "already exists" not in str(e).lower():
                    logger.warning(f"electrical_angle_tol_type migration warning: {e}")
                session.rollback()

            # Migration: Add theory_data and test_volts columns for slope optimization
            try:
                session.execute(text("ALTER TABLE track_results ADD COLUMN theory_data TEXT"))
                session.commit()
                logger.info("Migration: Added theory_data column to track_results")
            except Exception as e:
                if "duplicate column" not in str(e).lower() and "already exists" not in str(e).lower():
                    logger.warning(f"theory_data migration warning: {e}")
                session.rollback()

            try:
                session.execute(text("ALTER TABLE track_results ADD COLUMN test_volts FLOAT"))
                session.commit()
                logger.info("Migration: Added test_volts column to track_results")
            except Exception as e:
                if "duplicate column" not in str(e).lower() and "already exists" not in str(e).lower():
                    logger.warning(f"test_volts migration warning: {e}")
                session.rollback()

            # Migration: Add theory_data to final_test_tracks for slope optimization
            try:
                session.execute(text("ALTER TABLE final_test_tracks ADD COLUMN theory_data TEXT"))
                session.commit()
                logger.info("Migration: Added theory_data column to final_test_tracks")
            except Exception as e:
                if "duplicate column" not in str(e).lower() and "already exists" not in str(e).lower():
                    logger.warning(f"ft theory_data migration: {e}")
                session.rollback()

            # Migration: Add file_size / file_modified_date to the Final Test
            # and Smoothness result tables (2026-08-29). ProcessedFile rows had
            # them; these two didn't, so the incremental scan's stat fast-path
            # could never apply to an FT/smoothness path — every known file was
            # re-HASHED (full read over the share) on EVERY scan, and the heal
            # pass only touched processed_files so it never got better. See
            # Processor._is_processed / _load_processed_hashes.
            stat_columns = {
                "file_size": "INTEGER",
                "file_modified_date": "DATETIME",
            }
            for tbl in ("final_test_results", "smoothness_results"):
                for col_name, col_type in stat_columns.items():
                    try:
                        session.execute(text(
                            f"ALTER TABLE {tbl} ADD COLUMN {col_name} {col_type}"
                        ))
                        session.commit()
                        logger.info(f"Migration: Added {col_name} column to {tbl}")
                    except Exception as e:
                        if ("duplicate column" not in str(e).lower()
                                and "already exists" not in str(e).lower()):
                            logger.warning(f"{tbl}.{col_name} migration warning: {e}")
                        session.rollback()

            # Migration: Relax NOT NULL on sigma_gradient / sigma_threshold / sigma_pass
            # so UNTRIMMED tracks (test-sweep-only files with no laser-trim runs) can
            # be saved with sigma metrics absent. SQLite can't ALTER COLUMN nullability
            # directly, so this rebuilds track_results via the rename-table pattern.
            try:
                info = session.execute(text("PRAGMA table_info(track_results)")).fetchall()
                # PRAGMA table_info columns: cid, name, type, notnull, dflt_value, pk
                sigma_grad_notnull = next(
                    (row[3] for row in info if row[1] == 'sigma_gradient'), 0
                )
                if sigma_grad_notnull == 1:
                    logger.info(
                        "Running migration: relaxing NOT NULL on sigma_gradient / "
                        "sigma_threshold / sigma_pass in track_results"
                    )
                    create_sql = session.execute(text(
                        "SELECT sql FROM sqlite_master WHERE type='table' "
                        "AND name='track_results'"
                    )).scalar()
                    indexes = session.execute(text(
                        "SELECT name, sql FROM sqlite_master WHERE type='index' "
                        "AND tbl_name='track_results' AND sql IS NOT NULL"
                    )).fetchall()

                    new_sql = create_sql
                    for _col in ('sigma_gradient', 'sigma_threshold', 'sigma_pass'):
                        # Strip the column-level NOT NULL declaration. Preserves
                        # the type/length so the column data is untouched.
                        new_sql = re.sub(
                            rf'(\b{_col}\b\s+\w+(?:\(\d+\))?)\s+NOT\s+NULL',
                            r'\1',
                            new_sql,
                            count=1,
                            flags=re.IGNORECASE,
                        )
                    new_sql = new_sql.replace(
                        'CREATE TABLE track_results',
                        'CREATE TABLE track_results_new',
                        1,
                    )

                    # NOTE: No PRAGMA foreign_keys toggling here. SQLite ignores
                    # PRAGMA foreign_keys when issued inside an open transaction,
                    # so the typical "OFF / rebuild / ON" recipe is a no-op in
                    # this session-scoped path. The rebuild is FK-safe today
                    # because no table references track_results.id — if that
                    # changes, the recipe needs to move outside this transaction
                    # via a dedicated raw connection.
                    session.execute(text(new_sql))
                    session.execute(text(
                        "INSERT INTO track_results_new SELECT * FROM track_results"
                    ))
                    session.execute(text("DROP TABLE track_results"))
                    session.execute(text(
                        "ALTER TABLE track_results_new RENAME TO track_results"
                    ))
                    for _idx_name, _idx_sql in indexes:
                        # Re-raise on failure so the outer try/except triggers a
                        # rollback. Silently losing an index would degrade query
                        # performance without any signal to the operator.
                        session.execute(text(_idx_sql))
                    session.commit()
                    logger.info(
                        "Migration completed: sigma_gradient / sigma_threshold / "
                        "sigma_pass are now nullable"
                    )
            except Exception as e:
                session.rollback()
                logger.warning(f"sigma-nullable migration warning: {e}")

        # After session closes, re-run FT matching if model names were corrected
        if needs_rematch:
            try:
                logger.info("Re-matching Final Test records after model name cleanup...")
                stats = self.rematch_final_tests()
                logger.info(f"Post-cleanup FT rematch: {stats}")
            except Exception as e:
                logger.warning(f"FT rematch after cleanup failed: {e}")

    def _backfill_unit_ids(self, session) -> None:
        """Populate analysis_results.unit_id for rows that don't have one yet.

        Idempotent: only operates on NULL unit_id rows. Logs progress and a
        post-run sanity check (count of NULL vs non-NULL).
        """
        from laser_trim_analyzer.database.models import AnalysisResult as DBAR

        # Count rows that need backfill
        to_backfill = (
            session.query(func.count(DBAR.id))
            .filter(DBAR.unit_id.is_(None))
            .scalar()
        ) or 0
        if to_backfill == 0:
            logger.debug("unit_id backfill: nothing to do")
            return

        logger.info(f"unit_id backfill: starting on {to_backfill} rows")
        batch_size = 1000
        updated = 0
        skipped = 0  # rows that have nothing to backfill (junk serial / missing date)

        # Process in batches so we don't load 80k rows into memory at once.
        # The empty-string sentinel pattern below ensures unparseable rows
        # don't keep appearing in subsequent batches (we then convert the
        # sentinels back to NULL at the end).
        while True:
            rows = (
                session.query(DBAR.id, DBAR.model, DBAR.serial, DBAR.file_date)
                .filter(DBAR.unit_id.is_(None))
                .limit(batch_size)
                .all()
            )
            if not rows:
                break

            for row_id, model, serial, file_date in rows:
                uid = compute_unit_id(model, serial, file_date)
                if uid is None:
                    # Junk serial / missing date — mark with empty-string sentinel
                    # so the next batch query doesn't pick it up again.
                    session.execute(
                        text("UPDATE analysis_results SET unit_id = '' WHERE id = :i"),
                        {"i": row_id},
                    )
                    skipped += 1
                else:
                    session.execute(
                        text("UPDATE analysis_results SET unit_id = :u WHERE id = :i"),
                        {"u": uid, "i": row_id},
                    )
                    updated += 1
            session.commit()
            logger.info(
                f"unit_id backfill: progress {updated + skipped}/{to_backfill}"
            )

        # Restore NULL for unparseable rows.
        session.execute(text(
            "UPDATE analysis_results SET unit_id = NULL WHERE unit_id = ''"
        ))
        session.commit()

        # Post-flight sanity check
        non_null = (
            session.query(func.count(DBAR.id))
            .filter(DBAR.unit_id.isnot(None))
            .scalar()
        ) or 0
        null_now = (
            session.query(func.count(DBAR.id))
            .filter(DBAR.unit_id.is_(None))
            .scalar()
        ) or 0
        distinct_units = (
            session.query(func.count(func.distinct(DBAR.unit_id)))
            .filter(DBAR.unit_id.isnot(None))
            .scalar()
        ) or 0
        logger.info(
            f"unit_id backfill complete: "
            f"{updated} populated, {skipped} junk-serial/no-date, "
            f"{non_null} non-NULL total, {null_now} NULL total, "
            f"{distinct_units} distinct units"
        )

        # Spot-check: log three sample unit_ids so an operator can verify
        # the format on customer hardware where the DB isn't accessible.
        samples = (
            session.query(DBAR.unit_id)
            .filter(DBAR.unit_id.isnot(None))
            .limit(3)
            .all()
        )
        sample_strs = [r[0] for r in samples]
        logger.info(f"unit_id backfill sample unit_ids: {sample_strs}")

