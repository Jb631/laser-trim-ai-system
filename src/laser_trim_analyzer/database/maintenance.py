"""Database maintenance: health scan, cleanup, marker/stat repair, backfills, and the "Fix
Missing Tracks" repair pair -- the cleanup / repair / backfill utilities the Settings Database
card and scripts call.

Moved out of `database/manager.py` (2026-09-25, C2 Task 5, step 5) -- a pure move, byte-identical
method bodies (see the AST proof in this step's commit). `DatabaseManager` inherits this mixin
too, alongside `MigrationsMixin`, `SpecsMixin` and `FtMatchingMixin`, so every existing call site
(`db.scan_database_health()`, `db.execute_cleanup(...)`, the Settings Database card, the
`app_qa_sweep.py` checks, etc.) is unchanged.

What did NOT move: `skip_marker_hash`, `mark_file_skipped` and `_mark_file_skipped_in` sit
between two halves of this group in manager.py (the per-path skip marker is written by the
ingest-speed Task 7 save path, not by anything here) -- they stay in manager.py, named by the
brief as a Task-7 save body. `update_processed_file_stats` calls `mark_file_skipped`'s sibling
machinery only in the sense of sharing the same `processed_files` table; it does not call it, and
moved along with the rest of this group.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import and_, case, exists, func, or_, text

from laser_trim_analyzer.database.manager import GRADEABLE_STATUS_NAMES, json_array_absent, logger
from laser_trim_analyzer.database.models import (
    AnalysisResult as DBAnalysisResult,
    ProcessedFile as DBProcessedFile,
    QAAlert as DBQAAlert,
    StatusType as DBStatusType,
    TrackResult as DBTrackResult,
    UNREADABLE_PREFIX,
)


class MaintenanceMixin:
    """Cleanup / repair / backfill utilities: database health scan and validation, the cleanup
    preview/execute pair, failed-file and skipped-file marker counters and resets, the
    processed-file stat healer, the overall-status recompute repair, the max_deviation backfill,
    and the "Fix Missing Tracks" repair pair for both Final Test and trim records.

    Inherited by `DatabaseManager`. `self.session()`/`self._write_lock`/`self._new_session()`/
    `self._map_track_to_db(...)` and other `self.`-dispatched calls resolve through the instance,
    defined elsewhere on `DatabaseManager` (or its other mixins) -- not through this module's own
    globals.
    """

    def get_final_tests_missing_tracks(self) -> List[Dict[str, Any]]:
        """
        Get Final Test records that have 0 tracks stored.

        Returns:
            List of dicts with id, filename, file_path, model
        """
        from laser_trim_analyzer.database.models import (
            FinalTestResult as DBFinalTestResult,
            FinalTestTrack as DBFinalTestTrack,
        )

        with self.session() as session:
            # Subquery to count tracks per final test
            track_count_subq = (
                session.query(
                    DBFinalTestTrack.final_test_id,
                    func.count(DBFinalTestTrack.id).label('track_count')
                )
                .group_by(DBFinalTestTrack.final_test_id)
                .subquery()
            )

            # Get Final Tests with no tracks (LEFT JOIN where track_count is NULL)
            results = (
                session.query(DBFinalTestResult)
                .outerjoin(track_count_subq, DBFinalTestResult.id == track_count_subq.c.final_test_id)
                .filter(track_count_subq.c.track_count == None)
                .all()
            )
            return [
                {
                    "id": r.id,
                    "filename": r.filename,
                    "file_path": r.file_path,
                    "model": r.model,
                    "serial": r.serial,
                }
                for r in results
            ]

    def update_final_test_tracks(
        self,
        final_test_id: int,
        tracks: List[Dict[str, Any]]
    ) -> bool:
        """
        Update track data for an existing Final Test record.

        Used to fix records that were created before parser improvements.

        Args:
            final_test_id: ID of the Final Test record
            tracks: List of track data dicts

        Returns:
            True if successful
        """
        from laser_trim_analyzer.core.ft_regrade import ft_reference_fields
        from laser_trim_analyzer.database.models import (
            FinalTestResult as DBFinalTestResult,
            FinalTestTrack as DBFinalTestTrack,
        )

        with self._write_lock:
            try:
                with self.session() as session:
                    # Get existing record
                    result = session.get(DBFinalTestResult, final_test_id)
                    if not result:
                        logger.warning(f"Final Test ID {final_test_id} not found")
                        return False

                    # Delete existing tracks (if any)
                    session.query(DBFinalTestTrack).filter(
                        DBFinalTestTrack.final_test_id == final_test_id
                    ).delete()

                    # Add new tracks
                    for track_data in tracks:
                        position_values = track_data.get("electrical_angles") or track_data.get("positions")

                        db_track = DBFinalTestTrack(
                            final_test_id=final_test_id,
                            track_id=track_data.get("track_id", "default"),
                            status=DBStatusType.PASS if track_data.get("linearity_pass", True) else DBStatusType.FAIL,
                            linearity_spec=track_data.get("linearity_spec"),
                            linearity_error=track_data.get("linearity_error"),
                            linearity_pass=track_data.get("linearity_pass"),
                            linearity_fail_points=track_data.get("linearity_fail_points", 0),
                            position_data=position_values,
                            error_data=track_data.get("errors"),
                            theory_data=track_data.get("theory_values"),
                            electrical_angle_data=track_data.get("electrical_angles"),
                            upper_limits=track_data.get("upper_limits"),
                            lower_limits=track_data.get("lower_limits"),
                            max_deviation=track_data.get("max_deviation"),
                            max_deviation_position=track_data.get("max_deviation_angle"),
                            optimal_offset=track_data.get("optimal_offset"),
                            optimal_slope=track_data.get("optimal_slope"),
                            linearity_type=track_data.get("linearity_type"),
                            **ft_reference_fields(track_data),
                        )
                        session.add(db_track)

                    # Update linearity_error on main record if tracks have it
                    if tracks and tracks[0].get("linearity_error") is not None:
                        result.linearity_error = tracks[0].get("linearity_error")

                    session.commit()
                    logger.info(f"Updated Final Test {final_test_id} with {len(tracks)} tracks")
                    return True

            except Exception as e:
                logger.error(f"Error updating Final Test tracks: {e}")
                return False

    def get_trim_records_missing_tracks(self, linked_only: bool = True) -> List[Dict[str, Any]]:
        """
        Get Trim (AnalysisResult) records whose track measurements are missing.

        Two shapes count as missing, and re-parsing the source file is the
        repair for both:

        1. ZERO track rows — the record was stored before track persistence
           existed, so there is nothing to plot or re-grade.
        2. Every track row present but ALL of them array-less (position_data
           and error_data both decode to nothing) while the parent carries a
           gradeable status. That is a unit claiming a PASS/WARNING/FAIL
           disposition with no measurement anywhere behind it.

        UNTRIMMED and ERROR parents are excluded from shape 2 ON PURPOSE.
        Array-lessness is the *expected* state for both: an UNTRIMMED file is
        a test sweep with no laser-trim run (the parser deliberately moves its
        sweep into untrimmed_positions/untrimmed_errors and clears the trimmed
        arrays — see LaserTrimParser._parse_untrimmed_only_track), and an
        ERROR row never got far enough to measure anything. On the work
        database those two account for ~5,666 of the ~5,811 array-less track
        rows; flagging them would bury any genuine defect in by-design noise
        and would send the repair tool off to re-parse thousands of files
        that would come back byte-identical.

        Partial data is likewise NOT missing: a multi-track unit with one real
        track and one array-less track keeps its measurement and its verdict,
        so it is left alone. Re-parsing it would rewrite good rows.

        Args:
            linked_only: If True, only return records that are linked to Final Tests

        Returns:
            List of record info dicts with id, filename, file_path, model, serial
        """
        with self.session() as session:
            # Per-analysis track census: how many rows, and how many of those
            # carry no arrays at all.
            track_count_subq = (
                session.query(
                    DBTrackResult.analysis_id,
                    func.count(DBTrackResult.id).label('track_count'),
                    func.sum(
                        case(
                            (and_(json_array_absent(DBTrackResult.position_data),
                                  json_array_absent(DBTrackResult.error_data)), 1),
                            else_=0,
                        )
                    ).label('empty_track_count'),
                )
                .group_by(DBTrackResult.analysis_id)
                .subquery()
            )

            no_tracks = or_(
                track_count_subq.c.track_count == None,  # noqa: E711 (SQL NULL)
                track_count_subq.c.track_count == 0,
            )
            all_tracks_empty = and_(
                track_count_subq.c.track_count > 0,
                track_count_subq.c.track_count == track_count_subq.c.empty_track_count,
                DBAnalysisResult.overall_status.in_(GRADEABLE_STATUS_NAMES),
            )

            query = (
                session.query(DBAnalysisResult)
                .outerjoin(track_count_subq, DBAnalysisResult.id == track_count_subq.c.analysis_id)
                .filter(or_(no_tracks, all_tracks_empty))
            )

            if linked_only:
                # Get IDs of analyses that are linked to Final Tests
                from laser_trim_analyzer.database.models import FinalTestResult as DBFinalTestResult
                linked_ids = (
                    session.query(DBFinalTestResult.linked_trim_id)
                    .filter(DBFinalTestResult.linked_trim_id != None)
                    .distinct()
                    .all()
                )
                linked_id_list = [lid[0] for lid in linked_ids]
                query = query.filter(DBAnalysisResult.id.in_(linked_id_list))

            results = query.all()

            return [
                {
                    "id": r.id,
                    "filename": r.filename,
                    "file_path": r.file_path,
                    "model": r.model,
                    "serial": r.serial,
                }
                for r in results
            ]

    def update_trim_tracks(
        self,
        analysis_id: int,
        tracks: List["TrackResult"]
    ) -> bool:
        """
        Update track data for an existing Trim (AnalysisResult) record.

        Used to fix records that were created before track data storage was added.

        Args:
            analysis_id: ID of the AnalysisResult record
            tracks: List of TrackResult objects from re-parsing

        Returns:
            True if successful
        """
        try:
            with self.session() as session:
                # Get existing record
                result = session.get(DBAnalysisResult, analysis_id)
                if not result:
                    logger.warning(f"Analysis ID {analysis_id} not found")
                    return False

                # Delete existing tracks (if any)
                session.query(DBTrackResult).filter(
                    DBTrackResult.analysis_id == analysis_id
                ).delete()

                # Add new tracks
                for track in tracks:
                    db_track = self._map_track_to_db(track)
                    db_track.analysis_id = analysis_id
                    session.add(db_track)

                session.commit()
                logger.info(f"Updated Analysis {analysis_id} with {len(tracks)} tracks")
                return True

        except Exception as e:
            logger.error(f"Error updating Trim tracks: {e}")
            return False

    def update_trim_tracks_from_final_test(
        self,
        analysis_id: int,
        ft_tracks: List[Dict[str, Any]]
    ) -> bool:
        """
        Update Trim (AnalysisResult) track data from Final Test format data.

        Used when "Trim" records actually point to Final Test files.
        Converts FT track format to TrackResult format.

        Args:
            analysis_id: ID of the AnalysisResult record
            ft_tracks: List of track dicts from Final Test parser

        Returns:
            True if successful
        """
        try:
            with self.session() as session:
                # Get existing record
                result = session.get(DBAnalysisResult, analysis_id)
                if not result:
                    logger.warning(f"Analysis ID {analysis_id} not found")
                    return False

                # Delete existing tracks (if any)
                session.query(DBTrackResult).filter(
                    DBTrackResult.analysis_id == analysis_id
                ).delete()

                # Add new tracks converted from FT format
                for ft_track in ft_tracks:
                    # Get position data (FT format uses electrical_angles)
                    positions = ft_track.get("electrical_angles") or ft_track.get("positions", [])
                    errors = ft_track.get("errors", [])
                    upper_limits = ft_track.get("upper_limits", [])
                    lower_limits = ft_track.get("lower_limits", [])

                    # Calculate linearity metrics
                    linearity_error = ft_track.get("linearity_error", 0.0)
                    linearity_spec = ft_track.get("linearity_spec", 0.02)
                    linearity_pass = ft_track.get("linearity_pass", True)

                    # Create TrackResult-compatible DB record
                    db_track = DBTrackResult(
                        analysis_id=analysis_id,
                        track_id=ft_track.get("track_id", "default"),
                        status=DBStatusType.PASS if linearity_pass else DBStatusType.FAIL,
                        # Sigma values - use defaults for FT data
                        sigma_gradient=0.0,
                        sigma_threshold=1.0,
                        sigma_pass=True,
                        # Linearity values
                        linearity_spec=linearity_spec,
                        final_linearity_error_shifted=linearity_error,
                        linearity_pass=linearity_pass,
                        linearity_fail_points=ft_track.get("linearity_fail_points", 0),
                        # Track data for charts
                        position_data=positions,
                        error_data=errors,
                        upper_limits=upper_limits,
                        lower_limits=lower_limits,
                        # Travel length from position range
                        travel_length=max(positions) - min(positions) if positions and len(positions) > 1 else 1.0,
                    )
                    session.add(db_track)

                session.commit()
                logger.info(f"Updated Analysis {analysis_id} with {len(ft_tracks)} tracks from FT format")
                return True

        except Exception as e:
            logger.error(f"Error updating Trim tracks from FT: {e}")
            return False
    def scan_database_health(self) -> Dict[str, Any]:
        """
        Scan the entire database and return a health report.

        Identifies dirty/suspect records across multiple categories without
        modifying anything. Returns counts and record IDs for each issue.
        """
        health = {
            "total_analyses": 0,
            "total_tracks": 0,
            "issues": {},
            "total_dirty_records": 0,
        }

        with self.session() as session:
            health["total_analyses"] = (
                session.query(func.count(DBAnalysisResult.id)).scalar() or 0
            )
            health["total_tracks"] = (
                session.query(func.count(DBTrackResult.id)).scalar() or 0
            )

            dirty_ids = set()

            # 1. Unknown model
            unknown_model = session.query(DBAnalysisResult.id).filter(
                DBAnalysisResult.model == "Unknown"
            ).all()
            if unknown_model:
                ids = {r[0] for r in unknown_model}
                dirty_ids |= ids
                health["issues"]["unknown_model"] = {
                    "count": len(ids),
                    "label": "Unknown model (parser couldn't extract)",
                }

            # 2. Unknown serial
            unknown_serial = session.query(DBAnalysisResult.id).filter(
                DBAnalysisResult.serial == "Unknown"
            ).all()
            if unknown_serial:
                ids = {r[0] for r in unknown_serial}
                dirty_ids |= ids
                health["issues"]["unknown_serial"] = {
                    "count": len(ids),
                    "label": "Unknown serial number",
                }

            # 3. Missing file date
            null_date = session.query(DBAnalysisResult.id).filter(
                DBAnalysisResult.file_date.is_(None)
            ).all()
            if null_date:
                ids = {r[0] for r in null_date}
                dirty_ids |= ids
                health["issues"]["missing_file_date"] = {
                    "count": len(ids),
                    "label": "Missing file date",
                }

            # 4. ERROR status records
            error_records = session.query(DBAnalysisResult.id).filter(
                DBAnalysisResult.overall_status == DBStatusType.ERROR
            ).all()
            if error_records:
                ids = {r[0] for r in error_records}
                dirty_ids |= ids
                health["issues"]["error_status"] = {
                    "count": len(ids),
                    "label": "ERROR status (processing failed)",
                }

            # 5. Analyses with no tracks (orphaned)
            analyses_no_tracks = session.query(DBAnalysisResult.id).filter(
                ~exists().where(DBTrackResult.analysis_id == DBAnalysisResult.id)
            ).all()
            if analyses_no_tracks:
                ids = {r[0] for r in analyses_no_tracks}
                dirty_ids |= ids
                health["issues"]["no_tracks"] = {
                    "count": len(ids),
                    "label": "No track data (empty analyses)",
                }

            # 6. Track-level quality issues (negative sigma, all-zero data, etc.)
            bad_sigma = session.query(
                DBTrackResult.analysis_id
            ).filter(
                DBTrackResult.sigma_gradient < 0
            ).distinct().all()
            if bad_sigma:
                ids = {r[0] for r in bad_sigma}
                dirty_ids |= ids
                health["issues"]["negative_sigma"] = {
                    "count": len(ids),
                    "label": "Negative sigma gradient (impossible value)",
                }

            # 7. Tracks with no spec limits (can't determine pass/fail)
            no_limits = session.query(
                DBTrackResult.analysis_id
            ).filter(
                DBTrackResult.upper_limits.is_(None),
                DBTrackResult.lower_limits.is_(None),
                DBTrackResult.linearity_spec.is_(None),
            ).distinct().all()
            if no_limits:
                ids = {r[0] for r in no_limits}
                dirty_ids |= ids
                health["issues"]["no_spec_limits"] = {
                    "count": len(ids),
                    "label": "No spec limits (can't verify pass/fail)",
                }

            # 8. Already-flagged suspect quality
            suspect = session.query(DBAnalysisResult.id).filter(
                DBAnalysisResult.data_quality == "suspect"
            ).all()
            if suspect:
                ids = {r[0] for r in suspect}
                dirty_ids |= ids
                health["issues"]["suspect_quality"] = {
                    "count": len(ids),
                    "label": "Previously flagged as suspect",
                }

            health["total_dirty_records"] = len(dirty_ids)

        return health

    def retroactive_validate(self) -> Dict[str, Any]:
        """
        Retroactively validate ALL records in the database and update
        data_quality flags.

        Checks analysis-level and track-level quality issues, then
        updates the data_quality and data_quality_issues columns.

        Uses raw SQL updates to avoid SQLAlchemy dirty-tracking issues
        with JSON (list) columns that are unhashable.

        Returns summary of what was found and updated.
        """
        from sqlalchemy import update as sa_update

        summary = {"scanned": 0, "flagged": 0, "already_suspect": 0, "issues_by_type": {}}
        batch_size = 1000

        with self._write_lock:
            with self.session() as session:
                total = session.query(func.count(DBAnalysisResult.id)).scalar() or 0
                summary["scanned"] = total

                # Process in batches to avoid memory issues with large databases
                for offset in range(0, total, batch_size):
                    # Use read-only loading — we'll update via raw SQL to avoid
                    # SQLAlchemy dirty-tracking on JSON (list) columns
                    analyses = session.query(
                        DBAnalysisResult.id,
                        DBAnalysisResult.model,
                        DBAnalysisResult.serial,
                        DBAnalysisResult.file_date,
                        DBAnalysisResult.data_quality,
                    ).order_by(DBAnalysisResult.id).offset(offset).limit(batch_size).all()

                    for a_id, a_model, a_serial, a_file_date, a_dq in analyses:
                        issues = []

                        # Analysis-level checks
                        if a_model == "Unknown":
                            issues.append("Unknown model")
                        if a_serial == "Unknown":
                            issues.append("Unknown serial")
                        if a_file_date is None:
                            issues.append("Missing file date")

                        # Track-level checks — query track columns directly
                        tracks = session.query(
                            DBTrackResult.track_id,
                            DBTrackResult.sigma_gradient,
                            DBTrackResult.linearity_spec,
                            DBTrackResult.upper_limits,
                            DBTrackResult.lower_limits,
                            DBTrackResult.position_data,
                            DBTrackResult.error_data,
                        ).filter(
                            DBTrackResult.analysis_id == a_id
                        ).all()

                        if not tracks:
                            issues.append("No track data")

                        for t_id, t_sigma, t_lin_spec, t_upper, t_lower, t_pos, t_err in tracks:
                            tid = t_id or "?"

                            if t_sigma is not None and t_sigma < 0:
                                issues.append(f"{tid}: negative sigma_gradient ({t_sigma:.4f})")

                            if not t_upper and not t_lower and t_lin_spec is None:
                                issues.append(f"{tid}: no spec limits")

                            if t_err:
                                try:
                                    if all(v == 0 or v is None for v in t_err):
                                        issues.append(f"{tid}: all-zero error data")
                                except (TypeError, ValueError):
                                    issues.append(f"{tid}: corrupt error data")

                            if t_pos:
                                try:
                                    if len(t_pos) < 10:
                                        issues.append(f"{tid}: too few data points ({len(t_pos)})")
                                except TypeError:
                                    issues.append(f"{tid}: corrupt position data")

                            if t_pos and t_err:
                                try:
                                    if len(t_pos) != len(t_err):
                                        issues.append(
                                            f"{tid}: array mismatch (pos={len(t_pos)}, err={len(t_err)})"
                                        )
                                except TypeError:
                                    pass

                        # Update via raw SQL to avoid unhashable-list errors from JSON columns
                        if issues:
                            was_suspect = a_dq == "suspect"
                            session.execute(
                                sa_update(DBAnalysisResult)
                                .where(DBAnalysisResult.id == a_id)
                                .values(
                                    data_quality="suspect",
                                    data_quality_issues=", ".join(issues),
                                )
                            )
                            if was_suspect:
                                summary["already_suspect"] += 1
                            else:
                                summary["flagged"] += 1

                            for issue in issues:
                                category = issue.split(":")[0].strip() if ":" in issue else issue
                                summary["issues_by_type"][category] = summary["issues_by_type"].get(category, 0) + 1
                        else:
                            if a_dq == "suspect":
                                session.execute(
                                    sa_update(DBAnalysisResult)
                                    .where(DBAnalysisResult.id == a_id)
                                    .values(
                                        data_quality="good",
                                        data_quality_issues=None,
                                    )
                                )

                    session.flush()

                logger.info(
                    f"Retroactive validation: scanned {summary['scanned']}, "
                    f"flagged {summary['flagged']} new, "
                    f"{summary['already_suspect']} already suspect"
                )

        return summary

    def _collect_cleanup_ids(
        self,
        session,
        delete_non_mps: bool = False,
        mps_models: Optional[List[str]] = None,
        delete_before_date: Optional[datetime] = None,
        delete_suspect_quality: bool = False,
        delete_unknown: bool = False,
        delete_error_status: bool = False,
        delete_no_tracks: bool = False,
        delete_misclassified_ft: bool = False,
    ) -> tuple:
        """
        Collect record IDs matching cleanup criteria. Shared by preview and execute.

        Returns:
            (ids_to_delete set, by_reason dict)
        """
        ids_to_delete = set()
        by_reason = {}

        if delete_non_mps and mps_models:
            mps_set = set(m.strip() for m in mps_models if m.strip())
            non_mps = session.query(
                DBAnalysisResult.id, DBAnalysisResult.model
            ).filter(
                DBAnalysisResult.model.notin_(mps_set)
            ).all()
            non_mps_ids = {r[0] for r in non_mps}
            non_mps_models = sorted(set(r[1] for r in non_mps))
            ids_to_delete |= non_mps_ids
            by_reason["non_mps_models"] = {
                "count": len(non_mps_ids),
                "models": non_mps_models,
            }

        if delete_before_date:
            old_records = session.query(
                DBAnalysisResult.id
            ).filter(
                DBAnalysisResult.file_date < delete_before_date
            ).all()
            old_ids = {r[0] for r in old_records}
            ids_to_delete |= old_ids
            by_reason["before_date"] = {
                "count": len(old_ids),
                "date": delete_before_date.strftime("%Y-%m-%d"),
            }

        if delete_suspect_quality:
            suspect = session.query(
                DBAnalysisResult.id
            ).filter(
                DBAnalysisResult.data_quality == "suspect"
            ).all()
            suspect_ids = {r[0] for r in suspect}
            ids_to_delete |= suspect_ids
            by_reason["suspect_quality"] = {
                "count": len(suspect_ids),
            }

        if delete_unknown:
            unknown = session.query(
                DBAnalysisResult.id
            ).filter(
                or_(
                    DBAnalysisResult.model == "Unknown",
                    DBAnalysisResult.serial == "Unknown",
                )
            ).all()
            unknown_ids = {r[0] for r in unknown}
            ids_to_delete |= unknown_ids
            by_reason["unknown_model_serial"] = {
                "count": len(unknown_ids),
            }

        if delete_error_status:
            errors = session.query(
                DBAnalysisResult.id
            ).filter(
                DBAnalysisResult.overall_status == DBStatusType.ERROR
            ).all()
            error_ids = {r[0] for r in errors}
            ids_to_delete |= error_ids
            by_reason["error_status"] = {
                "count": len(error_ids),
            }

        if delete_no_tracks:
            no_tracks = session.query(
                DBAnalysisResult.id
            ).filter(
                ~exists().where(DBTrackResult.analysis_id == DBAnalysisResult.id)
            ).all()
            no_track_ids = {r[0] for r in no_tracks}
            ids_to_delete |= no_track_ids
            by_reason["no_tracks"] = {
                "count": len(no_track_ids),
            }

        if delete_misclassified_ft:
            # Find trim records that are actually Final Test files:
            # 1. Files from "Test Station" paths
            # 2. Files with _Redundant_ or _Primary_ in filename
            # 3. Files with "final" followed by a number in filename
            ft_patterns = [
                DBAnalysisResult.filename.like("%Test Station%"),
                DBAnalysisResult.filename.like("%test station%"),
                DBAnalysisResult.filename.like("%_Redundant_%"),
                DBAnalysisResult.filename.like("%_redundant_%"),
                DBAnalysisResult.filename.like("%_Primary_%"),
                DBAnalysisResult.filename.like("%_primary_%"),
            ]
            misclassified = session.query(
                DBAnalysisResult.id
            ).filter(
                or_(*ft_patterns)
            ).all()
            misc_ids = {r[0] for r in misclassified}

            # Also find "model final NNN" pattern files in trim table
            final_pattern = session.query(
                DBAnalysisResult.id
            ).filter(
                DBAnalysisResult.filename.like("% final %")
            ).all()
            final_ids = {r[0] for r in final_pattern}
            misc_ids |= final_ids

            ids_to_delete |= misc_ids
            by_reason["misclassified_ft"] = {
                "count": len(misc_ids),
            }

        return ids_to_delete, by_reason

    def preview_cleanup(
        self,
        delete_non_mps: bool = False,
        mps_models: Optional[List[str]] = None,
        delete_before_date: Optional[datetime] = None,
        delete_suspect_quality: bool = False,
        delete_unknown: bool = False,
        delete_error_status: bool = False,
        delete_no_tracks: bool = False,
        delete_misclassified_ft: bool = False,
    ) -> Dict[str, Any]:
        """
        Preview what a cleanup operation would delete WITHOUT actually deleting.

        Returns:
            Dict with counts and model lists for what would be deleted
        """
        preview = {
            "total_records": 0,
            "records_to_delete": 0,
            "models_to_delete": [],
            "by_reason": {},
        }

        with self.session() as session:
            preview["total_records"] = (
                session.query(func.count(DBAnalysisResult.id)).scalar() or 0
            )

            ids_to_delete, by_reason = self._collect_cleanup_ids(
                session,
                delete_non_mps=delete_non_mps,
                mps_models=mps_models,
                delete_before_date=delete_before_date,
                delete_suspect_quality=delete_suspect_quality,
                delete_unknown=delete_unknown,
                delete_error_status=delete_error_status,
                delete_no_tracks=delete_no_tracks,
                delete_misclassified_ft=delete_misclassified_ft,
            )

            preview["by_reason"] = by_reason
            preview["records_to_delete"] = len(ids_to_delete)

            if ids_to_delete:
                models = session.query(
                    DBAnalysisResult.model
                ).filter(
                    DBAnalysisResult.id.in_(ids_to_delete)
                ).distinct().all()
                preview["models_to_delete"] = sorted(m[0] for m in models)

        return preview

    def execute_cleanup(
        self,
        delete_non_mps: bool = False,
        mps_models: Optional[List[str]] = None,
        delete_before_date: Optional[datetime] = None,
        delete_suspect_quality: bool = False,
        delete_unknown: bool = False,
        delete_error_status: bool = False,
        delete_no_tracks: bool = False,
        delete_misclassified_ft: bool = False,
    ) -> Dict[str, int]:
        """
        Execute database cleanup — permanently delete matching records.

        Uses the same filters as preview_cleanup(). Deletes analysis records
        and associated tracks and alerts. Keeps processed_files records so
        the same bad files won't be reprocessed next time (the FK has
        ondelete=SET NULL so the link is safely cleared).

        Returns:
            Dict with deletion counts
        """
        deleted = {"analyses": 0, "tracks": 0, "alerts": 0}

        with self._write_lock:
            with self.session() as session:
                ids_to_delete, _ = self._collect_cleanup_ids(
                    session,
                    delete_non_mps=delete_non_mps,
                    mps_models=mps_models,
                    delete_before_date=delete_before_date,
                    delete_suspect_quality=delete_suspect_quality,
                    delete_unknown=delete_unknown,
                    delete_error_status=delete_error_status,
                    delete_no_tracks=delete_no_tracks,
                    delete_misclassified_ft=delete_misclassified_ft,
                )

                if not ids_to_delete:
                    return deleted

                # Delete in batches to avoid SQLite variable limits
                id_list = list(ids_to_delete)
                batch_size = 500

                for i in range(0, len(id_list), batch_size):
                    batch = id_list[i:i + batch_size]

                    deleted["tracks"] += session.query(DBTrackResult).filter(
                        DBTrackResult.analysis_id.in_(batch)
                    ).delete(synchronize_session=False)

                    deleted["alerts"] += session.query(DBQAAlert).filter(
                        DBQAAlert.analysis_id.in_(batch)
                    ).delete(synchronize_session=False)

                    # Keep processed_files records — prevents reprocessing
                    # the same bad files. FK ondelete=SET NULL clears the link.

                    deleted["analyses"] += session.query(DBAnalysisResult).filter(
                        DBAnalysisResult.id.in_(batch)
                    ).delete(synchronize_session=False)

                logger.info(
                    f"Database cleanup: deleted {deleted['analyses']} analyses, "
                    f"{deleted['tracks']} tracks, {deleted['alerts']} alerts "
                    f"(processed_files kept to prevent reprocessing)"
                )

        return deleted

    def count_failed_file_markers(self) -> int:
        """Count files being skipped because they FAILED TO READ (2026-09-17).

        A subset of `count_skipped_files`: same rows, narrowed to the ones
        whose reason carries `UNREADABLE_PREFIX`. That tag, not "has a reason",
        is the discriminator — every marker has a reason (`mark_file_skipped`
        records the content hash in it), and the duplicate markers say "same
        content as final_test_results id N". Scoping on NOT NULL would sweep
        in the 8,114 non-trim files and every duplicate.
        """
        with self.session() as session:
            return session.query(func.count(DBProcessedFile.id)).filter(
                DBProcessedFile.analysis_id.is_(None),
                DBProcessedFile.success == True,
                DBProcessedFile.error_message.like(UNREADABLE_PREFIX + "%"),
            ).scalar() or 0

    def reset_failed_file_markers(self) -> int:
        """Forget the "could not read this" markers so the files are re-offered.

        The escape hatch behind Settings → "Retry unreadable files": what a
        parser upgrade needs, and nothing more. Non-trim markers and duplicate
        markers are left in place — they did not fail to read, and re-offering
        them would undo the thing the markers exist for.

        Trim ERROR rows (success=False, with their analysis) are untouched:
        they are what the cleanup tools and the scan's "retrying earlier
        errors" line read. Dropping the marker is what makes the file eligible
        again; the file is then re-processed and, if it fails again, re-marked.

        Returns the number of markers cleared.
        """
        with self._write_lock:
            with self.session() as session:
                count = session.query(DBProcessedFile).filter(
                    DBProcessedFile.analysis_id.is_(None),
                    DBProcessedFile.success == True,
                    DBProcessedFile.error_message.like(UNREADABLE_PREFIX + "%"),
                ).delete(synchronize_session=False)

                logger.info(f"Cleared {count} unreadable-file markers; those "
                            f"files will be offered again on the next run")

        return count

    def count_skipped_files(self) -> int:
        """Count non-trim/non-FT files that were skipped and recorded."""
        with self.session() as session:
            return session.query(func.count(DBProcessedFile.id)).filter(
                DBProcessedFile.analysis_id.is_(None),
                DBProcessedFile.success == True,
            ).scalar() or 0

    def reset_skipped_files(self) -> int:
        """
        Remove processed_files entries for skipped non-trim files so they
        get re-evaluated on the next processing run.

        Only clears entries with analysis_id=NULL (no analysis was created),
        which are files that were detected as non-trim and skipped.

        Returns:
            Number of entries cleared
        """
        with self._write_lock:
            with self.session() as session:
                count = session.query(DBProcessedFile).filter(
                    DBProcessedFile.analysis_id.is_(None),
                    DBProcessedFile.success == True,
                ).delete(synchronize_session=False)

                logger.info(f"Reset {count} skipped file entries for reprocessing")

        return count
    def update_processed_file_stats(self, entries) -> Dict[str, int]:
        """Repair size/mtime on processed rows after a hash-confirm.

        entries: iterable of (file_hash, file_size, file_modified_date).
        Lets the incremental scan's stat fast-path work on the next run for
        rows whose recorded stat was missing or stale. Content identity is
        unchanged — only rows matched by their content hash are updated.

        Covers final_test_results and smoothness_results as well as
        processed_files (2026-08-29): FT rows never carried a stat, so the
        heal never reached them and every scan re-hashed the whole FT share.
        The first scan after this ships hash-confirms once, stamps the rows,
        and every scan after that is pure in-memory.

        Returns rows updated PER TABLE plus "total". Per-table counts, not one
        number, because the old single count hid the bug for six weeks: the
        scan logged "Repaired stat records for 150,938 processed files" while
        the UPDATE matched ~0 rows (FT files have no processed_files row).
        The caller logs queued-vs-updated so a silent no-op can't hide again.
        """
        from laser_trim_analyzer.database.models import (
            FinalTestResult as DBFinalTestResult,
            SmoothnessResult as DBSmoothnessResult,
        )

        tables = (("processed_files", DBProcessedFile),
                  ("final_test_results", DBFinalTestResult),
                  ("smoothness_results", DBSmoothnessResult))
        counts: Dict[str, int] = {name: 0 for name, _ in tables}
        with self._write_lock:
            with self.session() as session:
                for file_hash, file_size, file_modified_date in entries:
                    for name, model in tables:
                        counts[name] += session.query(model).filter(
                            model.file_hash == file_hash
                        ).update({
                            model.file_size: file_size,
                            model.file_modified_date: file_modified_date,
                        })
        counts["total"] = sum(counts[name] for name, _ in tables)
        return counts

    def recompute_overall_statuses(self, dry_run: bool = True,
                                   batch_size: int = 1000) -> Dict[str, Any]:
        """Re-grade every analysis' overall_status from its tracks' STORED
        pass flags, using the current (correct) rule. (2026-07-07, M4.)

        Why: ~42%% of historical rows are WARNING with three different
        historical meanings (old rule labeled linearity-FAILs as Warning;
        later gate changes were never backfilled). Linearity is a zero-
        tolerance customer requirement — a linearity-FAIL presenting as
        "Warning" is a misclassification, not a cosmetic quirk.

        Rule (mirrors analyzer._determine + processor rollup):
          track: FAIL if linearity_pass is False; else PASS if sigma_pass is
                 True; else WARNING. UNTRIMMED tracks excluded from judging.
          analysis: all-PASS -> PASS; any FAIL -> FAIL; else WARNING.

        Safety: analyses where any judged track has linearity_pass = NULL are
        SKIPPED (never regraded) — those are the un-evaluated/empty-array rows
        that Fix Missing Tracks must repair first. ERROR and UNTRIMMED
        analyses are untouched. dry_run=True only counts.

        Returns {"examined", "changed", "skipped_null_flags", "transitions":
        {"OLD->NEW": n}, "sample_changed_ids": [...]}.
        """
        from collections import defaultdict

        out: Dict[str, Any] = {"examined": 0, "changed": 0,
                               "skipped_null_flags": 0,
                               "transitions": defaultdict(int),
                               "sample_changed_ids": []}
        updates: List[tuple] = []  # (analysis_id, new_status)

        with self.session() as session:
            rows = (session.query(
                        DBAnalysisResult.id, DBAnalysisResult.overall_status,
                        DBTrackResult.status, DBTrackResult.linearity_pass,
                        DBTrackResult.sigma_pass)
                    .join(DBTrackResult,
                          DBTrackResult.analysis_id == DBAnalysisResult.id)
                    .filter(DBAnalysisResult.overall_status.notin_(
                        [DBStatusType.UNTRIMMED, DBStatusType.ERROR]))
                    .order_by(DBAnalysisResult.id)
                    .yield_per(5000))

            current: Dict[int, Any] = {}
            tracks_by_analysis: Dict[int, list] = {}
            for aid, overall, tstatus, lin, sig in rows:
                current[aid] = overall
                tracks_by_analysis.setdefault(aid, []).append((tstatus, lin, sig))

            for aid, tracks in tracks_by_analysis.items():
                out["examined"] += 1
                judged = [(lin, sig) for (tstatus, lin, sig) in tracks
                          if getattr(tstatus, "name", str(tstatus)) != "UNTRIMMED"]
                if not judged:
                    continue
                if any(lin is None for (lin, _sig) in judged):
                    out["skipped_null_flags"] += 1
                    continue
                track_statuses = [
                    DBStatusType.FAIL if lin is False
                    else (DBStatusType.PASS if sig is True else DBStatusType.WARNING)
                    for (lin, sig) in judged
                ]
                if all(s == DBStatusType.PASS for s in track_statuses):
                    new = DBStatusType.PASS
                elif any(s == DBStatusType.FAIL for s in track_statuses):
                    new = DBStatusType.FAIL
                else:
                    new = DBStatusType.WARNING

                old = current[aid]
                old_name = getattr(old, "name", str(old))
                if old_name != new.name:
                    out["changed"] += 1
                    out["transitions"][f"{old_name}->{new.name}"] += 1
                    if len(out["sample_changed_ids"]) < 10:
                        out["sample_changed_ids"].append(aid)
                    updates.append((aid, new))

        out["transitions"] = dict(out["transitions"])
        if dry_run or not updates:
            return out

        # Execute in batches; partial progress is preserved on error (same
        # philosophy as backfill_max_deviation).
        with self._write_lock:
            with self.session() as session:
                for i in range(0, len(updates), batch_size):
                    for aid, new in updates[i:i + batch_size]:
                        session.query(DBAnalysisResult).filter(
                            DBAnalysisResult.id == aid
                        ).update({DBAnalysisResult.overall_status: new},
                                 synchronize_session=False)
                    session.commit()
        logger.info(f"Status recompute: {out['changed']} of {out['examined']} "
                    f"regraded; {out['skipped_null_flags']} skipped (NULL flags); "
                    f"transitions={out['transitions']}")
        return out

    def backfill_max_deviation(self, batch_size: int = 1000) -> int:
        """
        Backfill max_deviation, max_deviation_position, and deviation_uniformity
        for existing tracks that have error_data but no max_deviation.

        Commits in batches so that partial progress is preserved if an error
        occurs mid-way.  This is intentional — a backfill that saves 900 of
        1000 rows is better than one that saves 0.

        Returns:
            Number of tracks updated (may be partial on error)
        """
        import json
        import statistics as stats_module

        updated = 0
        with self._write_lock:
            session = self._new_session()
            try:
                # Get total count first
                total = session.execute(text(
                    "SELECT COUNT(*) FROM track_results "
                    "WHERE max_deviation IS NULL AND error_data IS NOT NULL"
                )).scalar()

                if total == 0:
                    logger.info("No tracks need max_deviation backfill")
                    return 0

                logger.info(f"Backfilling max_deviation for {total} tracks...")

                last_id = 0
                while True:
                    rows = session.execute(text(
                        "SELECT id, error_data, position_data, optimal_offset "
                        "FROM track_results "
                        "WHERE max_deviation IS NULL AND error_data IS NOT NULL "
                        "AND id > :last_id "
                        "ORDER BY id LIMIT :limit"
                    ), {"limit": batch_size, "last_id": last_id}).fetchall()

                    if not rows:
                        break
                    last_id = rows[-1].id

                    for row in rows:
                        try:
                            errors = json.loads(row.error_data) if isinstance(row.error_data, str) else row.error_data
                            positions = json.loads(row.position_data) if isinstance(row.position_data, str) else row.position_data
                            opt_offset = row.optimal_offset or 0.0

                            if not errors or not positions:
                                continue

                            shifted = [e + opt_offset for e in errors]
                            abs_errs = [abs(e) for e in shifted]
                            max_dev = max(abs_errs)
                            max_idx = abs_errs.index(max_dev)
                            max_dev_pos = positions[max_idx] if max_idx < len(positions) else None

                            dev_unif = None
                            if len(abs_errs) > 1:
                                mean_abs = stats_module.mean(abs_errs)
                                if mean_abs > 0:
                                    dev_unif = stats_module.stdev(abs_errs) / mean_abs

                            session.execute(text(
                                "UPDATE track_results SET "
                                "max_deviation = :max_dev, "
                                "max_deviation_position = :max_dev_pos, "
                                "deviation_uniformity = :dev_unif "
                                "WHERE id = :id"
                            ), {
                                "max_dev": max_dev,
                                "max_dev_pos": max_dev_pos,
                                "dev_unif": dev_unif,
                                "id": row.id
                            })
                            updated += 1
                        except Exception as e:
                            logger.warning(f"Failed to backfill track {row.id}: {e}")

                    session.commit()
                    logger.info(f"Backfilled {updated}/{total} tracks...")

            except Exception as e:
                session.rollback()
                logger.error(f"Backfill error after {updated} updates: {e}")
            finally:
                session.close()

        logger.info(f"Backfill complete: {updated} tracks updated")
        return updated
