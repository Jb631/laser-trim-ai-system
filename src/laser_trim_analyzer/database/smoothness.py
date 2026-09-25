"""Output Smoothness save/read paths: the missing-tracks repair pair, search, single-result
read, and the fleet/per-model stats readers.

Moved out of `database/manager.py` (2026-09-25, C2 Task 5, step 6 -- the last step; §7 says stop
after this one) -- a pure move, byte-identical method bodies (see the AST proof in this step's
commit). `DatabaseManager` inherits this mixin too, alongside `MigrationsMixin`, `SpecsMixin`,
`FtMatchingMixin` and `MaintenanceMixin`, so every existing call site (the Smoothness page,
`scripts/fix_smoothness_tracks.py`, etc.) is unchanged.

What did NOT move: `save_smoothness_result` / `_save_smoothness_in` are the ingest-speed Task 7
save body for this table and stay in manager.py, immediately above where this group used to
start.
"""
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import case, desc, func

from laser_trim_analyzer.database.manager import logger
from laser_trim_analyzer.database.models import StatusType as DBStatusType


class SmoothnessMixin:
    """Output Smoothness reads and the missing-tracks repair pair: the "Fix Missing Tracks"
    counterpart for `search_smoothness_results` (the Smoothness page's search), a single-result
    read, and the fleet-wide / per-model stats readers.

    Inherited by `DatabaseManager`. `self.session()`/`self._write_lock` and other
    `self.`-dispatched calls resolve through the instance, defined elsewhere on `DatabaseManager`
    (or its other mixins) -- not through this module's own globals.
    """

    def get_smoothness_files_missing_tracks(self) -> List[Dict[str, Any]]:
        """
        Get Output Smoothness records that have 0 tracks stored.

        Used to repair records imported by the older code that wrote the
        result row but did not persist the per-position arrays needed to
        render the chart.

        Returns:
            List of dicts with id, filename, file_path, model, serial.
        """
        from laser_trim_analyzer.database.models import (
            SmoothnessResult as DBSmoothnessResult,
            SmoothnessTrack as DBSmoothnessTrack,
        )

        with self.session() as session:
            track_count_subq = (
                session.query(
                    DBSmoothnessTrack.smoothness_id,
                    func.count(DBSmoothnessTrack.id).label('track_count')
                )
                .group_by(DBSmoothnessTrack.smoothness_id)
                .subquery()
            )

            results = (
                session.query(DBSmoothnessResult)
                .outerjoin(
                    track_count_subq,
                    DBSmoothnessResult.id == track_count_subq.c.smoothness_id,
                )
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

    def update_smoothness_tracks(
        self,
        smoothness_id: int,
        tracks: List[Dict[str, Any]],
    ) -> bool:
        """
        Replace the per-track data for an existing Smoothness record.

        Used to fix records that were imported before the smoothness_tracks
        write was added to save_smoothness_result.
        """
        from laser_trim_analyzer.database.models import (
            SmoothnessTrack as DBSmoothnessTrack,
        )

        if not tracks:
            return False

        with self._write_lock:
            try:
                with self.session() as session:
                    # Delete any existing (likely zero) tracks first
                    session.query(DBSmoothnessTrack).filter(
                        DBSmoothnessTrack.smoothness_id == smoothness_id
                    ).delete(synchronize_session=False)

                    for track_data in tracks:
                        db_track = DBSmoothnessTrack(
                            smoothness_id=smoothness_id,
                            track_id=track_data.get("track_id", "default"),
                            status=DBStatusType.PASS if track_data.get("smoothness_pass", True) else DBStatusType.FAIL,
                            smoothness_spec=track_data.get("smoothness_spec"),
                            max_smoothness=track_data.get("max_smoothness"),
                            avg_smoothness=track_data.get("avg_smoothness"),
                            smoothness_pass=track_data.get("smoothness_pass"),
                            position_data=track_data.get("positions"),
                            smoothness_data=track_data.get("smoothness_values"),
                        )
                        session.add(db_track)
                    return True
            except Exception as e:
                logger.error(f"update_smoothness_tracks({smoothness_id}) failed: {e}")
                return False

    def search_smoothness_results(
        self, model: Optional[str] = None, limit: int = 500
    ) -> List[Dict[str, Any]]:
        """Search Output Smoothness results."""
        from laser_trim_analyzer.database.models import SmoothnessResult as DBSmoothnessResult

        with self.session() as session:
            query = session.query(DBSmoothnessResult)
            if model and model != "All Models":
                query = query.filter(DBSmoothnessResult.model == model)
            results = query.order_by(desc(DBSmoothnessResult.file_date)).limit(limit).all()
            return [
                {
                    "id": r.id, "filename": r.filename, "model": r.model,
                    "serial": r.serial, "element_label": r.element_label,
                    "file_date": r.file_date, "test_date": r.test_date,
                    "overall_status": r.overall_status.value if r.overall_status else "UNKNOWN",
                    "smoothness_spec": r.smoothness_spec,
                    "max_smoothness_value": r.max_smoothness_value,
                    "avg_smoothness_value": r.avg_smoothness_value,
                    "smoothness_pass": r.smoothness_pass,
                    "linked_trim_id": r.linked_trim_id,
                    "match_confidence": r.match_confidence,
                    "match_method": r.match_method,
                }
                for r in results
            ]

    def get_smoothness_result(self, result_id: int) -> Optional[Dict[str, Any]]:
        """Get a single Output Smoothness result by ID with tracks."""
        from laser_trim_analyzer.database.models import (
            SmoothnessResult as DBSmoothnessResult,
            SmoothnessTrack as DBSmoothnessTrack,
        )
        with self.session() as session:
            result = session.query(DBSmoothnessResult).filter(
                DBSmoothnessResult.id == result_id
            ).first()
            if not result:
                return None
            tracks = session.query(DBSmoothnessTrack).filter(
                DBSmoothnessTrack.smoothness_id == result_id
            ).all()
            return {
                "id": result.id, "filename": result.filename,
                "model": result.model, "serial": result.serial,
                "element_label": result.element_label,
                "file_date": result.file_date, "test_date": result.test_date,
                "overall_status": result.overall_status.value if result.overall_status else "UNKNOWN",
                "smoothness_spec": result.smoothness_spec,
                "max_smoothness_value": result.max_smoothness_value,
                "smoothness_pass": result.smoothness_pass,
                "linked_trim_id": result.linked_trim_id,
                "match_method": result.match_method,
                "match_confidence": result.match_confidence,
                "tracks": [
                    {
                        "track_id": t.track_id,
                        "smoothness_spec": t.smoothness_spec,
                        "max_smoothness": t.max_smoothness,
                        "smoothness_pass": t.smoothness_pass,
                        "positions": t.position_data or [],
                        "smoothness_values": t.smoothness_data or [],
                    }
                    for t in tracks
                ],
            }

    def get_smoothness_stats(self, days_back: int = 90) -> Dict[str, Any]:
        """Get Output Smoothness dashboard statistics."""
        from laser_trim_analyzer.database.models import SmoothnessResult as DBSmoothnessResult
        with self.session() as session:
            cutoff = datetime.now() - timedelta(days=days_back)
            total = session.query(func.count(DBSmoothnessResult.id)).filter(
                DBSmoothnessResult.file_date >= cutoff
            ).scalar() or 0
            if total == 0:
                return {"total": 0, "pass_rate": 0, "linked_count": 0, "link_rate": 0}
            passed = session.query(func.count(DBSmoothnessResult.id)).filter(
                DBSmoothnessResult.file_date >= cutoff,
                DBSmoothnessResult.smoothness_pass == True,
            ).scalar() or 0
            linked = session.query(func.count(DBSmoothnessResult.id)).filter(
                DBSmoothnessResult.file_date >= cutoff,
                DBSmoothnessResult.linked_trim_id.isnot(None),
            ).scalar() or 0
            return {
                "total": total,
                "pass_rate": round(passed / total * 100, 1),
                "linked_count": linked,
                "link_rate": round(linked / total * 100, 1),
            }

    def get_smoothness_stats_by_model(
        self, model: Optional[str] = None, days_back: int = 90
    ) -> List[Dict[str, Any]]:
        """Get Output Smoothness statistics grouped by model.

        Args:
            model: Optional model filter. If given, return stats for that model only.
            days_back: Number of days to look back (default 90).

        Returns:
            List of dicts sorted by pass_rate ascending (worst first), then margin.
        """
        from laser_trim_analyzer.database.models import SmoothnessResult as DBSmoothnessResult

        with self.session() as session:
            cutoff = datetime.now() - timedelta(days=days_back)

            query = session.query(
                DBSmoothnessResult.model,
                func.count(DBSmoothnessResult.id).label("count"),
                func.sum(
                    case(
                        (DBSmoothnessResult.smoothness_pass == True, 1),
                        else_=0,
                    )
                ).label("passed"),
                func.avg(DBSmoothnessResult.max_smoothness_value).label("avg_max_smoothness"),
                func.max(DBSmoothnessResult.max_smoothness_value).label("worst_case"),
                func.avg(DBSmoothnessResult.smoothness_spec).label("spec_limit"),
            ).filter(
                DBSmoothnessResult.file_date >= cutoff,
            ).group_by(DBSmoothnessResult.model)

            if model is not None:
                query = query.filter(DBSmoothnessResult.model == model)

            rows = query.all()

            results: List[Dict[str, Any]] = []
            for row in rows:
                count = row.count
                passed = row.passed or 0
                pass_rate = round(passed / count * 100, 1) if count else 0.0
                avg_max = round(row.avg_max_smoothness, 4) if row.avg_max_smoothness is not None else 0.0
                worst = round(row.worst_case, 4) if row.worst_case is not None else 0.0
                spec = round(row.spec_limit, 4) if row.spec_limit is not None else 0.0
                margin = round(spec - avg_max, 4)

                results.append({
                    "model": row.model,
                    "count": count,
                    "passed": passed,
                    "pass_rate": pass_rate,
                    "avg_max_smoothness": avg_max,
                    "worst_case": worst,
                    "spec_limit": spec,
                    "margin": margin,
                })

            results.sort(key=lambda r: (r["pass_rate"], r["margin"]))
            return results
