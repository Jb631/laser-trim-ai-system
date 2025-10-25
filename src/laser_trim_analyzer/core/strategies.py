# laser_trim_analyzer/core/strategies.py
"""
Strategy implementations for different system types.
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Any
import logging

from .interfaces import (
    TrackResult, TrimData, SystemType, Status,
    DataExtractor, MetricsCalculator, RiskCategory,
    SigmaMetrics, LinearityMetrics
)


class SystemStrategy(ABC):
    """Abstract base class for system-specific processing strategies."""

    def __init__(
            self,
            data_extractor: DataExtractor,
            metrics_calculator: MetricsCalculator,
            logger: Optional[logging.Logger] = None
    ):
        self.data_extractor = data_extractor
        self.metrics_calculator = metrics_calculator
        self.logger = logger or logging.getLogger(__name__)

    @abstractmethod
    async def process_file(
            self,
            filepath: str,
            file_info: Dict[str, Any],
            model: str,
            generate_plots: bool = True
    ) -> Dict[str, TrackResult]:
        """Process file according to system-specific logic."""
        pass

    async def process_track(
            self,
            filepath: str,
            track_id: str,
            sheet_info: Dict[str, Any],
            model: str,
            system: SystemType,
            generate_plots: bool = True
    ) -> Optional[TrackResult]:
        """Process a single track with common logic."""
        try:
            # Extract untrimmed data
            untrimmed_sheet = sheet_info.get("untrimmed")
            if not untrimmed_sheet:
                self.logger.error(f"No untrimmed sheet for track {track_id}")
                return None

            untrimmed_data = await self.data_extractor.extract_trim_data(
                filepath, untrimmed_sheet, system
            )

            if not untrimmed_data.is_valid:
                self.logger.error(f"Invalid untrimmed data for track {track_id}")
                return None

            # Extract unit properties
            unit_properties = await self.data_extractor.extract_unit_properties(
                filepath, sheet_info, system
            )

            # Calculate sigma metrics
            sigma_metrics = self.metrics_calculator.calculate_sigma_metrics(
                untrimmed_data, unit_properties, model
            )

            # Extract and process final data if available
            final_data = None
            linearity_metrics = None
            trim_effectiveness = None

            final_sheet = sheet_info.get("final")
            if final_sheet:
                final_data = await self.data_extractor.extract_trim_data(
                    filepath, final_sheet, system
                )

                if final_data.is_valid:
                    # Calculate linearity metrics
                    linearity_metrics = self.metrics_calculator.calculate_linearity_metrics(
                        final_data, sigma_metrics.threshold
                    )

                    # Calculate trim effectiveness
                    trim_effectiveness = self.metrics_calculator.calculate_trim_effectiveness(
                        untrimmed_data, final_data
                    )

            # Determine status
            status = self._determine_track_status(sigma_metrics, linearity_metrics)

            # Calculate failure probability
            failure_prob = self.metrics_calculator.calculate_failure_probability(
                sigma_metrics, linearity_metrics, unit_properties
            )

            # Determine risk category
            risk_category = self._determine_risk_category(failure_prob)

            # Create track result
            return TrackResult(
                track_id=track_id,
                status=status,
                untrimmed_data=untrimmed_data,
                final_data=final_data,
                unit_properties=unit_properties,
                sigma_metrics=sigma_metrics,
                linearity_metrics=linearity_metrics,
                trim_effectiveness=trim_effectiveness,
                # Zone analysis is computed in the main processors; not used here
                zone_analysis=None,
                failure_probability=failure_prob,
                risk_category=risk_category
            )

        except Exception as e:
            self.logger.error(f"Error processing track {track_id}: {e}", exc_info=True)
            return None

    def _determine_track_status(
            self,
            sigma_metrics: SigmaMetrics,
            linearity_metrics: Optional[LinearityMetrics]
    ) -> Status:
        """Determine track status from metrics."""
        if not sigma_metrics.passed:
            return Status.FAIL

        if linearity_metrics and not linearity_metrics.passed:
            return Status.FAIL

        if not linearity_metrics:
            return Status.WARNING

        return Status.PASS

    def _determine_risk_category(self, failure_probability: float) -> RiskCategory:
        """Determine risk category from failure probability."""
        if failure_probability > 0.7:
            return RiskCategory.HIGH
        elif failure_probability > 0.3:
            return RiskCategory.MEDIUM
        else:
            return RiskCategory.LOW


class SystemAStrategy(SystemStrategy):
    """Strategy for processing System A files (multi-track support)."""

    async def process_file(
            self,
            filepath: str,
            file_info: Dict[str, Any],
            model: str,
            generate_plots: bool = True
    ) -> Dict[str, TrackResult]:
        """Process System A file with multi-track support."""
        tracks = {}
        track_info = file_info.get("tracks", {})

        if not track_info:
            # Fallback to single track processing
            sheet_info = {
                "untrimmed": file_info.get("untrimmed"),
                "final": file_info.get("final"),
                "all_trims": file_info.get("all_trims", [])
            }

            result = await self.process_track(
                filepath, "default", sheet_info, model,
                SystemType.SYSTEM_A, generate_plots
            )

            if result:
                tracks["default"] = result
        else:
            # Process each track
            for track_id, sheet_info in track_info.items():
                self.logger.info(f"Processing track {track_id}")

                result = await self.process_track(
                    filepath, track_id, sheet_info, model,
                    SystemType.SYSTEM_A, generate_plots
                )

                if result:
                    tracks[track_id] = result

        return tracks


class SystemBStrategy(SystemStrategy):
    """Strategy for processing System B files."""

    async def process_file(
            self,
            filepath: str,
            file_info: Dict[str, Any],
            model: str,
            generate_plots: bool = True
    ) -> Dict[str, TrackResult]:
        """Process System B file (single track)."""
        sheet_info = {
            "untrimmed": file_info.get("untrimmed_sheet", "test"),
            "final": file_info.get("final_sheet", "Lin Error"),
            "all_trims": file_info.get("all_trims", [])
        }

        result = await self.process_track(
            filepath, "default", sheet_info, model,
            SystemType.SYSTEM_B, generate_plots
        )

        return {"default": result} if result else {}
