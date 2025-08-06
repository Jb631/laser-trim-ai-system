"""
Integration tests for the main processor.

Tests the complete processing pipeline including:
- File processing
- System detection
- Data extraction
- Analysis coordination
- ML integration
- Database storage
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

from laser_trim_analyzer.core.config import Config
from laser_trim_analyzer.core.models import (
    AnalysisStatus, SystemType, RiskCategory
)
from laser_trim_analyzer.core.processor import LaserTrimProcessor
from laser_trim_analyzer.database.manager import DatabaseManager
from laser_trim_analyzer.ml.predictors import MLPredictor


class TestLaserTrimProcessor:
    """Test suite for the main processor."""

    @pytest.fixture
    def test_config(self):
        """Create test configuration."""
        config = Config(
            debug=True,
            processing=Config.ProcessingConfig(
                max_workers=2,
                generate_plots=True,
                cache_enabled=True
            ),
            analysis=Config.AnalysisConfig(
                sigma_scaling_factor=24.0,
                high_risk_threshold=0.7,
                low_risk_threshold=0.3
            ),
            database=Config.DatabaseConfig(
                enabled=True,
                path=Path(tempfile.gettempdir()) / "test_laser_trim.db"
            )
        )
        return config

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_excel_file_system_a(self, temp_dir):
        """Create sample System A Excel file."""
        file_path = temp_dir / "8340_A12345_20240115.xlsx"

        # Create sample data
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            # TRK1 untrimmed sheet
            df_trk1_0 = pd.DataFrame({
                'A': [''] * 100,
                'B': [''] * 100,
                'C': [''] * 100,
                'D': [''] * 100,
                'E': [''] * 100,
                'F': [''] * 100,
                'G': np.random.normal(0, 0.01, 100),  # Error
                'H': np.linspace(0, 100, 100),  # Position
                'I': [0.05] * 100,  # Upper limit
                'J': [-0.05] * 100,  # Lower limit
            })
            df_trk1_0.to_excel(writer, sheet_name='SEC1 TRK1 0', index=False)

            # Add unit properties
            ws = writer.sheets['SEC1 TRK1 0']
            ws['B10'] = 10000  # Untrimmed resistance
            ws['B26'] = 300  # Unit length

            # TRK1 trimmed sheet
            df_trk1_trm = pd.DataFrame({
                'A': [''] * 100,
                'B': [''] * 100,
                'C': [''] * 100,
                'D': [''] * 100,
                'E': [''] * 100,
                'F': [''] * 100,
                'G': np.random.normal(0, 0.005, 100),  # Error (improved)
                'H': np.linspace(0, 100, 100),  # Position
                'I': [0.05] * 100,  # Upper limit
                'J': [-0.05] * 100,  # Lower limit
            })
            df_trk1_trm.to_excel(writer, sheet_name='SEC1 TRK1 TRM', index=False)

            # Add trimmed resistance
            ws_trm = writer.sheets['SEC1 TRK1 TRM']
            ws_trm['B10'] = 10230  # Trimmed resistance

        return file_path

    @pytest.fixture
    def sample_excel_file_system_b(self, temp_dir):
        """Create sample System B Excel file."""
        file_path = temp_dir / "8555_B67890_20240115.xlsx"

        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            # Test sheet (untrimmed)
            df_test = pd.DataFrame({
                'A': [''] * 100,
                'B': [''] * 100,
                'C': [''] * 100,
                'D': np.random.normal(0, 0.015, 100),  # Error
                'E': [''] * 100,
                'F': [0.08] * 100,  # Upper limit
                'G': [-0.08] * 100,  # Lower limit
                'H': [''] * 100,
                'I': np.linspace(0, 150, 100),  # Position
            })
            df_test.to_excel(writer, sheet_name='test', index=False)

            # Add properties
            ws = writer.sheets['test']
            ws['K1'] = 360  # Unit length
            ws['R1'] = 5000  # Untrimmed resistance

            # Lin Error sheet (trimmed)
            df_lin = pd.DataFrame({
                'A': [''] * 100,
                'B': [''] * 100,
                'C': [''] * 100,
                'D': np.random.normal(0, 0.008, 100),  # Error (improved)
                'E': [''] * 100,
                'F': [0.08] * 100,  # Upper limit
                'G': [-0.08] * 100,  # Lower limit
                'H': [''] * 100,
                'I': np.linspace(0, 150, 100),  # Position
            })
            df_lin.to_excel(writer, sheet_name='Lin Error', index=False)

            # Add trimmed resistance
            ws_lin = writer.sheets['Lin Error']
            ws_lin['R1'] = 5150  # Trimmed resistance

        return file_path

    @pytest.fixture
    async def processor(self, test_config):
        """Create processor instance."""
        # Initialize database
        db_manager = DatabaseManager(str(test_config.database.path))
        db_manager.init_db()

        # Create processor
        processor = LaserTrimProcessor(
            config=test_config,
            db_manager=db_manager
        )

        yield processor

        # Cleanup
        if test_config.database.path.exists():
            test_config.database.path.unlink()

    @pytest.mark.asyncio
    async def test_process_system_a_file(self, processor, sample_excel_file_system_a, temp_dir):
        """Test processing System A file."""
        output_dir = temp_dir / "output"
        output_dir.mkdir()

        # Process file
        result = await processor.process_file(
            sample_excel_file_system_a,
            output_dir
        )

        # Verify results
        assert result is not None
        assert result.metadata.system == SystemType.SYSTEM_A
        assert result.metadata.model == "8340"
        assert result.metadata.serial == "A12345"

        # Check tracks
        assert "TRK1" in result.tracks or "default" in result.tracks

        # Check primary track
        track = result.primary_track
        assert track is not None
        assert track.unit_properties.unit_length == 300
        assert track.unit_properties.untrimmed_resistance == 10000
        assert track.unit_properties.trimmed_resistance == 10230

        # Check analyses
        assert track.sigma_analysis is not None
        assert track.linearity_analysis is not None
        assert track.resistance_analysis is not None

        # Check sigma analysis
        assert track.sigma_analysis.sigma_gradient > 0
        assert track.sigma_analysis.sigma_threshold > 0
        assert isinstance(track.sigma_analysis.sigma_pass, bool)

        # Check database save
        assert result.db_id is not None

    @pytest.mark.asyncio
    async def test_process_system_b_file(self, processor, sample_excel_file_system_b, temp_dir):
        """Test processing System B file."""
        output_dir = temp_dir / "output"
        output_dir.mkdir()

        # Process file
        result = await processor.process_file(
            sample_excel_file_system_b,
            output_dir
        )

        # Verify results
        assert result is not None
        assert result.metadata.system == SystemType.SYSTEM_B
        assert result.metadata.model == "8555"
        assert result.metadata.serial == "B67890"

        # Check track
        assert "default" in result.tracks
        track = result.tracks["default"]

        # Check properties
        assert track.unit_properties.unit_length == 360
        assert track.unit_properties.untrimmed_resistance == 5000
        assert track.unit_properties.trimmed_resistance == 5150

        # Check trim effectiveness
        assert track.trim_effectiveness is not None
        assert track.trim_effectiveness.improvement_percent > 0

    @pytest.mark.asyncio
    async def test_batch_processing(self, processor, temp_dir, sample_excel_file_system_a, sample_excel_file_system_b):
        """Test batch processing multiple files."""
        output_dir = temp_dir / "batch_output"

        # Track progress
        progress_calls = []

        def progress_callback(current, total, filename):
            progress_calls.append((current, total, filename))

        # Process batch
        results = await processor.process_batch(
            temp_dir,
            output_dir,
            progress_callback=progress_callback
        )

        # Verify results
        assert len(results) == 2
        assert len(progress_calls) > 0

        # Check batch report
        batch_dirs = list(output_dir.iterdir())
        assert len(batch_dirs) == 1

        batch_report = batch_dirs[0] / "batch_summary.xlsx"
        assert batch_report.exists()

    @pytest.mark.asyncio
    async def test_ml_integration(self, processor, sample_excel_file_system_a, temp_dir):
        """Test ML predictor integration."""

        # Create mock ML predictor
        class MockMLPredictor:
            async def predict(self, data):
                return {
                    'failure_probability': 0.25,
                    'risk_assessment': {
                        'TRK1': {'risk_category': 'LOW'}
                    }
                }

        processor.ml_predictor = MockMLPredictor()

        output_dir = temp_dir / "output"
        output_dir.mkdir()

        # Process with ML
        result = await processor.process_file(
            sample_excel_file_system_a,
            output_dir
        )

        # Check ML predictions were added
        assert hasattr(result, 'ml_predictions')
        assert result.ml_predictions['failure_probability'] == 0.25

    @pytest.mark.asyncio
    async def test_error_handling(self, processor, temp_dir):
        """Test error handling for invalid files."""
        # Non-existent file
        with pytest.raises(FileNotFoundError):
            await processor.process_file(
                temp_dir / "non_existent.xlsx",
                temp_dir
            )

        # Invalid file type
        invalid_file = temp_dir / "invalid.txt"
        invalid_file.write_text("Not an Excel file")

        with pytest.raises(Exception):
            await processor.process_file(invalid_file, temp_dir)

    @pytest.mark.asyncio
    async def test_caching(self, processor, sample_excel_file_system_a, temp_dir):
        """Test result caching."""
        output_dir = temp_dir / "output"
        output_dir.mkdir()

        # Process file first time
        result1 = await processor.process_file(
            sample_excel_file_system_a,
            output_dir
        )

        # Process same file again (should use cache)
        result2 = await processor.process_file(
            sample_excel_file_system_a,
            output_dir
        )

        # Results should be the same object (cached)
        assert result1 is result2

    @pytest.mark.asyncio
    async def test_progress_callback(self, processor, sample_excel_file_system_a, temp_dir):
        """Test progress callback functionality."""
        output_dir = temp_dir / "output"
        output_dir.mkdir()

        progress_updates = []

        def progress_callback(message, progress):
            progress_updates.append((message, progress))

        # Process with progress callback
        await processor.process_file(
            sample_excel_file_system_a,
            output_dir,
            progress_callback=progress_callback
        )

        # Verify progress updates
        assert len(progress_updates) > 0
        assert progress_updates[0][1] < progress_updates[-1][1]  # Progress increases
        assert progress_updates[-1][1] == 1.0  # Ends at 100%