"""
Test Suite for Core Data Processing Engine

This module provides comprehensive tests for the data processor,
including validation of sigma calculations and data extraction.

Author: QA Team
Date: 2024
Version: 1.0.0
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import os
from typing import Dict, Any

from data_processor import (
    DataProcessor, SystemType, DataExtraction,
    UnitProperties, SigmaResults
)


class TestDataProcessor(unittest.TestCase):
    """Test cases for the DataProcessor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.processor = DataProcessor()
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temporary directory
        import shutil
        shutil.rmtree(self.test_dir)

    def test_system_detection(self):
        """Test system type detection."""
        # Create test files with different naming patterns
        test_cases = [
            ("8340_test.xlsx", SystemType.SYSTEM_B),
            ("6845_sample.xlsx", SystemType.SYSTEM_A),
            ("7890_data.xlsx", SystemType.SYSTEM_A),
            ("8555_measurement.xlsx", SystemType.SYSTEM_A),
        ]

        for filename, expected_system in test_cases:
            # Create dummy Excel file
            file_path = Path(self.test_dir) / filename
            self._create_dummy_excel(file_path)

            # Test detection
            detected = self.processor._detect_system(file_path)
            self.assertEqual(detected, expected_system,
                             f"Failed to detect {expected_system} for {filename}")

    def test_sigma_calculation(self):
        """Test sigma gradient calculation accuracy."""
        # Create test data
        position = np.linspace(0, 120, 100)
        # Add some noise to create error signal
        error = 0.001 * np.sin(position * 0.1) + 0.0001 * np.random.randn(100)

        # Create unit properties
        unit_props = UnitProperties(
            unit_length=120.0,
            travel_length=120.0,
            linearity_spec=0.01
        )

        # Calculate sigma
        result = self.processor._calculate_sigma_gradient(position, error, unit_props)

        # Verify result structure
        self.assertIsInstance(result, SigmaResults)
        self.assertIsInstance(result.sigma_gradient, float)
        self.assertIsInstance(result.sigma_threshold, float)
        self.assertIsInstance(result.sigma_pass, bool)

        # Verify calculation is within expected range
        self.assertGreater(result.sigma_gradient, 0)
        self.assertLess(result.sigma_gradient, 1.0)  # Should be small for smooth data

        # Verify threshold calculation
        expected_threshold = (unit_props.linearity_spec / unit_props.unit_length) * 24.0
        self.assertAlmostEqual(result.sigma_threshold, expected_threshold, places=6)

    def test_filter_implementation(self):
        """Test the filter matches MATLAB implementation."""
        # Create test signal
        test_signal = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])

        # Apply filter
        filtered = self.processor._apply_filter(test_signal)

        # Verify filter properties
        self.assertEqual(len(filtered), len(test_signal))
        # First and last values should be preserved
        self.assertAlmostEqual(filtered[0], test_signal[0], places=6)
        # Filter should smooth the signal
        self.assertLess(np.var(filtered), np.var(test_signal))

    def test_data_extraction_system_a(self):
        """Test data extraction for System A files."""
        # Create test Excel file with System A format
        file_path = Path(self.test_dir) / "system_a_test.xlsx"
        self._create_system_a_test_file(file_path)

        # Process file
        excel_file = pd.ExcelFile(file_path)
        data = self.processor._extract_system_a_data(
            excel_file, "SEC1 TRK1 0", "TRK1"
        )

        # Verify extraction
        self.assertIsInstance(data, DataExtraction)
        self.assertGreater(len(data.position), 0)
        self.assertGreater(len(data.error), 0)
        self.assertEqual(len(data.position), len(data.error))

        # Verify data is sorted by position
        self.assertTrue(np.all(np.diff(data.position) >= 0))

    def test_data_validation_and_cleaning(self):
        """Test data validation and cleaning functionality."""
        # Create data with NaN values and unsorted positions
        data = DataExtraction(
            position=np.array([3, 1, 2, np.nan, 4, 5]),
            error=np.array([0.1, 0.2, np.nan, 0.3, 0.4, 0.5]),
            sheet_name="test"
        )

        # Clean data
        cleaned = self.processor._validate_and_clean_data(data)

        # Verify NaN values removed
        self.assertFalse(np.any(np.isnan(cleaned.position)))
        self.assertFalse(np.any(np.isnan(cleaned.error)))

        # Verify data is sorted
        self.assertTrue(np.all(np.diff(cleaned.position) >= 0))

        # Verify same length
        self.assertEqual(len(cleaned.position), len(cleaned.error))

    def test_unit_properties_extraction(self):
        """Test extraction of unit properties from specific cells."""
        # Create test file with unit properties
        file_path = Path(self.test_dir) / "unit_props_test.xlsx"
        self._create_file_with_unit_properties(file_path)

        # Extract properties
        excel_file = pd.ExcelFile(file_path)
        props = self.processor._extract_unit_properties_system_a(
            excel_file, "Sheet1"
        )

        # Verify extraction
        self.assertIsNotNone(props.unit_length)
        self.assertIsNotNone(props.untrimmed_resistance)
        self.assertAlmostEqual(props.unit_length, 150.0, places=1)
        self.assertAlmostEqual(props.untrimmed_resistance, 1000.0, places=1)

    def test_multi_track_support(self):
        """Test processing of multi-track System A files."""
        # Create multi-track test file
        file_path = Path(self.test_dir) / "multi_track_test.xlsx"
        self._create_multi_track_file(file_path)

        # Process file
        results = self.processor.process_file(file_path)

        # Verify both tracks processed
        self.assertIn('tracks', results)
        self.assertIn('TRK1', results['tracks'])
        self.assertIn('TRK2', results['tracks'])

        # Verify each track has complete results
        for track_id in ['TRK1', 'TRK2']:
            track_data = results['tracks'][track_id]
            self.assertIn('untrimmed_data', track_data)
            self.assertIn('unit_properties', track_data)
            self.assertIn('sigma_results', track_data)

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test non-existent file
        with self.assertRaises(ValueError):
            self.processor.process_file("non_existent_file.xlsx")

        # Test invalid file format
        invalid_file = Path(self.test_dir) / "test.txt"
        invalid_file.write_text("not an excel file")
        with self.assertRaises(ValueError):
            self.processor.process_file(invalid_file)

    def test_batch_processing(self):
        """Test batch processing of multiple files."""
        # Create multiple test files
        for i in range(3):
            file_path = Path(self.test_dir) / f"test_file_{i}.xlsx"
            self._create_dummy_excel(file_path)

        # Process batch
        results = self.processor.batch_process(self.test_dir)

        # Verify all files processed
        self.assertEqual(len(results), 3)
        for i in range(3):
            self.assertIn(f"test_file_{i}.xlsx", results)

    # Helper methods for creating test files

    def _create_dummy_excel(self, file_path: Path):
        """Create a dummy Excel file for testing."""
        df = pd.DataFrame({
            'A': range(10),
            'B': range(10)
        })
        df.to_excel(file_path, index=False)

    def _create_system_a_test_file(self, file_path: Path):
        """Create a test file with System A format."""
        # Create test data
        n_points = 50
        position = np.linspace(0, 120, n_points)
        error = 0.001 * np.sin(position * 0.1)

        # Create DataFrame with System A column layout
        data = pd.DataFrame({
            'A': [''] * n_points,  # Empty columns
            'B': [''] * n_points,
            'C': [''] * n_points,
            'D': np.random.randn(n_points),  # Measured volts
            'E': range(n_points),  # Index
            'F': np.random.randn(n_points),  # Theory volts
            'G': error,  # Error
            'H': position,  # Position
            'I': error + 0.01,  # Upper limit
            'J': error - 0.01  # Lower limit
        })

        # Write to Excel with sheet names
        with pd.ExcelWriter(file_path) as writer:
            data.to_excel(writer, sheet_name='SEC1 TRK1 0', index=False, header=False)
            data.to_excel(writer, sheet_name='SEC1 TRK1 TRM', index=False, header=False)

    def _create_file_with_unit_properties(self, file_path: Path):
        """Create a test file with unit properties in specific cells."""
        # Create empty DataFrame
        df = pd.DataFrame(index=range(30), columns=list('ABCDEFGHIJ'))

        # Set unit properties in specific cells
        df.loc[25, 'B'] = 150.0  # Unit length in B26 (row 26, 0-based is 25)
        df.loc[9, 'B'] = 1000.0  # Resistance in B10 (row 10, 0-based is 9)

        df.to_excel(file_path, sheet_name='Sheet1', index=False, header=False)

    def _create_multi_track_file(self, file_path: Path):
        """Create a test file with multiple tracks."""
        # Create test data for two tracks
        n_points = 50
        position = np.linspace(0, 120, n_points)

        # Track 1 data
        error1 = 0.001 * np.sin(position * 0.1)
        data1 = pd.DataFrame({
            'A': [''] * n_points,
            'B': [''] * n_points,
            'C': [''] * n_points,
            'D': np.random.randn(n_points),
            'E': range(n_points),
            'F': np.random.randn(n_points),
            'G': error1,
            'H': position,
            'I': error1 + 0.01,
            'J': error1 - 0.01
        })

        # Track 2 data (slightly different)
        error2 = 0.001 * np.cos(position * 0.1)
        data2 = pd.DataFrame({
            'A': [''] * n_points,
            'B': [''] * n_points,
            'C': [''] * n_points,
            'D': np.random.randn(n_points),
            'E': range(n_points),
            'F': np.random.randn(n_points),
            'G': error2,
            'H': position,
            'I': error2 + 0.01,
            'J': error2 - 0.01
        })

        # Write to Excel with multiple sheets
        with pd.ExcelWriter(file_path) as writer:
            data1.to_excel(writer, sheet_name='SEC1 TRK1 0', index=False, header=False)
            data1.to_excel(writer, sheet_name='SEC1 TRK1 TRM', index=False, header=False)
            data2.to_excel(writer, sheet_name='SEC1 TRK2 0', index=False, header=False)
            data2.to_excel(writer, sheet_name='SEC1 TRK2 TRM', index=False, header=False)


class TestSigmaCalculationAccuracy(unittest.TestCase):
    """Specific tests for sigma calculation accuracy."""

    def setUp(self):
        """Set up test fixtures."""
        self.processor = DataProcessor()

    def test_known_sigma_values(self):
        """Test sigma calculation with known expected values."""
        # Test case 1: Constant error (should have near-zero sigma)
        position = np.linspace(0, 100, 100)
        error = np.ones_like(position) * 0.01

        unit_props = UnitProperties(
            unit_length=100.0,
            linearity_spec=0.01
        )

        result = self.processor._calculate_sigma_gradient(position, error, unit_props)
        self.assertLess(result.sigma_gradient, 0.0001)  # Should be very small

        # Test case 2: Linear error (should have constant gradient)
        error = 0.001 * position
        result = self.processor._calculate_sigma_gradient(position, error, unit_props)
        # The gradient should be approximately 0.001
        mean_gradient = np.mean(result.gradients)
        self.assertAlmostEqual(mean_gradient, 0.001, places=4)

    def test_gradient_step_size(self):
        """Test that gradient calculation uses correct step size."""
        position = np.linspace(0, 100, 20)
        error = 0.001 * position

        unit_props = UnitProperties(unit_length=100.0)
        result = self.processor._calculate_sigma_gradient(position, error, unit_props)

        # Verify number of gradients is correct
        expected_gradient_count = len(position) - DataProcessor.GRADIENT_STEP
        self.assertEqual(len(result.gradients), expected_gradient_count)


if __name__ == '__main__':
    unittest.main()