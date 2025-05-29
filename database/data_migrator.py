"""
Data Migrator for Laser Trim AI System

This module handles importing existing data into the database,
including legacy data and batch imports.

Author: Laser Trim AI System
Date: 2024
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
from datetime import datetime

from .database_manager import DatabaseManager
from ..config import Config


class DataMigrator:
    """Handles data migration and imports into the database."""

    def __init__(self, db_manager: DatabaseManager, config: Config):
        """
        Initialize data migrator.

        Args:
            db_manager: Database manager instance
            config: System configuration
        """
        self.db = db_manager
        self.config = config
        self.logger = logging.getLogger(__name__)

    def import_from_excel(self, excel_path: Path, run_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Import analysis results from Excel file.

        Args:
            excel_path: Path to Excel file
            run_id: Optional run ID to associate with imported data

        Returns:
            Dictionary with import statistics
        """
        self.logger.info(f"Importing data from {excel_path}")

        try:
            # Read Excel file
            df = pd.read_excel(excel_path)

            # Create run if not provided
            if run_id is None:
                run_id = self.db.create_analysis_run(
                    input_folder=str(excel_path.parent),
                    configuration={'source': 'excel_import'}
                )

            # Map Excel columns to database fields
            column_mapping = {
                'File': 'filename',
                'Model': 'model',
                'Serial': 'serial',
                'System': 'system',
                'Sigma Gradient': 'sigma_gradient',
                'Sigma Threshold': 'sigma_threshold',
                'Sigma Pass': 'sigma_pass',
                'Linearity Pass': 'linearity_pass',
                'Overall Status': 'overall_status',
                'Failure Probability': 'failure_probability',
                'Risk Category': 'risk_category'
            }

            # Import records
            imported = 0
            failed = 0

            for _, row in df.iterrows():
                try:
                    # Create result dictionary
                    result = {}
                    for excel_col, db_field in column_mapping.items():
                        if excel_col in row:
                            result[db_field] = row[excel_col]

                    # Save to database
                    self.db.save_file_result(run_id, result)
                    imported += 1

                except Exception as e:
                    self.logger.error(f"Failed to import row: {e}")
                    failed += 1

            # Update run statistics
            self.db.update_analysis_run(
                run_id,
                processed_files=imported,
                failed_files=failed,
                total_files=len(df)
            )

            return {
                'success': True,
                'imported': imported,
                'failed': failed,
                'total': len(df),
                'run_id': run_id
            }

        except Exception as e:
            self.logger.error(f"Excel import failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'imported': 0,
                'failed': 0
            }

    def import_from_json(self, json_path: Path, run_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Import analysis results from JSON file.

        Args:
            json_path: Path to JSON file
            run_id: Optional run ID to associate with imported data

        Returns:
            Dictionary with import statistics
        """
        self.logger.info(f"Importing data from {json_path}")

        try:
            # Read JSON file
            with open(json_path, 'r') as f:
                data = json.load(f)

            # Handle both single result and list of results
            if isinstance(data, dict):
                results = [data]
            else:
                results = data

            # Create run if not provided
            if run_id is None:
                run_id = self.db.create_analysis_run(
                    input_folder=str(json_path.parent),
                    configuration={'source': 'json_import'}
                )

            # Import results
            imported = 0
            failed = 0

            for result in results:
                try:
                    self.db.save_file_result(run_id, result)
                    imported += 1
                except Exception as e:
                    self.logger.error(f"Failed to import result: {e}")
                    failed += 1

            # Update run statistics
            self.db.update_analysis_run(
                run_id,
                processed_files=imported,
                failed_files=failed,
                total_files=len(results)
            )

            return {
                'success': True,
                'imported': imported,
                'failed': failed,
                'total': len(results),
                'run_id': run_id
            }

        except Exception as e:
            self.logger.error(f"JSON import failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'imported': 0,
                'failed': 0
            }

    def import_legacy_data(self, legacy_dir: Path) -> Dict[str, Any]:
        """
        Import data from legacy system format.

        Args:
            legacy_dir: Directory containing legacy data files

        Returns:
            Dictionary with import statistics
        """
        self.logger.info(f"Importing legacy data from {legacy_dir}")

        # Create import run
        run_id = self.db.create_analysis_run(
            input_folder=str(legacy_dir),
            configuration={'source': 'legacy_import'}
        )

        stats = {
            'excel_files': 0,
            'json_files': 0,
            'csv_files': 0,
            'total_imported': 0,
            'total_failed': 0
        }

        # Process all files in directory
        for file_path in legacy_dir.iterdir():
            if file_path.suffix == '.xlsx':
                result = self.import_from_excel(file_path, run_id)
                stats['excel_files'] += 1
                stats['total_imported'] += result.get('imported', 0)
                stats['total_failed'] += result.get('failed', 0)

            elif file_path.suffix == '.json':
                result = self.import_from_json(file_path, run_id)
                stats['json_files'] += 1
                stats['total_imported'] += result.get('imported', 0)
                stats['total_failed'] += result.get('failed', 0)

            elif file_path.suffix == '.csv':
                result = self._import_from_csv(file_path, run_id)
                stats['csv_files'] += 1
                stats['total_imported'] += result.get('imported', 0)
                stats['total_failed'] += result.get('failed', 0)

        # Update run with final statistics
        self.db.update_analysis_run(
            run_id,
            processed_files=stats['total_imported'],
            failed_files=stats['total_failed'],
            total_files=stats['total_imported'] + stats['total_failed']
        )

        return stats

    def _import_from_csv(self, csv_path: Path, run_id: int) -> Dict[str, Any]:
        """Import from CSV file."""
        try:
            df = pd.read_csv(csv_path)

            # Similar to Excel import but for CSV
            imported = 0
            failed = 0

            for _, row in df.iterrows():
                try:
                    result = {
                        'filename': row.get('filename', csv_path.name),
                        'model': row.get('model', 'Unknown'),
                        'serial': row.get('serial', 'Unknown'),
                        'sigma_gradient': float(row.get('sigma_gradient', 0)),
                        'sigma_pass': bool(row.get('sigma_pass', False)),
                        'overall_status': row.get('status', 'Unknown')
                    }

                    self.db.save_file_result(run_id, result)
                    imported += 1

                except Exception as e:
                    self.logger.error(f"Failed to import CSV row: {e}")
                    failed += 1

            return {
                'success': True,
                'imported': imported,
                'failed': failed
            }

        except Exception as e:
            self.logger.error(f"CSV import failed: {e}")
            return {
                'success': False,
                'imported': 0,
                'failed': 0,
                'error': str(e)
            }

    def export_for_backup(self, output_dir: Path, days_back: Optional[int] = None) -> Path:
        """
        Export database data for backup purposes.

        Args:
            output_dir: Directory to save backup
            days_back: Optional limit on how many days to export

        Returns:
            Path to backup file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Export to both Excel and JSON for redundancy
        excel_path = output_dir / f'backup_{timestamp}.xlsx'
        json_path = output_dir / f'backup_{timestamp}.json'

        # Export to Excel
        self.db.export_to_excel(excel_path, days_back or 365)

        # Export to JSON
        df = self.db.get_historical_data(days_back=days_back)

        # Convert to records and handle datetime serialization
        records = df.to_dict('records')
        for record in records:
            for key, value in record.items():
                if isinstance(value, pd.Timestamp):
                    record[key] = value.isoformat()
                elif pd.isna(value):
                    record[key] = None

        with open(json_path, 'w') as f:
            json.dump(records, f, indent=2)

        self.logger.info(f"Backup exported to {output_dir}")

        return excel_path

    def validate_import(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate data before import.

        Args:
            data: List of records to validate

        Returns:
            List of validation errors
        """
        errors = []

        required_fields = ['filename', 'model', 'sigma_gradient']

        for i, record in enumerate(data):
            record_errors = []

            # Check required fields
            for field in required_fields:
                if field not in record or record[field] is None:
                    record_errors.append(f"Missing required field: {field}")

            # Validate data types
            if 'sigma_gradient' in record:
                try:
                    float(record['sigma_gradient'])
                except (TypeError, ValueError):
                    record_errors.append("sigma_gradient must be numeric")

            # Validate boolean fields
            for bool_field in ['sigma_pass', 'linearity_pass']:
                if bool_field in record and record[bool_field] is not None:
                    if not isinstance(record[bool_field], bool):
                        record_errors.append(f"{bool_field} must be boolean")

            if record_errors:
                errors.append({
                    'record_index': i,
                    'errors': record_errors,
                    'record': record
                })

        return errors