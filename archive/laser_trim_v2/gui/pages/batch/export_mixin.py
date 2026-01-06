"""
ExportMixin - Export functionality for BatchProcessingPage.

This module provides batch export methods:
- _export_batch_results: Main export dispatcher
- _export_batch_excel: Excel export with comprehensive report
- _export_batch_excel_legacy: Legacy Excel export fallback

Note (v3 Redesign): CSV and HTML exports are deprecated but retained for
backwards compatibility. The UI now only shows Excel export for simplicity.

Extracted from batch_processing_page.py during Phase 4 file splitting.
"""

import logging
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox

logger = logging.getLogger(__name__)


class ExportMixin:
    """
    Mixin providing batch export functionality.

    Requires BatchProcessingPage as parent class with:
    - self.batch_results: Dict of analysis results
    - self.selected_files: List of selected file paths
    - self.report_generator: ReportGenerator instance
    - self.include_raw_data_var: Checkbox variable
    - self._safe_after: Thread-safe after method
    """

    def _export_batch_results(self, format_type: str = 'excel'):
        """Export batch processing results in specified format."""
        if not self.batch_results:
            messagebox.showerror("Error", "No results to export")
            return

        # Log the number of results to help debug
        logger.info(f"Exporting {len(self.batch_results)} batch results as {format_type}")

        # Set file extension and types based on format
        if format_type == 'excel':
            default_ext = ".xlsx"
            file_types = [("Excel files", "*.xlsx"), ("All files", "*.*")]
        elif format_type == 'html':
            default_ext = ".html"
            file_types = [("HTML files", "*.html"), ("All files", "*.*")]
        elif format_type == 'csv':
            default_ext = ".csv"
            file_types = [("CSV files", "*.csv"), ("All files", "*.*")]
        else:
            messagebox.showerror("Error", f"Unsupported format: {format_type}")
            return

        # Ask for export location
        file_path = filedialog.asksaveasfilename(
            title=f"Export Batch Results as {format_type.upper()}",
            defaultextension=default_ext,
            filetypes=file_types
        )

        if file_path:
            try:
                path_obj = Path(file_path)

                if format_type == 'excel':
                    self._export_batch_excel(path_obj)
                elif format_type == 'html':
                    self._export_batch_html(path_obj)
                elif format_type == 'csv':
                    self._export_batch_csv(path_obj)

                messagebox.showinfo("Export Complete", f"Batch results exported to:\n{file_path}")
                logger.info(f"Batch results exported to: {file_path} (format: {format_type})")

                # Also generate a summary JSON report alongside
                if format_type in ['excel', 'html']:
                    summary_path = path_obj.with_suffix('.summary.json')
                    try:
                        results_list = list(self.batch_results.values())
                        self.report_generator.generate_summary_report(results_list, summary_path)
                        logger.info(f"Summary report generated: {summary_path}")
                    except Exception as e:
                        logger.warning(f"Failed to generate summary report: {e}")

            except Exception as e:
                logger.error(f"Batch export failed: {e}")
                error_msg = f"Failed to export batch results:\n\n{str(e)}"
                if "Permission denied" in str(e):
                    error_msg += "\n\nPlease ensure the output file is not open in another program."
                elif "No space left" in str(e):
                    error_msg += "\n\nInsufficient disk space. Please free up some space and try again."
                elif "Invalid file path" in str(e):
                    error_msg += "\n\nThe selected file path is invalid. Please choose a different location."
                messagebox.showerror("Export Failed", error_msg)

    def _export_batch_excel(self, file_path: Path):
        """Export batch results to Excel format using comprehensive report generator."""
        try:
            # Convert batch_results dict to list of AnalysisResult objects
            results_list = list(self.batch_results.values())

            # Validate we have actual results
            if not results_list:
                raise ValueError("No results found in batch_results")

            logger.info(f"Exporting {len(results_list)} results to Excel")

            # Get the include raw data option
            include_raw_data = self.include_raw_data_var.get()

            # Use the enhanced Excel exporter for comprehensive data export
            try:
                from laser_trim_analyzer.utils.enhanced_excel_export import EnhancedExcelExporter
                enhanced_exporter = EnhancedExcelExporter()

                # Use batch export method
                enhanced_exporter.export_batch_comprehensive(
                    results=results_list,
                    output_path=file_path,
                    include_individual_details=include_raw_data,
                    max_individual_sheets=10 if include_raw_data else 0
                )

                logger.info("Used enhanced Excel exporter for comprehensive data export")
            except ImportError:
                # Fallback to standard report generator
                logger.warning("Enhanced Excel exporter not available, using standard export")
                self.report_generator.generate_comprehensive_excel_report(
                    results=results_list,
                    output_path=file_path,
                    include_raw_data=include_raw_data
                )

            logger.info(f"Batch results exported using comprehensive report generator to: {file_path} (raw data: {include_raw_data})")

        except Exception as e:
            # If comprehensive report fails, fall back to legacy export method
            logger.warning(f"Comprehensive report generation failed, using legacy export: {e}")
            self._export_batch_excel_legacy(file_path)

    def _export_batch_html(self, file_path: Path):
        """Export batch results to HTML format."""
        try:
            # Convert batch_results dict to list of AnalysisResult objects
            results_list = list(self.batch_results.values())

            # Generate HTML report
            self.report_generator.generate_html_report(
                results=results_list,
                output_path=file_path,
                title=f"Batch Analysis Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            )

            logger.info(f"Batch results exported to HTML: {file_path}")

        except Exception as e:
            logger.error(f"HTML export failed: {e}")
            raise

    def _export_batch_excel_legacy(self, file_path: Path):
        """Legacy export method for batch results to Excel format."""
        import pandas as pd

        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = []
            all_track_data = []

            for file_path_str, result in self.batch_results.items():
                file_name = Path(file_path_str).name

                try:
                    # Get values safely with error handling
                    model = getattr(result.metadata, 'model', 'Unknown')
                    serial = getattr(result.metadata, 'serial', 'Unknown')

                    # Handle system safely (attribute is 'system', not 'system_type')
                    system_type = 'Unknown'
                    if hasattr(result.metadata, 'system'):
                        system_type = getattr(result.metadata.system, 'value', str(result.metadata.system))
                    elif hasattr(result.metadata, 'system_type'):
                        # Fallback for legacy attribute name
                        system_type = getattr(result.metadata.system_type, 'value', str(result.metadata.system_type))

                    # Handle trim/test date (prefer test_date, fallback to file_date)
                    trim_date = 'Unknown'
                    if hasattr(result.metadata, 'test_date') and result.metadata.test_date:
                        if hasattr(result.metadata.test_date, 'strftime'):
                            trim_date = result.metadata.test_date.strftime('%Y-%m-%d %H:%M:%S')
                        else:
                            trim_date = str(result.metadata.test_date)
                    elif hasattr(result.metadata, 'file_date') and result.metadata.file_date:
                        if hasattr(result.metadata.file_date, 'strftime'):
                            trim_date = result.metadata.file_date.strftime('%Y-%m-%d %H:%M:%S')
                        else:
                            trim_date = str(result.metadata.file_date)
                    elif hasattr(result.metadata, 'analysis_date'):
                        # Legacy fallback
                        if hasattr(result.metadata.analysis_date, 'strftime'):
                            trim_date = result.metadata.analysis_date.strftime('%Y-%m-%d %H:%M:%S')
                        else:
                            trim_date = str(result.metadata.analysis_date)

                    # Handle overall_status safely
                    overall_status = 'Unknown'
                    if hasattr(result, 'overall_status'):
                        overall_status = getattr(result.overall_status, 'value', str(result.overall_status))

                    # Handle validation_status safely
                    validation_status = 'N/A'
                    if hasattr(result, 'overall_validation_status'):
                        validation_status = getattr(result.overall_validation_status, 'value',
                                                   str(result.overall_validation_status))

                    # Get track counts safely
                    track_count = 0
                    pass_count = 0
                    fail_count = 0

                    if hasattr(result, 'tracks'):
                        # Handle both dict (from analysis) and list (from DB) formats
                        if isinstance(result.tracks, dict):
                            track_count = len(result.tracks)
                            tracks_iter = result.tracks.values()
                        else:
                            track_count = len(result.tracks)
                            tracks_iter = result.tracks

                        for track in tracks_iter:
                            if hasattr(track, 'overall_status'):
                                track_status = getattr(track.status, 'value', str(track.status))
                                if track_status == 'Pass':
                                    pass_count += 1
                                else:
                                    fail_count += 1
                            elif hasattr(track, 'status'):
                                # Database tracks have 'status' not 'overall_status'
                                track_status = getattr(track.status, 'value', str(track.status))
                                if track_status == 'Pass':
                                    pass_count += 1
                                else:
                                    fail_count += 1

                    # Summary row
                    summary_data.append({
                        'File': file_name,
                        'Model': model,
                        'Serial': serial,
                        'System_Type': system_type,
                        'Trim_Date': trim_date,
                        'Overall_Status': overall_status,
                        'Validation_Status': validation_status,
                        'Processing_Time': f"{getattr(result, 'processing_time', 0):.2f}",
                        'Track_Count': track_count,
                        'Pass_Count': pass_count,
                        'Fail_Count': fail_count
                    })

                    # Individual track data - moved inside try block for safety
                    if hasattr(result, 'tracks') and result.tracks:
                        # Handle both dict and list formats
                        if isinstance(result.tracks, dict):
                            tracks_items = result.tracks.items()
                        else:
                            # For list format, create enumerated items
                            tracks_items = enumerate(result.tracks)

                        for track_id, track in tracks_items:
                            try:
                                track_row = {
                                    'File': file_name,
                                    'Model': model,
                                    'Serial': serial,
                                    'Track_ID': str(track_id),
                                    'Track_Status': getattr(track.status, 'value', 'Unknown') if hasattr(track, 'status') else 'Unknown',
                                    'Sigma_Gradient': track.sigma_analysis.sigma_gradient if hasattr(track, 'sigma_analysis') and track.sigma_analysis else None,
                                    'Sigma_Threshold': track.sigma_analysis.sigma_threshold if hasattr(track, 'sigma_analysis') and track.sigma_analysis else None,
                                    'Sigma_Pass': track.sigma_analysis.sigma_pass if hasattr(track, 'sigma_analysis') and track.sigma_analysis else None,
                                    'Linearity_Spec': track.linearity_analysis.linearity_spec if hasattr(track, 'linearity_analysis') and track.linearity_analysis else None,
                                    'Linearity_Pass': track.linearity_analysis.linearity_pass if hasattr(track, 'linearity_analysis') and track.linearity_analysis else None,
                                    'Risk_Category': track.risk_category.value if hasattr(track, 'risk_category') else 'Unknown'
                                }
                                all_track_data.append(track_row)
                            except Exception as track_error:
                                logger.error(f"Error processing track {track_id} for {file_name}: {track_error}")
                                # Continue with next track

                except Exception as e:
                    # Log error and continue with next result
                    logger.error(f"Error processing result for export: {e}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    # Add minimal data for this file
                    summary_data.append({
                        'File': file_name,
                        'Model': 'Error',
                        'Serial': 'Error',
                        'Error': str(e)
                    })
                    # Show warning about specific file
                    self._safe_after(0, lambda fn=file_name, err=str(e): messagebox.showwarning(
                        "Export Data Warning",
                        f"Could not export complete data for {fn}:\n{err}\n\n"
                        "Partial data will be included in the export."
                    ))

            # Ensure we have at least some data to write
            if not summary_data:
                # Create a minimal summary if no data was processed successfully
                summary_data.append({
                    'File': 'No data available',
                    'Model': 'N/A',
                    'Serial': 'N/A',
                    'Status': 'Export failed - no processable data'
                })

            # Write summary sheet
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Batch Summary', index=False)

            # Write track details sheet only if we have data
            if all_track_data:
                tracks_df = pd.DataFrame(all_track_data)
                tracks_df.to_excel(writer, sheet_name='Track Details', index=False)

            # Calculate statistics with error handling
            total_tracks = 0
            tracks_passed = 0
            tracks_failed = 0
            files_validated = 0
            files_warning = 0
            files_failed_validation = 0

            for result in self.batch_results.values():
                try:
                    # Count tracks safely
                    if hasattr(result, 'tracks'):
                        if isinstance(result.tracks, dict):
                            total_tracks += len(result.tracks)
                            for track in result.tracks.values():
                                if hasattr(track, 'overall_status'):
                                    if getattr(track.status, 'value', '') == 'Pass':
                                        tracks_passed += 1
                                    else:
                                        tracks_failed += 1
                                elif hasattr(track, 'status'):
                                    if getattr(track.status, 'value', '') == 'Pass':
                                        tracks_passed += 1
                                    else:
                                        tracks_failed += 1
                        elif isinstance(result.tracks, list):
                            total_tracks += len(result.tracks)
                            for track in result.tracks:
                                if hasattr(track, 'overall_status'):
                                    if getattr(track.status, 'value', '') == 'Pass':
                                        tracks_passed += 1
                                    else:
                                        tracks_failed += 1
                                elif hasattr(track, 'status'):
                                    if getattr(track.status, 'value', '') == 'Pass':
                                        tracks_passed += 1
                                    else:
                                        tracks_failed += 1

                    # Count validation status safely
                    if hasattr(result, 'overall_validation_status'):
                        val_status = getattr(result.overall_validation_status, 'value', '')
                        if val_status == 'VALIDATED':
                            files_validated += 1
                        elif val_status == 'WARNING':
                            files_warning += 1
                        elif val_status == 'FAILED':
                            files_failed_validation += 1

                except Exception as stat_error:
                    logger.error(f"Error calculating statistics: {stat_error}")

            # Statistics sheet
            pass_rate = f"{(tracks_passed / total_tracks * 100):.1f}%" if total_tracks > 0 else "0%"
            success_rate = f"{(len(self.batch_results) / len(self.selected_files) * 100):.1f}%" if len(self.selected_files) > 0 else "0%"

            stats_data = {
                'Metric': [
                    'Total Files Processed',
                    'Total Files Selected',
                    'Success Rate',
                    'Total Tracks Analyzed',
                    'Tracks Passed',
                    'Tracks Failed',
                    'Pass Rate',
                    'Files Validated',
                    'Files with Warnings',
                    'Files with Validation Errors'
                ],
                'Value': [
                    len(self.batch_results),
                    len(self.selected_files),
                    success_rate,
                    total_tracks,
                    tracks_passed,
                    tracks_failed,
                    pass_rate,
                    files_validated,
                    files_warning,
                    files_failed_validation
                ]
            }

            stats_df = pd.DataFrame(stats_data)
            stats_df.to_excel(writer, sheet_name='Statistics', index=False)

            # Ensure at least one sheet is visible by accessing the workbook
            # This is important to prevent the "at least one sheet must be visible" error
            if hasattr(writer, 'book'):
                writer.book.active = 0  # Make the first sheet (Batch Summary) active

    def _export_batch_csv(self, file_path: Path):
        """Export batch results to CSV format."""
        import pandas as pd

        # Create comprehensive CSV export
        export_data = []

        for file_path_str, result in self.batch_results.items():
            file_name = Path(file_path_str).name

            try:
                # Get metadata safely
                model = getattr(result.metadata, 'model', 'Unknown') if hasattr(result, 'metadata') else 'Unknown'
                serial = getattr(result.metadata, 'serial', 'Unknown') if hasattr(result, 'metadata') else 'Unknown'

                # Handle system safely (attribute is 'system', not 'system_type')
                system_type = 'Unknown'
                if hasattr(result, 'metadata') and hasattr(result.metadata, 'system'):
                    system_type = getattr(result.metadata.system, 'value', str(result.metadata.system))
                elif hasattr(result, 'metadata') and hasattr(result.metadata, 'system_type'):
                    # Legacy fallback
                    system_type = getattr(result.metadata.system_type, 'value', str(result.metadata.system_type))

                # Handle trim/test date (prefer test_date, fallback to file_date)
                trim_date = 'Unknown'
                if hasattr(result, 'metadata'):
                    if hasattr(result.metadata, 'test_date') and result.metadata.test_date:
                        if hasattr(result.metadata.test_date, 'strftime'):
                            trim_date = result.metadata.test_date.strftime('%Y-%m-%d %H:%M:%S')
                        else:
                            trim_date = str(result.metadata.test_date)
                    elif hasattr(result.metadata, 'file_date') and result.metadata.file_date:
                        if hasattr(result.metadata.file_date, 'strftime'):
                            trim_date = result.metadata.file_date.strftime('%Y-%m-%d %H:%M:%S')
                        else:
                            trim_date = str(result.metadata.file_date)
                    elif hasattr(result.metadata, 'analysis_date'):
                        # Legacy fallback
                        if hasattr(result.metadata.analysis_date, 'strftime'):
                            trim_date = result.metadata.analysis_date.strftime('%Y-%m-%d %H:%M:%S')
                        else:
                            trim_date = str(result.metadata.analysis_date)

                # Handle overall_status safely
                overall_status = getattr(result.overall_status, 'value', 'Unknown') if hasattr(result, 'overall_status') else 'Unknown'
                processing_time = f"{getattr(result, 'processing_time', 0):.2f}"
                validation_status = getattr(result.overall_validation_status, 'value', 'N/A') if hasattr(result, 'overall_validation_status') else 'N/A'

                # Process tracks
                if hasattr(result, 'tracks') and result.tracks:
                    # Handle both dict and list formats
                    if isinstance(result.tracks, dict):
                        tracks_items = result.tracks.items()
                    else:
                        tracks_items = enumerate(result.tracks)

                    for track_id, track in tracks_items:
                        try:
                            row = {
                                'File': file_name,
                                'Model': model,
                                'Serial': serial,
                                'System_Type': system_type,
                                'Trim_Date': trim_date,
                                'Track_ID': str(track_id),
                                'Overall_Status': overall_status,
                                'Track_Status': getattr(track.overall_status, 'value', 'Unknown') if hasattr(track, 'overall_status') else getattr(track.status, 'value', 'Unknown') if hasattr(track, 'status') else 'Unknown',
                                'Processing_Time': processing_time,
                                'Validation_Status': validation_status
                            }

                            # Add detailed analysis data
                            if hasattr(track, 'sigma_analysis') and track.sigma_analysis:
                                row.update({
                                    'Sigma_Gradient': getattr(track.sigma_analysis, 'sigma_gradient', None),
                                    'Sigma_Threshold': getattr(track.sigma_analysis, 'sigma_threshold', None),
                                    'Sigma_Pass': getattr(track.sigma_analysis, 'sigma_pass', None),
                                    'Sigma_Improvement': getattr(track.sigma_analysis, 'improvement_percent', None)
                                })

                            if hasattr(track, 'linearity_analysis') and track.linearity_analysis:
                                row.update({
                                    'Linearity_Spec': getattr(track.linearity_analysis, 'linearity_spec', None),
                                    'Linearity_Pass': getattr(track.linearity_analysis, 'linearity_pass', None),
                                    'Linearity_Error': getattr(track.linearity_analysis, 'linearity_error', None)
                                })

                            if hasattr(track, 'resistance_analysis') and track.resistance_analysis:
                                row.update({
                                    'Resistance_Before': getattr(track.resistance_analysis, 'resistance_before', None),
                                    'Resistance_After': getattr(track.resistance_analysis, 'resistance_after', None),
                                    'Resistance_Change_Percent': getattr(track.resistance_analysis, 'resistance_change_percent', None)
                                })

                            if hasattr(track, 'risk_category'):
                                row['Risk_Category'] = getattr(track.risk_category, 'value', 'Unknown')

                            export_data.append(row)

                        except Exception as track_error:
                            logger.error(f"Error processing track {track_id} for CSV export: {track_error}")
                            # Add minimal row for this track
                            export_data.append({
                                'File': file_name,
                                'Model': model,
                                'Serial': serial,
                                'Track_ID': str(track_id),
                                'Error': str(track_error)
                            })
                else:
                    # No tracks - add a summary row
                    export_data.append({
                        'File': file_name,
                        'Model': model,
                        'Serial': serial,
                        'System_Type': system_type,
                        'Trim_Date': trim_date,
                        'Overall_Status': overall_status,
                        'Processing_Time': processing_time,
                        'Validation_Status': validation_status,
                        'Track_Count': 0
                    })

            except Exception as e:
                logger.error(f"Error processing result for CSV export: {e}")
                # Add error row
                export_data.append({
                    'File': file_name,
                    'Error': str(e)
                })

        # Ensure we have at least one row
        if not export_data:
            export_data.append({
                'File': 'No data available',
                'Status': 'Export failed - no processable data'
            })

        # Convert to DataFrame and save
        df = pd.DataFrame(export_data)
        df.to_csv(file_path, index=False)
