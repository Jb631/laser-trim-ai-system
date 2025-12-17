"""
ExportMixin - Export/report functionality for MultiTrackPage.

This module provides multi-track export methods:
- _export_comparison_report: Export Excel comparison report
- _generate_pdf_report: Generate comprehensive PDF report
- _generate_risk_assessment_text: Generate risk assessment text

Extracted from multi_track_page.py during Phase 4 file splitting.
"""

import logging
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox

import pandas as pd

logger = logging.getLogger(__name__)


class ExportMixin:
    """
    Mixin providing multi-track export functionality.

    Requires MultiTrackPage as parent class with:
    - self.current_unit_data: Dict of current unit data
    - self.comparison_data: Dict of comparison results
    - self.logger: Logger instance
    """

    def _export_comparison_report(self):
        """Export multi-track comparison report to Excel."""
        if not self.current_unit_data:
            messagebox.showwarning("No Data", "No unit data available to export")
            return

        try:
            # Get save location
            unit_id = f"{self.current_unit_data.get('model', 'Unknown')}_{self.current_unit_data.get('serial', 'Unknown')}"
            initial_filename = f"multitrack_report_{unit_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

            filename = filedialog.asksaveasfilename(
                title="Export Multi-Track Report",
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
                initialfile=initial_filename
            )

            if not filename:
                return

            # Create comprehensive Excel export
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Unit summary
                unit_summary = pd.DataFrame([{
                    'Model': self.current_unit_data.get('model', 'Unknown'),
                    'Serial': self.current_unit_data.get('serial', 'Unknown'),
                    'Track Count': self.current_unit_data.get('track_count', 0),
                    'File Count': self.current_unit_data.get('total_files', 0),
                    'Overall Status': self.current_unit_data.get('overall_status', 'Unknown'),
                    'Consistency': self.current_unit_data.get('consistency', 'Unknown'),
                    'Sigma CV': f"{self.current_unit_data.get('sigma_cv', 0):.1f}%",
                    'Linearity CV': f"{self.current_unit_data.get('linearity_cv', 0):.1f}%",
                    'Resistance CV': f"{self.current_unit_data.get('resistance_cv', 0):.1f}%",
                    'Analysis Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }])
                unit_summary.to_excel(writer, sheet_name='Unit Summary', index=False)

                # Track comparison data
                if self.comparison_data and self.comparison_data.get('comparison_performed'):
                    comparison_df = pd.DataFrame([self.comparison_data])
                    comparison_df.to_excel(writer, sheet_name='Comparison Analysis', index=False)

                # Individual track data with validation information
                track_data = []
                # Extract tracks from files structure
                all_tracks = {}
                if 'files' in self.current_unit_data:
                    for file_data in self.current_unit_data.get('files', []):
                        file_tracks = file_data.get('tracks', {})
                        all_tracks.update(file_tracks)
                elif 'tracks' in self.current_unit_data:
                    all_tracks = self.current_unit_data['tracks']

                for track_id, track_info in all_tracks.items():
                    # Handle dictionary format
                    if isinstance(track_info, dict):
                        track_record = {
                            'Track ID': track_id,
                            'Status': track_info.get('overall_status', 'Unknown'),
                            'Sigma Gradient': track_info.get('sigma_gradient', 0),
                            'Sigma Pass': track_info.get('sigma_pass', False),
                            'Sigma Margin': track_info.get('sigma_margin', 0),
                            'Linearity Error': track_info.get('linearity_error', 0),
                            'Linearity Pass': track_info.get('linearity_pass', False),
                            'Linearity Spec': track_info.get('linearity_spec', 0),
                            'Resistance Change %': track_info.get('resistance_change_percent', 0),
                            'Travel Length': track_info.get('travel_length', 0),
                            'File Path': track_info.get('file_path', 'N/A')
                        }
                        track_data.append(track_record)
                    else:
                        # Handle object format (legacy)
                        try:
                            primary_track = track_info.primary_track if hasattr(track_info, 'primary_track') else track_info
                            track_record = {
                                'Track ID': track_id,
                                'Status': primary_track.status.value if hasattr(primary_track.status, 'value') else str(primary_track.status),
                                'Sigma Gradient': primary_track.sigma_analysis.sigma_gradient if hasattr(primary_track, 'sigma_analysis') else 0,
                                'Sigma Pass': primary_track.sigma_analysis.sigma_pass if hasattr(primary_track, 'sigma_analysis') else False,
                                'Linearity Error': primary_track.linearity_analysis.final_linearity_error_shifted if hasattr(primary_track, 'linearity_analysis') else 0,
                                'Linearity Pass': primary_track.linearity_analysis.linearity_pass if hasattr(primary_track, 'linearity_analysis') else False,
                                'Resistance Change %': primary_track.unit_properties.resistance_change_percent if hasattr(primary_track, 'unit_properties') else 0
                            }
                            track_data.append(track_record)
                        except:
                            pass

                if track_data:
                    tracks_df = pd.DataFrame(track_data)
                    tracks_df.to_excel(writer, sheet_name='Track Details', index=False)

                # Validation summary sheet
                validation_summary_data = []
                for track_id, track_info in all_tracks.items():
                    if isinstance(track_info, dict):
                        # Create validation summary from available data
                        validation_record = {
                            'Track ID': track_id,
                            'Overall Status': track_info.get('overall_status', 'Unknown'),
                            'Sigma Test': 'PASS' if track_info.get('sigma_pass', False) else 'FAIL',
                            'Sigma Gradient': track_info.get('sigma_gradient', 0),
                            'Sigma Threshold': track_info.get('sigma_spec', 0),
                            'Linearity Test': 'PASS' if track_info.get('linearity_pass', False) else 'FAIL',
                            'Linearity Error': track_info.get('linearity_error', 0),
                            'Linearity Spec': track_info.get('linearity_spec', 0),
                            'Resistance Test': track_info.get('validation_status', 'Unknown'),
                            'Resistance Change %': track_info.get('resistance_change_percent', 0),
                            'Industry Grade': track_info.get('industry_grade', 'N/A')
                        }
                        validation_summary_data.append(validation_record)

                if validation_summary_data:
                    validation_df = pd.DataFrame(validation_summary_data)
                    validation_df.to_excel(writer, sheet_name='Validation Summary', index=False)

            messagebox.showinfo("Export Complete", f"Multi-track report exported to:\n{filename}")
            self.logger.info(f"Exported multi-track report to {filename}")

        except Exception as e:
            error_msg = f"Failed to export report: {str(e)}"
            messagebox.showerror("Export Error", error_msg)
            self.logger.error(f"Export failed: {e}")

    def _generate_pdf_report(self):
        """Generate PDF report for multi-track analysis."""
        if not self.current_unit_data:
            messagebox.showwarning("No Data", "No multi-track data available to generate report")
            return

        # Extract tracks from files structure
        all_tracks = {}
        if 'files' in self.current_unit_data:
            for file_data in self.current_unit_data.get('files', []):
                file_tracks = file_data.get('tracks', {})
                all_tracks.update(file_tracks)
        elif 'tracks' in self.current_unit_data:
            all_tracks = self.current_unit_data['tracks']

        if not all_tracks:
            messagebox.showwarning("No Data", "No multi-track data available to generate report")
            return

        # Ask for save location
        default_filename = f"{self.current_unit_data.get('model', 'unit')}_{self.current_unit_data.get('serial', 'unknown')}_multi_track_report.pdf"
        filename = filedialog.asksaveasfilename(
            title="Save PDF Report",
            defaultextension=".pdf",
            initialfile=default_filename,
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )

        if not filename:
            return

        try:
            # Import matplotlib backends for PDF
            from matplotlib.backends.backend_pdf import PdfPages
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec

            # Create PDF with multiple pages
            with PdfPages(filename) as pdf:
                # Page 1: Summary and Overview
                fig = plt.figure(figsize=(8.5, 11))
                fig.suptitle(f'Multi-Track Analysis Report\n{self.current_unit_data.get("model", "N/A")} - {self.current_unit_data.get("serial", "N/A")}',
                            fontsize=16, fontweight='bold')

                # Create grid layout
                gs = GridSpec(6, 2, figure=fig, hspace=0.4, wspace=0.3)

                # Summary text
                ax_summary = fig.add_subplot(gs[0:2, :])
                ax_summary.axis('off')

                summary_text = f"""
Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Unit Information:
* Model: {self.current_unit_data.get('model', 'N/A')}
* Serial: {self.current_unit_data.get('serial', 'N/A')}
* Total Tracks: {self.current_unit_data.get('track_count', 0)}
* Overall Status: {self.current_unit_data.get('overall_status', 'N/A')}

Consistency Analysis:
* Consistency Grade: {self.current_unit_data.get('consistency', 'N/A')}
* Sigma CV: {self.current_unit_data.get('sigma_cv', 0):.2f}%
* Linearity CV: {self.current_unit_data.get('linearity_cv', 0):.2f}%
"""
                ax_summary.text(0.1, 0.9, summary_text, transform=ax_summary.transAxes,
                               fontsize=10, verticalalignment='top', fontfamily='monospace')

                # Track summary table
                ax_table = fig.add_subplot(gs[2:4, :])
                ax_table.axis('off')

                # Prepare table data
                table_data = [['Track ID', 'Status', 'Sigma Gradient', 'Linearity Error (V)']]

                if all_tracks:
                    for track_id, result in all_tracks.items():
                        if isinstance(result, dict):
                            # Handle dictionary format
                            table_data.append([
                                track_id,
                                result.get('overall_status', 'Unknown'),
                                f"{result.get('sigma_gradient', 0):.6f}",
                                f"{result.get('linearity_error', 0):.4f}"
                            ])
                        elif hasattr(result, 'primary_track') and result.primary_track:
                            # Handle object format
                            primary_track = result.primary_track
                            table_data.append([
                                track_id,
                                primary_track.status.value,
                                f"{primary_track.sigma_analysis.sigma_gradient:.6f}",
                                f"{primary_track.linearity_analysis.final_linearity_error_shifted:.4f}"
                            ])

                if len(table_data) > 1:
                    table = ax_table.table(cellText=table_data[1:],
                                         colLabels=table_data[0],
                                         cellLoc='center',
                                         loc='center',
                                         colWidths=[0.2, 0.2, 0.3, 0.3])
                    table.auto_set_font_size(False)
                    table.set_fontsize(9)
                    table.scale(1.2, 1.5)

                    # Color code status cells
                    for i in range(1, len(table_data)):
                        status = table_data[i][1]
                        if status == 'PASS':
                            table[(i, 1)].set_facecolor('#90EE90')
                        elif status == 'FAIL':
                            table[(i, 1)].set_facecolor('#FFB6C1')
                        elif status == 'WARNING':
                            table[(i, 1)].set_facecolor('#FFFFE0')

                # Risk Assessment
                ax_risk = fig.add_subplot(gs[4:6, :])
                ax_risk.axis('off')

                risk_text = self._generate_risk_assessment_text()
                ax_risk.text(0.1, 0.9, risk_text, transform=ax_risk.transAxes,
                           fontsize=10, verticalalignment='top', fontfamily='monospace')

                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

                # Page 2: Track Comparison Charts
                if all_tracks and len(all_tracks) > 0:
                    fig2 = plt.figure(figsize=(8.5, 11))
                    fig2.suptitle('Track Comparison Charts', fontsize=16, fontweight='bold')

                    # Prepare data for charts
                    track_ids = []
                    sigma_values = []
                    linearity_values = []

                    for track_id, track_data in all_tracks.items():
                        track_ids.append(track_id)
                        # Handle different data structures
                        if isinstance(track_data, dict):
                            sigma_values.append(track_data.get('sigma_gradient', 0))
                            linearity_values.append(abs(track_data.get('linearity_error', 0)))
                        elif hasattr(track_data, 'sigma_gradient'):
                            sigma_values.append(getattr(track_data, 'sigma_gradient', 0))
                            linearity_values.append(abs(getattr(track_data, 'linearity_error', 0)))

                    if track_ids:
                        # Sigma comparison
                        ax1 = fig2.add_subplot(2, 1, 1)
                        bars1 = ax1.bar(track_ids, sigma_values, color='skyblue', edgecolor='navy')
                        ax1.set_xlabel('Track ID')
                        ax1.set_ylabel('Sigma Gradient')
                        ax1.set_title('Sigma Gradient by Track')
                        ax1.grid(True, alpha=0.3)

                        # Add value labels on bars
                        for bar, value in zip(bars1, sigma_values):
                            height = bar.get_height()
                            ax1.text(bar.get_x() + bar.get_width()/2., height,
                                   f'{value:.6f}', ha='center', va='bottom', fontsize=8)

                        # Linearity comparison
                        ax2 = fig2.add_subplot(2, 1, 2)
                        bars2 = ax2.bar(track_ids, linearity_values, color='lightcoral', edgecolor='darkred')
                        ax2.set_xlabel('Track ID')
                        ax2.set_ylabel('Linearity Error (V)')
                        ax2.set_title('Linearity Error by Track')
                        ax2.grid(True, alpha=0.3)

                        # Add value labels on bars
                        for bar, value in zip(bars2, linearity_values):
                            height = bar.get_height()
                            ax2.text(bar.get_x() + bar.get_width()/2., height,
                                   f'{value:.4f}', ha='center', va='bottom', fontsize=8)

                    plt.tight_layout()
                    pdf.savefig(fig2, bbox_inches='tight')
                    plt.close(fig2)

                # Page 3: Individual Track Error Plots (if available)
                track_count = 0
                tracks_per_page = 4
                track_list = []

                if 'tracks' in self.current_unit_data:
                    for track_id, track_data in all_tracks.items():
                        # Handle different data structures
                        if isinstance(track_data, dict):
                            # Check for position and error data
                            if 'position_data' in track_data and 'error_data' in track_data:
                                if track_data['position_data'] and track_data['error_data']:
                                    track_list.append((track_id, track_data))
                        elif hasattr(track_data, 'position_data') and hasattr(track_data, 'error_data'):
                            if track_data.position_data and track_data.error_data:
                                track_list.append((track_id, track_data))

                if track_list:
                    # Create pages with 4 tracks each
                    for page_start in range(0, len(track_list), tracks_per_page):
                        fig3 = plt.figure(figsize=(8.5, 11))
                        fig3.suptitle('Track Error Plots', fontsize=16, fontweight='bold')

                        page_tracks = track_list[page_start:page_start + tracks_per_page]

                        for i, (track_id, track_data) in enumerate(page_tracks):
                            ax = fig3.add_subplot(2, 2, i + 1)

                            # Get position and error data
                            if isinstance(track_data, dict):
                                positions = track_data.get('position_data', [])
                                errors = track_data.get('error_data', [])
                                spec_limit = track_data.get('linearity_spec', 0.01)
                            else:
                                positions = track_data.position_data
                                errors = track_data.error_data
                                spec_limit = getattr(track_data, 'linearity_spec', 0.01)

                            # Plot error data
                            ax.plot(positions, errors, 'b-', linewidth=1.5, label='Error')

                            # Add zero line
                            ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)

                            # Add spec limit lines
                            ax.axhline(y=spec_limit, color='r', linestyle='--', alpha=0.7, label=f'Spec: +/-{spec_limit:.4f}V')
                            ax.axhline(y=-spec_limit, color='r', linestyle='--', alpha=0.7)

                            ax.set_xlabel('Position (mm)')
                            ax.set_ylabel('Error (V)')
                            ax.set_title(f'Track {track_id}')
                            ax.grid(True, alpha=0.3)
                            ax.legend(fontsize=8)

                        plt.tight_layout()
                        pdf.savefig(fig3, bbox_inches='tight')
                        plt.close(fig3)

            messagebox.showinfo("Success", f"PDF report saved to:\n{filename}")
            self.logger.info(f"PDF report generated: {filename}")

        except Exception as e:
            self.logger.error(f"Failed to generate PDF report: {e}")
            messagebox.showerror("Error", f"Failed to generate PDF report:\n{str(e)}")

    def _generate_risk_assessment_text(self) -> str:
        """Generate risk assessment text based on consistency analysis."""
        consistency = self.current_unit_data.get('consistency', 'UNKNOWN')
        sigma_cv = self.current_unit_data.get('sigma_cv', 0)
        linearity_cv = self.current_unit_data.get('linearity_cv', 0)

        risk_level = "UNKNOWN"
        recommendations = []

        if consistency == 'EXCELLENT':
            risk_level = "LOW"
            recommendations = [
                "* Excellent track-to-track consistency",
                "* Continue current manufacturing process",
                "* Regular monitoring recommended"
            ]
        elif consistency == 'GOOD':
            risk_level = "LOW-MEDIUM"
            recommendations = [
                "* Good overall consistency",
                "* Minor variations detected",
                "* Review process parameters periodically"
            ]
        elif consistency == 'FAIR':
            risk_level = "MEDIUM"
            recommendations = [
                "* Moderate consistency issues detected",
                "* Review laser trimming parameters",
                "* Consider process optimization"
            ]
        elif consistency == 'POOR':
            risk_level = "HIGH"
            recommendations = [
                "* Significant track-to-track variations",
                "* Immediate process review recommended",
                "* Check equipment calibration",
                "* Consider re-trimming if possible"
            ]

        text = f"""
Risk Assessment:
* Risk Level: {risk_level}
* Consistency Grade: {consistency}

Key Metrics:
* Sigma Coefficient of Variation: {sigma_cv:.2f}%
* Linearity Coefficient of Variation: {linearity_cv:.2f}%

Recommendations:
{chr(10).join(recommendations)}
"""
        return text
