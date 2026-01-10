"""PDF export functionality for Trends page."""

import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches

logger = logging.getLogger(__name__)


def export_trends_summary_pdf(
    active_models_data: List[Dict],
    output_path: Path
) -> None:
    """
    Export summary view to PDF - statistics table for all active models.

    Args:
        active_models_data: List of model summary dicts from get_active_models_summary()
        output_path: Path to save PDF
    """
    plt.style.use('default')

    with PdfPages(output_path) as pdf:
        fig = plt.figure(figsize=(11, 8.5), facecolor='white')
        fig.suptitle('Trends Summary - All Active Models',
                     fontsize=16, fontweight='bold', y=0.95)

        # Create table data
        headers = ['Model', 'Total\nSamples', 'Sigma\nPass %', 'Linearity\nPass %',
                   'Overall\nPass %', 'Avg\nSigma', 'Anomalies']

        table_data = []
        for model_data in active_models_data:
            row = [
                model_data.get('model', 'Unknown'),
                str(model_data.get('total_samples', 0)),
                f"{model_data.get('sigma_pass_rate', 0):.1f}%",
                f"{model_data.get('linearity_pass_rate', 0):.1f}%",
                f"{model_data.get('overall_pass_rate', 0):.1f}%",
                f"{model_data.get('avg_sigma', 0):.4f}",
                str(model_data.get('anomaly_count', 0)),
            ]
            table_data.append(row)

        # Create table
        ax = fig.add_subplot(111)
        ax.axis('tight')
        ax.axis('off')

        table = ax.table(
            cellText=table_data,
            colLabels=headers,
            cellLoc='center',
            loc='center',
            colWidths=[0.15, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # Style header
        for i in range(len(headers)):
            cell = table[(0, i)]
            cell.set_facecolor('#4A90E2')
            cell.set_text_props(weight='bold', color='white')

        # Color-code rows by overall pass rate
        for i, row_data in enumerate(table_data):
            pass_rate_str = row_data[4].rstrip('%')
            pass_rate = float(pass_rate_str)

            if pass_rate >= 90:
                color = '#90EE90'  # Light green
            elif pass_rate >= 80:
                color = '#FFFF99'  # Light yellow
            else:
                color = '#FFB6C1'  # Light red

            for j in range(len(headers)):
                table[(i+1, j)].set_facecolor(color)

        # Add metadata
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.5, 0.05, f'Generated: {timestamp}',
                ha='center', fontsize=8, color='gray')

        pdf.savefig(fig, facecolor='white')
        plt.close(fig)

    logger.info(f"Exported trends summary PDF to: {output_path}")


def export_trends_detail_pdf(
    model: str,
    trend_data: Dict[str, Any],
    model_stats: Dict[str, Any],
    output_path: Path
) -> None:
    """
    Export detail view to PDF - multi-page report with charts and stats.

    Args:
        model: Model number
        trend_data: Output from get_model_trend_data()
        model_stats: Dict of model statistics (from detail view labels)
        output_path: Path to save PDF
    """
    plt.style.use('default')

    data_points = trend_data.get("data_points", [])
    sigma_threshold = trend_data.get("threshold")
    linearity_spec = trend_data.get("linearity_spec")
    linearity_pass_rates = trend_data.get("linearity_pass_rates_by_day", [])

    # Filter out anomalies
    filtered_points = [dp for dp in data_points if not dp.get("is_anomaly", False)]

    with PdfPages(output_path) as pdf:
        # PAGE 1: Overview + Sigma Trend
        _create_sigma_page(pdf, model, filtered_points, sigma_threshold, model_stats)

        # PAGE 2: Linearity Pass Rate Trend
        _create_linearity_page(pdf, model, linearity_pass_rates, linearity_spec, model_stats)

        # PAGE 3: Distributions
        _create_distributions_page(pdf, model, filtered_points)

    logger.info(f"Exported trends detail PDF to: {output_path}")


def _create_sigma_page(pdf, model, data_points, threshold, stats):
    """Create page 1: Sigma trend chart with stats."""
    fig = plt.figure(figsize=(11, 8.5), facecolor='white')
    fig.suptitle(f'Sigma Gradient Trend - {model}',
                 fontsize=14, fontweight='bold', y=0.96)

    gs = fig.add_gridspec(3, 1, hspace=0.3, height_ratios=[1, 3, 1],
                          left=0.1, right=0.95, top=0.92, bottom=0.08)

    # Stats table at top
    ax_stats = fig.add_subplot(gs[0])
    ax_stats.axis('off')

    stats_text = (
        f"Total Samples: {stats.get('total_samples', '--')}    "
        f"Anomalies: {stats.get('anomalies', '--')}    "
        f"Sigma Pass: {stats.get('sigma_pass_rate', '--')}    "
        f"Overall Pass: {stats.get('overall_pass_rate', '--')}    "
        f"Avg Sigma: {stats.get('avg_sigma', '--')}    "
        f"Threshold: {stats.get('threshold', '--')}"
    )
    ax_stats.text(0.5, 0.5, stats_text, ha='center', va='center', fontsize=10)

    # Sigma scatter chart
    ax_chart = fig.add_subplot(gs[1])

    if len(data_points) >= 2:
        dates = [dp["date"] for dp in data_points]
        sigmas = [dp["sigma_gradient"] for dp in data_points]
        pass_flags = [dp["sigma_pass"] for dp in data_points]

        # Plot points colored by pass/fail
        for date, sigma, passed in zip(dates, sigmas, pass_flags):
            color = 'green' if passed else 'red'
            ax_chart.scatter(date, sigma, c=color, s=30, alpha=0.6)

        # Threshold line
        if threshold:
            ax_chart.axhline(y=threshold, color='blue', linestyle='--',
                           linewidth=1.5, label=f'Threshold: {threshold:.4f}')

        # Rolling average
        window = min(30, len(sigmas))
        rolling_avg = []
        for i in range(len(sigmas)):
            start = max(0, i - window + 1)
            rolling_avg.append(sum(sigmas[start:i+1]) / (i - start + 1))
        ax_chart.plot(dates, rolling_avg, 'b-', linewidth=2, alpha=0.7, label='30-Day Rolling Avg')

        # Set Y-axis limits based on percentiles to handle outliers
        sorted_sigmas = sorted(sigmas)
        p1 = sorted_sigmas[int(len(sorted_sigmas) * 0.01)]  # 1st percentile
        p99 = sorted_sigmas[int(len(sorted_sigmas) * 0.99)]  # 99th percentile
        y_range = p99 - p1
        y_min = max(0, p1 - y_range * 0.1)  # Add 10% padding
        y_max = p99 + y_range * 0.1
        ax_chart.set_ylim(y_min, y_max)

        ax_chart.set_xlabel('Date', fontsize=10)
        ax_chart.set_ylabel('Sigma Gradient', fontsize=10)
        ax_chart.legend(loc='upper right')
        ax_chart.grid(True, alpha=0.3)
        ax_chart.tick_params(axis='x', rotation=45)
    else:
        ax_chart.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
        ax_chart.axis('off')

    # Legend at bottom
    ax_legend = fig.add_subplot(gs[2])
    ax_legend.axis('off')

    pass_patch = mpatches.Patch(color='green', alpha=0.6, label='PASS')
    fail_patch = mpatches.Patch(color='red', alpha=0.6, label='FAIL')
    ax_legend.legend(handles=[pass_patch, fail_patch], loc='center', ncol=2, fontsize=10)

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    fig.text(0.5, 0.02, f'Generated: {timestamp}', ha='center', fontsize=8, color='gray')

    pdf.savefig(fig, facecolor='white')
    plt.close(fig)


def _create_linearity_page(pdf, model, linearity_pass_rates_by_day, spec, stats):
    """Create page 2: Linearity pass rate trend chart (aggregated by day)."""
    fig = plt.figure(figsize=(11, 8.5), facecolor='white')
    fig.suptitle(f'Linearity Pass Rate Trend - {model}',
                 fontsize=14, fontweight='bold', y=0.96)

    gs = fig.add_gridspec(3, 1, hspace=0.3, height_ratios=[1, 3, 1],
                          left=0.1, right=0.95, top=0.92, bottom=0.08)

    # Stats at top
    ax_stats = fig.add_subplot(gs[0])
    ax_stats.axis('off')

    stats_text = (
        f"Linearity Pass Rate: {stats.get('linearity_pass_rate', '--')}    "
        f"Linearity Spec: {spec if spec else 'N/A'}"
    )
    ax_stats.text(0.5, 0.5, stats_text, ha='center', va='center', fontsize=10)

    # Linearity pass rate chart
    ax_chart = fig.add_subplot(gs[1])

    if len(linearity_pass_rates_by_day) >= 2:
        dates = [datetime.strptime(pr["date"], "%Y-%m-%d") for pr in linearity_pass_rates_by_day]
        pass_rates = [pr["pass_rate"] for pr in linearity_pass_rates_by_day]
        totals = [pr["total"] for pr in linearity_pass_rates_by_day]

        # Plot pass rate line
        ax_chart.plot(dates, pass_rates, color='#4A90E2', linewidth=2,
                     marker='o', markersize=4, alpha=0.8, label='Daily Pass Rate')

        # Rolling average (30-day weighted)
        window = min(30, len(pass_rates))
        rolling_avg = []
        for i in range(len(pass_rates)):
            start = max(0, i - window + 1)
            window_rates = pass_rates[start:i + 1]
            window_totals = totals[start:i + 1]
            weighted_avg = sum(r * t for r, t in zip(window_rates, window_totals)) / sum(window_totals)
            rolling_avg.append(weighted_avg)
        ax_chart.plot(dates, rolling_avg, 'b-', linewidth=3, alpha=0.7, label='30-Day Rolling Avg')

        # Target line at 80%
        ax_chart.axhline(y=80, color='orange', linestyle='--',
                       linewidth=2, alpha=0.7, label='80% Target')

        ax_chart.set_xlabel('Date', fontsize=10)
        ax_chart.set_ylabel('Pass Rate (%)', fontsize=10)
        ax_chart.set_ylim(0, 105)  # 0-100% range with padding
        ax_chart.legend(loc='lower left', fontsize=9)
        ax_chart.grid(True, alpha=0.3)
        ax_chart.tick_params(axis='x', rotation=45)
    else:
        ax_chart.text(0.5, 0.5, 'No linearity data available', ha='center', va='center')
        ax_chart.axis('off')

    # Note about visualization
    ax_legend = fig.add_subplot(gs[2])
    ax_legend.axis('off')

    note_text = (
        "Note: Pass rate aggregated by day. Shows % of units with 0 fail points. "
        "Helps identify if linearity performance is improving or declining over time."
    )
    ax_legend.text(0.5, 0.5, note_text, ha='center', va='center', fontsize=9,
                   style='italic', color='gray')

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    fig.text(0.5, 0.02, f'Generated: {timestamp}', ha='center', fontsize=8, color='gray')

    pdf.savefig(fig, facecolor='white')
    plt.close(fig)


def _create_distributions_page(pdf, model, data_points):
    """Create page 3: Sigma and fail points distributions (excludes anomalies)."""
    fig = plt.figure(figsize=(11, 8.5), facecolor='white')
    fig.suptitle(f'Distributions - {model} (Anomalies Excluded)',
                 fontsize=14, fontweight='bold', y=0.96)

    gs = fig.add_gridspec(2, 1, hspace=0.3, left=0.1, right=0.95, top=0.92, bottom=0.08)

    # Filter out anomalies for cleaner distributions
    normal_points = [dp for dp in data_points if not dp.get("is_anomaly", False)]
    anomaly_count = len(data_points) - len(normal_points)

    # Sigma distribution (normal samples only)
    ax_sigma = fig.add_subplot(gs[0])
    sigmas = [dp["sigma_gradient"] for dp in normal_points if dp["sigma_gradient"] is not None]

    if len(sigmas) >= 20:
        # Use percentile-based range to exclude outliers from bins
        sorted_sigmas = sorted(sigmas)
        p1 = sorted_sigmas[int(len(sorted_sigmas) * 0.01)]  # 1st percentile
        p99 = sorted_sigmas[int(len(sorted_sigmas) * 0.99)]  # 99th percentile

        # Filter data to percentile range for cleaner histogram
        filtered_sigmas = [s for s in sigmas if p1 <= s <= p99]

        bins = 50
        counts, bins_edges, patches = ax_sigma.hist(filtered_sigmas, bins=bins, color='#4A90E2',
                                                      alpha=0.7, edgecolor='black', linewidth=0.5)
        ax_sigma.set_xlabel('Sigma Gradient', fontsize=10)
        ax_sigma.set_ylabel('Frequency', fontsize=10)

        # Show total count and outlier count in title
        outlier_count = len(sigmas) - len(filtered_sigmas)
        title = f'Sigma Gradient Distribution (n={len(filtered_sigmas)}'
        if outlier_count > 0:
            title += f', {outlier_count} outliers excluded)'
        else:
            title += ')'
        ax_sigma.set_title(title, fontsize=12, fontweight='bold')
        ax_sigma.grid(True, alpha=0.3, axis='y')

        # Add mean and median lines (from filtered data)
        mean_val = sum(filtered_sigmas) / len(filtered_sigmas)
        median_val = sorted(filtered_sigmas)[len(filtered_sigmas)//2]
        ax_sigma.axvline(mean_val, color='red', linestyle='--', linewidth=1.5,
                        label=f'Mean: {mean_val:.4f}', alpha=0.7)
        ax_sigma.axvline(median_val, color='orange', linestyle='--', linewidth=1.5,
                        label=f'Median: {median_val:.4f}', alpha=0.7)
        ax_sigma.legend(fontsize=9)
    else:
        ax_sigma.text(0.5, 0.5, 'Insufficient data for distribution',
                     ha='center', va='center', transform=ax_sigma.transAxes)
        ax_sigma.set_title('Sigma Gradient Distribution', fontsize=12, fontweight='bold')

    # Fail points distribution
    ax_fail_points = fig.add_subplot(gs[1])
    fail_points_data = [dp.get("fail_points", 0) for dp in normal_points
                        if dp.get("fail_points") is not None]

    if len(fail_points_data) >= 20:
        # Create bins that make sense for fail points (0, 1-5, 6-10, etc.)
        max_fp = max(fail_points_data)
        if max_fp <= 10:
            bins = range(0, max_fp + 2)
        else:
            bins = 30

        counts, bins_edges, patches = ax_fail_points.hist(fail_points_data, bins=bins,
                                                           color='#90EE90', alpha=0.7,
                                                           edgecolor='black', linewidth=0.5)
        ax_fail_points.set_xlabel('Fail Points Count', fontsize=10)
        ax_fail_points.set_ylabel('Frequency', fontsize=10)
        ax_fail_points.set_title(f'Fail Points Distribution (n={len(fail_points_data)})',
                                fontsize=12, fontweight='bold')
        ax_fail_points.grid(True, alpha=0.3, axis='y')

        # Add pass rate annotation
        pass_count = sum(1 for fp in fail_points_data if fp == 0)
        pass_rate = (pass_count / len(fail_points_data)) * 100
        ax_fail_points.text(0.98, 0.98, f'Pass Rate: {pass_rate:.1f}%\n(0 fail points)',
                           transform=ax_fail_points.transAxes, fontsize=10,
                           verticalalignment='top', horizontalalignment='right',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax_fail_points.text(0.5, 0.5, 'Insufficient data for distribution',
                           ha='center', va='center', transform=ax_fail_points.transAxes)
        ax_fail_points.set_title('Fail Points Distribution', fontsize=12, fontweight='bold')

    # Footer with note
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    footer_text = f'Generated: {timestamp}'
    if anomaly_count > 0:
        footer_text += f'  |  {anomaly_count} anomalies excluded from distributions'
    fig.text(0.5, 0.02, footer_text, ha='center', fontsize=8, color='gray')

    pdf.savefig(fig, facecolor='white')
    plt.close(fig)
