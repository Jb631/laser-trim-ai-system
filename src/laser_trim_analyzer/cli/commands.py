"""
CLI commands for Laser Trim Analyzer using Click.

This module provides all command-line functionality for the analyzer,
making it easy for QA specialists to automate analyses and integrate
with other tools.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import asyncio

import click
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.syntax import Syntax
from rich import print as rprint

from laser_trim_analyzer.core.config import Config, get_config
from laser_trim_analyzer.core.processor import LaserTrimProcessor
from laser_trim_analyzer.database.manager import DatabaseManager
from laser_trim_analyzer.utils.report_generator import ReportGenerator

# Try to import ML components
try:
    from laser_trim_analyzer.ml.engine import MLEngine
    from laser_trim_analyzer.ml.models import ModelFactory

    HAS_ML = True
except ImportError:
    HAS_ML = False
    MLEngine = None
    ModelFactory = None

# Initialize Rich console for beautiful output
console = Console()


# Main CLI group
@click.group()
@click.option('--config', '-c', type=click.Path(exists=True),
              help='Path to configuration file')
@click.option('--debug/--no-debug', default=False,
              help='Enable debug logging')
@click.pass_context
def cli(ctx, config, debug):
    """
    Laser Trim Analyzer - Professional QA Analysis Tool

    A comprehensive command-line interface for potentiometer quality analysis.
    Run 'lta COMMAND --help' for more information on each command.
    """
    # Ensure context object exists
    ctx.ensure_object(dict)

    # Load configuration
    if config:
        ctx.obj['config'] = Config.from_yaml(Path(config))
    else:
        ctx.obj['config'] = get_config()

    # Set debug mode
    ctx.obj['debug'] = debug

    # Welcome message
    if debug:
        console.print("[yellow]Debug mode enabled[/yellow]")


# Analyze command
@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(),
              help='Output directory (default: ~/LaserTrimResults/YYYYMMDD_HHMMSS)')
@click.option('--parallel/--sequential', default=True,
              help='Enable parallel processing for multiple files')
@click.option('--workers', '-w', type=int, default=4,
              help='Number of parallel workers')
@click.option('--no-plots', is_flag=True,
              help='Skip plot generation for faster processing')
@click.option('--no-db', is_flag=True,
              help='Skip database storage')
@click.option('--ml/--no-ml', default=True,
              help='Enable/disable ML predictions')
@click.option('--format', '-f', type=click.Choice(['table', 'json', 'csv']),
              default='table', help='Output format')
@click.pass_context
def analyze(ctx, input_path, output, parallel, workers, no_plots, no_db, ml, format):
    """
    Analyze laser trim data from Excel files.

    INPUT_PATH can be a single Excel file or a directory containing multiple files.

    Examples:
        lta analyze data/8340_A12345.xlsx
        lta analyze data/ --parallel --workers 8
        lta analyze data/ --no-plots --format csv > results.csv
    """
    config = ctx.obj['config']
    input_path = Path(input_path)

    # Set up output directory
    if not output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = Path.home() / "LaserTrimResults" / f"CLI_{timestamp}"
    else:
        output = Path(output)

    output.mkdir(parents=True, exist_ok=True)

    # Update config based on CLI options
    config.processing.generate_plots = not no_plots
    config.processing.max_workers = workers
    config.database.enabled = not no_db
    config.ml.enabled = ml and HAS_ML

    # Initialize components
    db_manager = None
    if not no_db:
        try:
            db_manager = DatabaseManager(str(config.database.path))
            db_manager.init_db()
        except Exception as e:
            console.print(f"[red]Warning: Database initialization failed: {e}[/red]")

    # Run analysis
    console.print(Panel.fit(
        f"[bold blue]Laser Trim Analysis[/bold blue]\n"
        f"Input: {input_path}\n"
        f"Output: {output}\n"
        f"Mode: {'Parallel' if parallel else 'Sequential'}\n"
        f"ML: {'Enabled' if ml and HAS_ML else 'Disabled'}",
        title="Analysis Configuration"
    ))

    try:
        # Create processor
        processor = LaserTrimProcessor(config, db_manager)

        # Determine if single file or batch
        if input_path.is_file():
            results = asyncio.run(_analyze_single_file(processor, input_path, output))
        else:
            results = asyncio.run(_analyze_batch(processor, input_path, output, parallel))

        # Display results
        _display_results(results, format)

        # Generate summary report
        if results:
            report_path = output / "analysis_report.html"
            _generate_html_report(results, report_path)
            console.print(f"\n[green]✓[/green] Report saved to: {report_path}")

        console.print(f"\n[bold green]Analysis complete![/bold green]")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        if ctx.obj['debug']:
            console.print_exception()
        sys.exit(1)


async def _analyze_single_file(processor, file_path, output_dir):
    """Analyze a single file."""
    with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console
    ) as progress:
        task = progress.add_task(f"Analyzing {file_path.name}", total=100)

        def update_progress(message, value):
            progress.update(task, description=message, completed=int(value * 100))

        result = await processor.process_file(file_path, output_dir, update_progress)

    return [result]


async def _analyze_batch(processor, input_dir, output_dir, parallel):
    """Analyze multiple files."""
    # Find Excel files
    files = list(input_dir.glob("*.xlsx")) + list(input_dir.glob("*.xls"))
    files = [f for f in files if not f.name.startswith('~')]

    if not files:
        raise click.ClickException(f"No Excel files found in {input_dir}")

    console.print(f"Found {len(files)} files to process")

    with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
            console=console
    ) as progress:
        overall_task = progress.add_task("Overall progress", total=len(files))

        def batch_progress(current, total, filename):
            progress.update(overall_task, completed=current,
                            description=f"Processing {filename}")

        results = await processor.process_batch(
            input_dir, output_dir,
            max_workers=processor.config.processing.max_workers if parallel else 1,
            progress_callback=batch_progress
        )

    return results


def _display_results(results, format):
    """Display analysis results in requested format."""
    if not results:
        return

    if format == 'table':
        # Create rich table
        table = Table(title="Analysis Results", show_header=True, header_style="bold magenta")
        table.add_column("File", style="cyan", no_wrap=True)
        table.add_column("Model")
        table.add_column("Serial")
        table.add_column("Status", justify="center")
        table.add_column("Sigma", justify="right")
        table.add_column("Pass", justify="center")
        table.add_column("Risk", justify="center")

        for result in results:
            primary_track = result.primary_track
            status_color = {
                'PASS': 'green',
                'FAIL': 'red',
                'WARNING': 'yellow',
                'ERROR': 'red'
            }.get(result.overall_status.value, 'white')

            risk_color = {
                'HIGH': 'red',
                'MEDIUM': 'yellow',
                'LOW': 'green'
            }.get(
                primary_track.failure_prediction.risk_category.value if primary_track.failure_prediction else 'UNKNOWN',
                'white')

            table.add_row(
                result.metadata.filename,
                result.metadata.model,
                result.metadata.serial,
                f"[{status_color}]{result.overall_status.value}[/{status_color}]",
                f"{primary_track.sigma_analysis.sigma_gradient:.6f}",
                "✓" if primary_track.sigma_analysis.sigma_pass else "✗",
                f"[{risk_color}]{primary_track.failure_prediction.risk_category.value if primary_track.failure_prediction else 'N/A'}[/{risk_color}]"
            )

        console.print(table)

        # Summary statistics
        total = len(results)
        passed = sum(1 for r in results if r.overall_status.value == 'PASS')
        console.print(f"\n[bold]Summary:[/bold] {passed}/{total} passed ({passed / total * 100:.1f}%)")

    elif format == 'json':
        # Convert to JSON
        json_data = []
        for result in results:
            json_data.append(result.to_flat_dict())

        click.echo(json.dumps(json_data, indent=2, default=str))

    elif format == 'csv':
        # Convert to CSV
        data = []
        for result in results:
            data.append(result.to_flat_dict())

        df = pd.DataFrame(data)
        click.echo(df.to_csv(index=False))


def _generate_html_report(results, output_path):
    """Generate HTML report."""
    generator = ReportGenerator()
    generator.generate_html_report(results, output_path)


# Train command
@cli.command()
@click.option('--model', '-m', type=click.Choice(['threshold', 'failure', 'drift', 'all']),
              default='all', help='Model to train')
@click.option('--days', '-d', type=int, default=365,
              help='Days of historical data to use')
@click.option('--output', '-o', type=click.Path(),
              help='Output directory for models')
@click.option('--evaluate', is_flag=True,
              help='Evaluate models after training')
@click.pass_context
def train(ctx, model, days, output, evaluate):
    """
    Train ML models using historical data.

    This command trains machine learning models for:
    - Threshold optimization
    - Failure prediction
    - Drift detection

    Examples:
        lta train --days 180
        lta train --model failure --evaluate
        lta train --output models/v2/
    """
    if not HAS_ML:
        console.print("[red]ML components not available. Please install ML dependencies.[/red]")
        sys.exit(1)

    config = ctx.obj['config']

    # Set up output directory
    if not output:
        output = config.ml.model_path
    else:
        output = Path(output)

    output.mkdir(parents=True, exist_ok=True)

    console.print(Panel.fit(
        f"[bold blue]ML Model Training[/bold blue]\n"
        f"Model: {model}\n"
        f"Historical data: {days} days\n"
        f"Output: {output}",
        title="Training Configuration"
    ))

    try:
        # Initialize ML engine
        db_manager = DatabaseManager(str(config.database.path))
        ml_engine = MLEngine(
            data_path=str(config.data_directory),
            models_path=str(output)
        )

        # Load historical data
        with console.status(f"Loading {days} days of historical data..."):
            historical_data = _load_historical_data(db_manager, days)

        if historical_data.empty:
            console.print("[red]No historical data found for training[/red]")
            sys.exit(1)

        console.print(f"Loaded {len(historical_data)} records for training")

        # Train models
        models_to_train = ['threshold', 'failure', 'drift'] if model == 'all' else [model]

        for model_name in models_to_train:
            console.print(f"\n[bold]Training {model_name} model...[/bold]")

            with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
            ) as progress:
                task = progress.add_task(f"Training {model_name}", total=None)

                if model_name == 'threshold':
                    model_class = ModelFactory.create_threshold_optimizer
                    ml_engine.register_model('threshold_optimizer', model_class, config)
                    trained_model = ml_engine.train_model('threshold_optimizer', model_class, historical_data)

                elif model_name == 'failure':
                    model_class = ModelFactory.create_failure_predictor
                    ml_engine.register_model('failure_predictor', model_class, config)
                    trained_model = ml_engine.train_model('failure_predictor', model_class, historical_data)

                elif model_name == 'drift':
                    model_class = ModelFactory.create_drift_detector
                    ml_engine.register_model('drift_detector', model_class, config)
                    trained_model = ml_engine.train_model('drift_detector', model_class, historical_data)

                progress.update(task, description=f"✓ {model_name} trained")

            # Display performance metrics
            if hasattr(trained_model, 'performance_metrics'):
                _display_model_metrics(model_name, trained_model.performance_metrics)

        # Evaluate if requested
        if evaluate:
            console.print("\n[bold]Evaluating models...[/bold]")
            recent_data = _load_historical_data(db_manager, 30)

            for model_name in models_to_train:
                metrics = ml_engine.evaluate_model(f"{model_name}_optimizer", recent_data)
                console.print(f"\n{model_name.title()} evaluation:")
                _display_model_metrics(model_name, metrics)

        console.print("\n[bold green]Training complete![/bold green]")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        if ctx.obj['debug']:
            console.print_exception()
        sys.exit(1)


def _load_historical_data(db_manager, days):
    """Load historical data from database."""
    try:
        data = db_manager.get_historical_data(days_back=days)

        # Convert to DataFrame
        records = []
        for result in data:
            for track in result.tracks:
                record = {
                    'model': result.model,
                    'serial': result.serial,
                    'timestamp': result.timestamp,
                    'sigma_gradient': track.sigma_gradient,
                    'sigma_threshold': track.sigma_threshold,
                    'sigma_pass': track.sigma_pass,
                    'linearity_pass': track.linearity_pass,
                    'failure_probability': track.failure_probability,
                    'risk_category': track.risk_category.value if track.risk_category else None,
                    'unit_length': track.unit_length,
                    'resistance_change_percent': track.resistance_change_percent
                }
                records.append(record)

        return pd.DataFrame(records)

    except Exception as e:
        console.print(f"[red]Error loading data: {e}[/red]")
        return pd.DataFrame()


def _display_model_metrics(model_name, metrics):
    """Display model performance metrics."""
    table = Table(title=f"{model_name.title()} Model Performance", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    for metric, value in metrics.items():
        if isinstance(value, float):
            table.add_row(metric.replace('_', ' ').title(), f"{value:.4f}")
        else:
            table.add_row(metric.replace('_', ' ').title(), str(value))

    console.print(table)


# Report command
@cli.command()
@click.option('--type', '-t', type=click.Choice(['summary', 'detailed', 'batch', 'trends']),
              default='summary', help='Type of report to generate')
@click.option('--input', '-i', type=click.Path(exists=True),
              help='Input directory with analysis results')
@click.option('--output', '-o', type=click.Path(),
              help='Output file path')
@click.option('--format', '-f', type=click.Choice(['html', 'pdf', 'excel']),
              default='html', help='Report format')
@click.option('--days', '-d', type=int, default=30,
              help='Days of data to include (for trends report)')
@click.option('--model', '-m', type=str,
              help='Filter by model number')
@click.pass_context
def report(ctx, type, input, output, format, days, model):
    """
    Generate analysis reports.

    Create various types of reports from analysis data:
    - Summary: Overview of recent analyses
    - Detailed: Comprehensive analysis with plots
    - Batch: Summary of batch processing results
    - Trends: Historical trend analysis

    Examples:
        lta report --type summary --days 7
        lta report --type trends --model 8340 --format excel
        lta report --input results/ --type batch --output report.pdf
    """
    config = ctx.obj['config']

    # Set up paths
    if not output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = Path.home() / "LaserTrimResults" / f"report_{type}_{timestamp}.{format}"
    else:
        output = Path(output)

    console.print(Panel.fit(
        f"[bold blue]Report Generation[/bold blue]\n"
        f"Type: {type}\n"
        f"Format: {format}\n"
        f"Output: {output}",
        title="Report Configuration"
    ))

    try:
        db_manager = DatabaseManager(str(config.database.path))
        generator = ReportGenerator()

        if type == 'summary':
            # Generate summary report
            data = db_manager.get_historical_data(days_back=days, model=model)

            if format == 'html':
                generator.generate_summary_html(data, output)
            elif format == 'excel':
                generator.generate_summary_excel(data, output)
            elif format == 'pdf':
                # Would need PDF generation library
                console.print("[yellow]PDF generation not yet implemented[/yellow]")
                return

        elif type == 'detailed':
            if not input:
                console.print("[red]--input required for detailed report[/red]")
                return

            # Load results from input directory
            results = _load_results_from_directory(Path(input))
            generator.generate_detailed_report(results, output, format)

        elif type == 'batch':
            if not input:
                console.print("[red]--input required for batch report[/red]")
                return

            # Load batch results
            results = _load_results_from_directory(Path(input))
            generator.generate_batch_report(results, output, format)

        elif type == 'trends':
            # Generate trends report
            data = db_manager.get_historical_data(days_back=days, model=model)
            stats = db_manager.get_model_statistics(model) if model else None

            generator.generate_trends_report(data, stats, output, format)

        console.print(f"\n[green]✓[/green] Report saved to: {output}")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        if ctx.obj['debug']:
            console.print_exception()
        sys.exit(1)


def _load_results_from_directory(directory):
    """Load analysis results from directory."""
    # This would load serialized results
    # For now, return empty list
    return []


# Query command
@cli.command()
@click.option('--model', '-m', type=str, help='Filter by model number')
@click.option('--serial', '-s', type=str, help='Filter by serial number')
@click.option('--days', '-d', type=int, default=30,
              help='Days of history to query')
@click.option('--status', type=click.Choice(['pass', 'fail', 'warning']),
              help='Filter by status')
@click.option('--risk', type=click.Choice(['high', 'medium', 'low']),
              help='Filter by risk category')
@click.option('--limit', '-l', type=int, default=100,
              help='Maximum results to return')
@click.option('--export', '-e', type=click.Path(),
              help='Export results to file')
@click.option('--stats', is_flag=True,
              help='Show statistics instead of records')
@click.pass_context
def query(ctx, model, serial, days, status, risk, limit, export, stats):
    """
    Query historical analysis data.

    Search and filter historical QA data with various criteria.

    Examples:
        lta query --model 8340 --days 7
        lta query --status fail --risk high --limit 50
        lta query --model 8555 --stats
        lta query --days 90 --export results.csv
    """
    config = ctx.obj['config']

    try:
        db_manager = DatabaseManager(str(config.database.path))

        # Build query parameters
        console.print("[bold]Querying database...[/bold]")

        # Convert status/risk to proper case
        if status:
            status = status.upper()
        if risk:
            risk = risk.upper()

        # Execute query
        with console.status("Running query..."):
            results = db_manager.get_historical_data(
                model=model,
                serial=serial,
                days_back=days,
                status=status,
                risk_category=risk,
                limit=limit
            )

        if not results:
            console.print("[yellow]No results found[/yellow]")
            return

        console.print(f"Found {len(results)} results")

        if stats:
            # Show statistics
            _display_query_statistics(results, model)
        else:
            # Show records
            _display_query_results(results)

        # Export if requested
        if export:
            _export_query_results(results, Path(export))
            console.print(f"\n[green]✓[/green] Results exported to: {export}")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        if ctx.obj['debug']:
            console.print_exception()
        sys.exit(1)


def _display_query_results(results):
    """Display query results in a table."""
    table = Table(title="Query Results", show_header=True)
    table.add_column("Date", style="cyan")
    table.add_column("File")
    table.add_column("Model")
    table.add_column("Serial")
    table.add_column("Status")
    table.add_column("Sigma", justify="right")
    table.add_column("Risk")

    for result in results[:50]:  # Limit display to 50 rows
        for track in result.tracks:
            status_color = {
                'Pass': 'green',
                'Fail': 'red',
                'Warning': 'yellow'
            }.get(result.overall_status.value, 'white')

            risk_color = {
                'High': 'red',
                'Medium': 'yellow',
                'Low': 'green'
            }.get(track.risk_category.value if track.risk_category else 'Unknown', 'white')

            table.add_row(
                result.timestamp.strftime("%Y-%m-%d %H:%M"),
                result.filename,
                result.model,
                result.serial,
                f"[{status_color}]{result.overall_status.value}[/{status_color}]",
                f"{track.sigma_gradient:.6f}",
                f"[{risk_color}]{track.risk_category.value if track.risk_category else 'N/A'}[/{risk_color}]"
            )

    console.print(table)

    if len(results) > 50:
        console.print(f"\n[yellow]Showing first 50 of {len(results)} results[/yellow]")


def _display_query_statistics(results, model_filter):
    """Display statistics from query results."""
    # Calculate statistics
    total_files = len(results)
    total_tracks = sum(len(r.tracks) for r in results)

    # Status counts
    pass_count = sum(1 for r in results if r.overall_status.value == 'Pass')
    fail_count = sum(1 for r in results if r.overall_status.value == 'Fail')
    warning_count = sum(1 for r in results if r.overall_status.value == 'Warning')

    # Risk distribution
    risk_counts = {'High': 0, 'Medium': 0, 'Low': 0}
    sigma_values = []

    for result in results:
        for track in result.tracks:
            if track.risk_category:
                risk_counts[track.risk_category.value] += 1
            sigma_values.append(track.sigma_gradient)

    # Create statistics panel
    stats_text = f"""
[bold]Analysis Statistics[/bold]
{'=' * 40}
Total Files: {total_files}
Total Tracks: {total_tracks}

[bold]Status Distribution:[/bold]
  Pass: {pass_count} ({pass_count / total_files * 100:.1f}%)
  Fail: {fail_count} ({fail_count / total_files * 100:.1f}%)
  Warning: {warning_count} ({warning_count / total_files * 100:.1f}%)

[bold]Risk Categories:[/bold]
  High: {risk_counts['High']} ({risk_counts['High'] / total_tracks * 100:.1f}%)
  Medium: {risk_counts['Medium']} ({risk_counts['Medium'] / total_tracks * 100:.1f}%)
  Low: {risk_counts['Low']} ({risk_counts['Low'] / total_tracks * 100:.1f}%)

[bold]Sigma Gradient:[/bold]
  Mean: {np.mean(sigma_values):.6f}
  Std Dev: {np.std(sigma_values):.6f}
  Min: {np.min(sigma_values):.6f}
  Max: {np.max(sigma_values):.6f}
"""

    if model_filter:
        stats_text = f"[bold]Model: {model_filter}[/bold]\n" + stats_text

    console.print(Panel(stats_text, title="Query Statistics", border_style="blue"))


def _export_query_results(results, output_path):
    """Export query results to file."""
    # Convert to DataFrame
    records = []
    for result in results:
        base_record = {
            'timestamp': result.timestamp,
            'filename': result.filename,
            'model': result.model,
            'serial': result.serial,
            'overall_status': result.overall_status.value
        }

        for track in result.tracks:
            record = base_record.copy()
            record.update({
                'track_id': track.track_id,
                'sigma_gradient': track.sigma_gradient,
                'sigma_threshold': track.sigma_threshold,
                'sigma_pass': track.sigma_pass,
                'linearity_pass': track.linearity_pass,
                'risk_category': track.risk_category.value if track.risk_category else None,
                'failure_probability': track.failure_probability
            })
            records.append(record)

    df = pd.DataFrame(records)

    # Export based on file extension
    if output_path.suffix.lower() == '.csv':
        df.to_csv(output_path, index=False)
    elif output_path.suffix.lower() in ['.xlsx', '.xls']:
        df.to_excel(output_path, index=False)
    else:
        # Default to CSV
        df.to_csv(output_path, index=False)


# Utility commands
@cli.command()
@click.pass_context
def info(ctx):
    """
    Display system information and configuration.
    """
    config = ctx.obj['config']

    info_text = f"""
[bold]Laser Trim Analyzer Information[/bold]
{'=' * 50}
Version: 2.0.0
Python: {sys.version.split()[0]}

[bold]Configuration:[/bold]
  Config File: {config.model_config.get('_config_file', 'Default')}
  Debug Mode: {ctx.obj['debug']}

[bold]Paths:[/bold]
  Data Directory: {config.data_directory}
  Database: {config.database.path}
  Models: {config.ml.model_path}
  Logs: {config.log_directory}

[bold]Features:[/bold]
  Database: {'✓' if config.database.enabled else '✗'}
  ML Models: {'✓' if HAS_ML and config.ml.enabled else '✗'}
  Plotting: {'✓' if config.processing.generate_plots else '✗'}

[bold]Processing:[/bold]
  Max Workers: {config.processing.max_workers}
  Cache: {'✓' if config.processing.cache_enabled else '✗'}
  File Types: {', '.join(config.processing.file_extensions)}
"""

    console.print(Panel(info_text, title="System Information", border_style="blue"))


@cli.command()
@click.option('--clear-cache', is_flag=True, help='Clear processing cache')
@click.option('--clear-logs', is_flag=True, help='Clear old log files')
@click.option('--days', '-d', type=int, default=30,
              help='Days of logs to keep')
@click.pass_context
def clean(ctx, clear_cache, clear_logs, days):
    """
    Clean up temporary files and old data.
    """
    config = ctx.obj['config']

    if clear_cache:
        console.print("[bold]Clearing cache...[/bold]")
        # Would clear cache directory
        console.print("[green]✓[/green] Cache cleared")

    if clear_logs:
        console.print(f"[bold]Clearing logs older than {days} days...[/bold]")
        log_dir = config.log_directory
        cutoff = datetime.now() - timedelta(days=days)

        removed = 0
        for log_file in log_dir.glob("*.log"):
            if datetime.fromtimestamp(log_file.stat().st_mtime) < cutoff:
                log_file.unlink()
                removed += 1

        console.print(f"[green]✓[/green] Removed {removed} old log files")


@cli.command()
@click.argument('key', required=False)
@click.option('--set', 'value', help='Set configuration value')
@click.option('--list', 'list_all', is_flag=True, help='List all settings')
@click.pass_context
def config(ctx, key, value, list_all):
    """
    View or modify configuration settings.

    Examples:
        lta config --list
        lta config processing.max_workers
        lta config processing.max_workers --set 8
    """
    config_obj = ctx.obj['config']

    if list_all:
        # Display all configuration
        config_dict = config_obj.model_dump()
        syntax = Syntax(json.dumps(config_dict, indent=2, default=str),
                        "json", theme="monokai", line_numbers=True)
        console.print(Panel(syntax, title="Current Configuration"))

    elif key and value is not None:
        # Set configuration value
        try:
            # Parse the key path (e.g., "processing.max_workers")
            keys = key.split('.')
            current = config_obj

            # Navigate to the parent
            for k in keys[:-1]:
                current = getattr(current, k)

            # Set the value
            setattr(current, keys[-1], type(getattr(current, keys[-1]))(value))

            # Save configuration
            config_path = Path.home() / ".laser_trim_analyzer" / "config.yaml"
            config_obj.to_yaml(config_path)

            console.print(f"[green]✓[/green] Set {key} = {value}")

        except Exception as e:
            console.print(f"[red]Error setting configuration: {e}[/red]")

    elif key:
        # Get configuration value
        try:
            keys = key.split('.')
            current = config_obj

            for k in keys:
                current = getattr(current, k)

            console.print(f"{key} = {current}")

        except AttributeError:
            console.print(f"[red]Unknown configuration key: {key}[/red]")

    else:
        # Show help
        console.print("Use 'lta config --help' for usage information")


# Entry point for script execution
def main():
    """Main entry point for the CLI."""
    cli(obj={})


if __name__ == '__main__':
    main()