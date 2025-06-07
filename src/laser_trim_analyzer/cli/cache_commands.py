"""
CLI commands for cache management.

This module provides command-line interface for managing the application's cache.
"""

import click
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from ..core.cache_manager import get_cache_manager
from ..core.cache_config import (
    setup_cache_from_preset, CachePreset, get_cache_info,
    optimize_cache, clear_old_cache_files
)

console = Console()


@click.group(name='cache')
def cache_cli():
    """Cache management commands."""
    pass


@cache_cli.command()
def stats():
    """Display cache statistics."""
    try:
        info = get_cache_info()
        
        # Create stats table
        table = Table(
            title="Cache Statistics",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta"
        )
        
        table.add_column("Cache Type", style="cyan", no_wrap=True)
        table.add_column("Type", style="yellow")
        table.add_column("Entries", justify="right", style="green")
        table.add_column("Size (MB)", justify="right", style="blue")
        table.add_column("Strategy", style="magenta")
        
        for cache_name, cache_stats in info['stats'].items():
            if isinstance(cache_stats, dict) and 'type' in cache_stats:
                table.add_row(
                    cache_name,
                    cache_stats.get('type', 'Unknown'),
                    str(cache_stats.get('entries', 0)),
                    f"{cache_stats.get('size_mb', 0):.2f}",
                    cache_stats.get('strategy', 'N/A')
                )
        
        console.print(table)
        
        # System memory info
        if 'system_memory' in info['stats']:
            mem = info['stats']['system_memory']
            console.print(
                Panel(
                    f"[bold]System Memory[/bold]\n"
                    f"Total: {mem['total_mb']:.0f} MB\n"
                    f"Available: {mem['available_mb']:.0f} MB\n"
                    f"Usage: {mem['percent_used']:.1f}%\n"
                    f"Threshold: {mem['threshold']:.0f}%",
                    title="Memory Status",
                    border_style="yellow"
                )
            )
        
        # Total cache usage
        console.print(
            f"\n[bold green]Total cache memory usage:[/bold green] "
            f"{info['total_memory_usage_mb']:.2f} MB"
        )
        
    except Exception as e:
        console.print(f"[red]Error getting cache stats: {e}[/red]")


@cache_cli.command()
@click.option(
    '--type',
    'cache_type',
    type=click.Choice(['all', 'general', 'file_content', 'model_predictions', 'config', 'analysis_results']),
    default='all',
    help='Cache type to clear'
)
@click.confirmation_option(prompt='Are you sure you want to clear the cache?')
def clear(cache_type: str):
    """Clear cache data."""
    try:
        manager = get_cache_manager()
        
        if cache_type == 'all':
            manager.clear()
            console.print("[green]✓ All caches cleared successfully[/green]")
        else:
            manager.clear(cache_type)
            console.print(f"[green]✓ {cache_type} cache cleared successfully[/green]")
            
    except Exception as e:
        console.print(f"[red]Error clearing cache: {e}[/red]")


@cache_cli.command()
@click.argument('pattern')
@click.option(
    '--type',
    'cache_type',
    type=click.Choice(['general', 'file_content', 'model_predictions', 'config', 'analysis_results']),
    default='general',
    help='Cache type to search'
)
def invalidate(pattern: str, cache_type: str):
    """Invalidate cache entries matching a pattern."""
    try:
        manager = get_cache_manager()
        count = manager.invalidate_pattern(pattern, cache_type)
        
        console.print(
            f"[green]✓ Invalidated {count} entries matching '{pattern}' "
            f"in {cache_type} cache[/green]"
        )
        
    except Exception as e:
        console.print(f"[red]Error invalidating cache: {e}[/red]")


@cache_cli.command()
def optimize():
    """Optimize cache by clearing expired entries."""
    try:
        console.print("[yellow]Optimizing cache...[/yellow]")
        
        stats = optimize_cache()
        
        console.print(
            Panel(
                f"[bold]Optimization Results[/bold]\n"
                f"Expired entries cleared: {stats['expired_cleared']}\n"
                f"Old files cleared: {stats['old_files_cleared']}\n"
                f"Memory freed: {stats['memory_freed_mb']:.2f} MB",
                title="Cache Optimization",
                border_style="green"
            )
        )
        
    except Exception as e:
        console.print(f"[red]Error optimizing cache: {e}[/red]")


@cache_cli.command()
@click.option(
    '--days',
    default=7,
    help='Clear files older than this many days'
)
def clean_old(days: int):
    """Clear old cache files."""
    try:
        console.print(f"[yellow]Clearing cache files older than {days} days...[/yellow]")
        
        cleared = clear_old_cache_files(days)
        
        console.print(f"[green]✓ Cleared {cleared} old cache files[/green]")
        
    except Exception as e:
        console.print(f"[red]Error clearing old files: {e}[/red]")


@cache_cli.command()
@click.option(
    '--preset',
    type=click.Choice([p.value for p in CachePreset]),
    required=True,
    help='Cache configuration preset'
)
def setup(preset: str):
    """Setup cache with a predefined configuration."""
    try:
        preset_enum = CachePreset(preset)
        setup_cache_from_preset(preset_enum)
        
        console.print(
            f"[green]✓ Cache configured with '{preset}' preset[/green]\n"
            f"[dim]Run 'cache stats' to view current configuration[/dim]"
        )
        
    except Exception as e:
        console.print(f"[red]Error setting up cache: {e}[/red]")


@cache_cli.command()
@click.argument('file_path', type=click.Path(exists=True))
def clear_file(file_path: str):
    """Clear cache for a specific file."""
    try:
        from ..core.cached_processor import CachedFileProcessor
        from ..core.config import Config
        
        # Create temporary processor to access cache clearing
        config = Config()
        processor = CachedFileProcessor(config, Path.cwd())
        
        file = Path(file_path)
        processor.clear_file_cache(file)
        
        console.print(f"[green]✓ Cleared cache for {file.name}[/green]")
        
    except Exception as e:
        console.print(f"[red]Error clearing file cache: {e}[/red]")


@cache_cli.command()
def info():
    """Display detailed cache information."""
    try:
        manager = get_cache_manager()
        info = get_cache_info()
        
        # Cache directory info
        console.print(
            Panel(
                f"[bold]Cache Directory:[/bold] {manager.cache_dir}\n"
                f"[bold]Cache Types:[/bold] {', '.join(info['cache_types'])}\n"
                f"[bold]Total Memory Usage:[/bold] {info['total_memory_usage_mb']:.2f} MB",
                title="Cache Configuration",
                border_style="blue"
            )
        )
        
        # Detailed stats for each cache
        for cache_name, cache_stats in info['stats'].items():
            if isinstance(cache_stats, dict) and 'type' in cache_stats:
                details = []
                for key, value in cache_stats.items():
                    if key not in ['type']:
                        details.append(f"{key}: {value}")
                
                console.print(
                    Panel(
                        '\n'.join(details),
                        title=f"{cache_name} Cache",
                        border_style="cyan"
                    )
                )
        
    except Exception as e:
        console.print(f"[red]Error getting cache info: {e}[/red]")


# Add cache commands to main CLI
def register_cache_commands(cli_group):
    """Register cache commands with the main CLI."""
    cli_group.add_command(cache_cli)