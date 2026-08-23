"""Display utilities and formatting."""

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from typing import Dict


def create_progress(console: Console) -> Progress:
    """Create a standard rich Progress bar for file processing."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        console=console,
    )


def print_arguments(args, console: Console):
    """Display the current configuration arguments."""
    console.rule("[bold cyan]Running with these parameters:\n")
    console.print(f"[bold]Input directory:[/bold] {args.input_dir}")
    console.print(f"[bold]Output directory:[/bold] {args.output_dir}")
    console.print(f"[bold]Compression:[/bold] {'Enabled' if args.compress_files else 'Disabled'}")
    console.print(f"[bold]Model:[/bold] {args.model}\n")


def sorting_summary(
    console: Console,
    video_count: int,
    image_count: int,
    corrupted_count: int,
    processing_error_count: int,
    class_counts: Dict[str, int],
) -> None:
    """Display final sorting summary table."""
    table = Table(title="Sorting Summary")
    table.add_column("Category", style="cyan", no_wrap=True)
    table.add_column("Count", style="magenta")

    table.add_row("Videos", str(video_count))
    table.add_row("Images", str(image_count))
    table.add_section()
    table.add_row("Corrupted Files", str(corrupted_count))
    table.add_row("Processing Errors", str(processing_error_count))
    table.add_section()

    for class_name, count in class_counts.items():
        table.add_row(class_name, str(count))

    console.print(table)


def visualize_summary(
    console: Console,
    class_counts: Dict[str, int],
    output_path: str,
    class_name: str
) -> None:
    """Display visualization summary."""
    summary = (
        f"[bold green]Image saved to:[/bold green] {output_path}\n"
        f"[bold cyan]Detected class:[/bold cyan] {class_name} "
        f"(count: {class_counts[class_name]})"
    )
    console.print(Panel(summary, title="Visualize Summary", expand=False))