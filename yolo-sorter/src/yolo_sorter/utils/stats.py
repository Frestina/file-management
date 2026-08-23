"""Statistics and reporting utilities."""

from rich.console import Console
from .file_utils import get_total_size_recursive


def calculate_compression_stats(
    original_size_gib: float, 
    output_dir: str, 
    console: Console
) -> None:
    """
    Calculate and display compression statistics.
    
    Args:
        original_size_gib: Original total size in GiB
        output_dir: Output directory to calculate final size
        console: Rich console for display
    """
    total_bytes = get_total_size_recursive(output_dir)
    total_gib_output = total_bytes / (1024**3)
    
    if original_size_gib > 0:
        reduction_percentage = ((original_size_gib - total_gib_output) / original_size_gib) * 100
        console.print(
            f'Size reduction: {original_size_gib:.2f} GiB -> '
            f'{total_gib_output:.2f} GiB, '
            f'({reduction_percentage:.1f}% reduction)'
        )
    else:
        console.print(
            f'Size before: {original_size_gib:.2f} GiB -> '
            f'after: {total_gib_output:.2f} GiB'
        )


def log_file_compression_result(
    file_name: str, 
    src_path: str, 
    compressed_path: str, 
    console: Console
) -> None:
    """
    Log compression results with size information.
    
    Args:
        file_name: Name of the file being compressed
        src_path: Path to original file
        compressed_path: Path to compressed file
        console: Rich console for display
    """
    import os
    
    original_size = os.path.getsize(src_path) / (1024 * 1024)  # MB
    compressed_size = os.path.getsize(compressed_path) / (1024 * 1024)  # MB
    
    if original_size > 0:
        reduction = ((original_size - compressed_size) / original_size) * 100
        console.print(
            f"[cyan]Compressed:[/cyan] {file_name} "
            f"({original_size:.1f}MB → {compressed_size:.1f}MB, {reduction:.1f}% reduction)"
        )
    else:
        console.print(f"[cyan]Compressed:[/cyan] {file_name}")