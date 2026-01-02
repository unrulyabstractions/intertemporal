"""Camera-ready figure export functionality."""

from __future__ import annotations

import shutil
from pathlib import Path


def export_camera_ready(
    viz_dir: Path,
    output_name: str = "behavior",
    paper_dir: Path | None = None,
) -> Path:
    """
    Export ALL plots to output folder for publication.

    Copies all PNG files from viz/ subdirectories to the output folder,
    flattening the directory structure.

    Args:
        viz_dir: Directory containing visualization subdirectories
        output_name: Name for output folder (default: "behavior")
        paper_dir: Optional path to paper directory for additional copy

    Returns:
        Path to the output directory
    """
    output_dir = viz_dir.parent / output_name

    # Clear any existing files
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 80)
    print("  CAMERA-READY EXPORT")
    print("=" * 80)
    print(f"  Output: {output_dir}")
    print()

    exported = 0

    # Find all PNG files in viz/ and copy to output folder
    for png_file in sorted(viz_dir.rglob("*.png")):
        # Create flat filename from path: conflicts/clusters/foo.png -> conflicts_clusters_foo.png
        rel_path = png_file.relative_to(viz_dir)
        flat_name = str(rel_path).replace("/", "_")
        dest_path = output_dir / flat_name
        shutil.copy2(png_file, dest_path)
        print(f"  {flat_name}")
        exported += 1

    print()
    print(f"  Exported {exported} camera-ready figures to: {output_dir}")

    # Also copy to paper directory if specified
    if paper_dir is not None:
        paper_output = paper_dir / output_name
        if paper_output.exists():
            shutil.rmtree(paper_output)
        shutil.copytree(output_dir, paper_output)
        print(f"  Copied to paper directory: {paper_output}")

    print("=" * 80)

    return output_dir
