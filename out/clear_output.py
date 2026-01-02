#!/usr/bin/env python3
"""
Clear all generated output files while preserving test fixtures and documentation.

Deletes everything in out/ that is git-ignored, keeping:
- */test/ folders and their contents
- README.md
- This script (clear_output.py)

Usage:
    python out/clear_output.py            # Delete files (requires --force if >10MB)
    python out/clear_output.py --dry-run  # Preview what would be deleted
    python out/clear_output.py --force    # Delete files without confirmation
"""

# Threshold in bytes above which --force is required (10 MB)
FORCE_THRESHOLD_BYTES = 10 * 1024 * 1024

import argparse
import shutil
import subprocess
from pathlib import Path

OUT_DIR = Path(__file__).parent

# Files/folders to always keep (relative to out/)
KEEP_PATTERNS = {
    "clear_output.py",
    "README.md",
}


def is_in_test_folder(path: Path) -> bool:
    """Check if path is inside a /test/ folder."""
    parts = path.relative_to(OUT_DIR).parts
    return "test" in parts


def is_protected(path: Path) -> bool:
    """Check if path should be protected from deletion."""
    rel_path = path.relative_to(OUT_DIR)

    # Check if it's a keep pattern
    if rel_path.name in KEEP_PATTERNS:
        return True

    # Check if it's inside a test folder
    if is_in_test_folder(path):
        return True

    return False


def get_files_to_delete() -> list[Path]:
    """Get list of files/folders to delete."""
    to_delete = []

    # Walk through out/ directory
    for item in OUT_DIR.iterdir():
        if item.name.startswith("."):
            continue

        if is_protected(item):
            continue

        if item.is_dir():
            # For directories, check if they contain only deletable content
            has_protected = False
            for sub in item.rglob("*"):
                if is_protected(sub):
                    has_protected = True
                    break

            if has_protected:
                # Directory has protected content, find deletable files within
                for sub in item.rglob("*"):
                    if sub.is_file() and not is_protected(sub):
                        to_delete.append(sub)
            else:
                # Entire directory can be deleted
                to_delete.append(item)
        else:
            to_delete.append(item)

    return sorted(to_delete, key=lambda p: str(p))


def get_empty_dirs_after_deletion(deleted: list[Path]) -> list[Path]:
    """Find directories that would be empty after deletion."""
    empty_dirs = []

    # Get all parent directories of deleted files
    parent_dirs = set()
    for path in deleted:
        if path.is_file():
            parent_dirs.add(path.parent)

    # Check each parent directory
    for dir_path in parent_dirs:
        if dir_path == OUT_DIR:
            continue

        # Check if directory would be empty after deletion
        remaining = []
        for item in dir_path.iterdir():
            if item not in deleted and not any(item == d or d in item.parents for d in deleted):
                remaining.append(item)

        if not remaining:
            empty_dirs.append(dir_path)

    return sorted(empty_dirs, key=lambda p: len(str(p)), reverse=True)


def format_size(size_bytes: int) -> str:
    """Format byte size as human-readable string."""
    if size_bytes >= 1024 * 1024 * 1024:
        return f"{size_bytes / 1024 / 1024 / 1024:.2f} GB"
    elif size_bytes >= 1024 * 1024:
        return f"{size_bytes / 1024 / 1024:.2f} MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes} bytes"


def calculate_total_size(paths: list[Path]) -> int:
    """Calculate total size of files to be deleted."""
    total_size = 0
    for path in paths:
        if path.is_file():
            total_size += path.stat().st_size
        elif path.is_dir():
            for f in path.rglob("*"):
                if f.is_file():
                    total_size += f.stat().st_size
    return total_size


def main():
    parser = argparse.ArgumentParser(
        description="Clear generated output files, keeping test fixtures."
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Preview what would be deleted without actually deleting"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force deletion without confirmation (required for >10MB)"
    )
    args = parser.parse_args()

    to_delete = get_files_to_delete()

    if not to_delete:
        print("Nothing to delete. Output directory is clean.")
        return

    # Calculate total size
    total_size = calculate_total_size(to_delete)
    size_str = format_size(total_size)

    if args.dry_run:
        print(f"Would delete {len(to_delete)} items ({size_str}):")
        print()
        for path in to_delete:
            if path.is_dir():
                file_count = sum(1 for _ in path.rglob("*") if _.is_file())
                print(f"  [DIR]  {path.relative_to(OUT_DIR)}/ ({file_count} files)")
            else:
                print(f"  [FILE] {path.relative_to(OUT_DIR)}")
        print()
        if total_size > FORCE_THRESHOLD_BYTES:
            print(f"⚠️  Large deletion ({size_str}). Use --force to confirm deletion.")
        else:
            print("Run without --dry-run to actually delete these files.")
    else:
        # Check if --force is required for large deletions
        if total_size > FORCE_THRESHOLD_BYTES and not args.force:
            print(f"⚠️  About to delete {len(to_delete)} items ({size_str}).")
            print()
            print(f"This exceeds the safety threshold of {format_size(FORCE_THRESHOLD_BYTES)}.")
            print("Use --force (or -f) to confirm this deletion, or --dry-run to preview.")
            return

        print(f"Deleting {len(to_delete)} items ({size_str})...")
        for path in to_delete:
            if path.is_dir():
                shutil.rmtree(path)
                print(f"  Deleted directory: {path.relative_to(OUT_DIR)}")
            else:
                path.unlink()
                print(f"  Deleted file: {path.relative_to(OUT_DIR)}")

        print(f"\nDone. Freed {size_str}.")


if __name__ == "__main__":
    main()
