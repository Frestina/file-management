# File Renamer

A Python library for renaming files with timestamps and class names.

## Installation

```bash
uv add file-renamer
```

## Usage

```python
from file_renamer import FileRenamer, get_creation_time

# Get file creation time
timestamp = get_creation_time("path/to/file.jpg")

# Generate new filename
renamer = FileRenamer()
new_name = renamer.generate_filename("path/to/file.jpg", "cat", timestamp)
print(new_name)  # cat_2024-01-15_14-30-52.jpg

# Resolve collisions before writing
dest = renamer.get_unique_filename("output/cat", new_name)
print(dest)  # output/cat/cat_2024-01-15_14-30-52_001.jpg
```

The extension is taken from the path passed to `generate_filename`, so pass the
file that will actually be written — a `.png` compressed to `.jpg` must be named
from the `.jpg`.

Pass `FileRenamer(timestamp_format=...)` to change the timestamp layout. The
default includes seconds, since camera traps fire in bursts and would otherwise
collide within the same minute.

## Bulk renaming

```python
from file_renamer import rename_files

# Renames to: folderName_creationTime.extension, recursively
rename_files("path/to/folder")
```
