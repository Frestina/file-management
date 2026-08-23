from datetime import datetime
from pathlib import Path

from .timestamp_utils import get_creation_time, format_timestamp


def rename_files(folder_path: str):
  """
  Rename files in a folder to: folderName_creationTime.extension
  For subfolders: subfolderName_creationTime.extension
  
  Args:
      folder_path (str): Path to the folder containing files to rename
      include_subfolders (bool): Whether to process subfolders recursively
  """

  folder = Path(folder_path)

  if not folder.exists() or not folder.is_dir():
    print(f"Error: '{folder_path}' is not a valid directory")
    return

  allowed_extensions = {'.jpg', '.jpeg', '.png', '.avi'}

  renamed_count = 0
  skipped_count = 0

  all_files = folder.rglob('*')

  files = [f for f in all_files if f.is_file() and f.suffix.lower() in allowed_extensions]

  if not files:
    print("No files found in the directory or dub directories")
    return
  
  all_files_count = len([f for f in folder.glob('*')])
  skipped_count = all_files_count - len(files)

  for file_path in files:
    try:
      creation_time = get_creation_time(file_path)
      time_string = format_timestamp(creation_time)
      extension = file_path.suffix

      if file_path.parent == folder:
        prefix = folder.name
      else:
        prefix = file_path.parent.name

      new_name = f"{prefix}_{time_string}{extension}"
      new_path = file_path.parent / new_name

      if new_path.exists():
        counter = 1
        base_new_name = f"{prefix}_{time_string}"
        while new_path.exists():
          new_name = f"{base_new_name}_{counter}{extension}"
          new_path = file_path.parent / new_name
          counter += 1

      file_path.rename(new_path)
      renamed_count += 1

    except Exception as e:
      print(f"Error renaming {file_path.name}: {e}")
  
  print(f"\nCompleted! Renamed {renamed_count} files.")
  if skipped_count > 0:
    print(f"Skipped {skipped_count} files (not image/video files).")
    


class FileRenamer:
  """Builds timestamped, class-prefixed filenames for sorted files."""

  def __init__(self, timestamp_format: str = "%Y-%m-%d_%H-%M-%S"):
    """
    Args:
        timestamp_format: strftime format used for the timestamp part.
            Seconds are included by default because camera traps fire in
            bursts and would otherwise collide within the same minute.
    """
    self.timestamp_format = timestamp_format

  def generate_filename(
    self,
    file_path: str,
    class_name: str,
    timestamp: float = None
  ) -> str:
    """
    Build a filename of the form: className_timestamp.extension

    Args:
        file_path: File the name is built from. The extension is taken from
            here, so pass the file that will actually be written (a source
            .png compressed to .jpg must keep the .jpg suffix).
        class_name: Detected class, used as the filename prefix.
        timestamp: Seconds since epoch. Read from file_path when omitted.

    Returns:
        The new filename, without any directory part.
    """
    path = Path(file_path)

    if timestamp is None:
      timestamp = get_creation_time(path)

    time_string = format_timestamp(timestamp, self.timestamp_format)

    # Lowercase the suffix so a camera's .JPG/.AVI and an already-lowercase
    # .jpg land on the same convention in the sorted library.
    return f"{class_name}_{time_string}{path.suffix.lower()}"

  def get_unique_filename(self, directory: str, filename: str) -> Path:
    """
    Return a path inside directory that does not exist yet.

    Appends _001, _002, ... to the stem until the name is free.
    """
    directory = Path(directory)
    dest_path = directory / filename

    if not dest_path.exists():
      return dest_path

    stem = Path(filename).stem
    extension = Path(filename).suffix
    counter = 1

    while dest_path.exists():
      dest_path = directory / f"{stem}_{counter:03d}{extension}"
      counter += 1

    return dest_path
