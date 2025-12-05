import os
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def copy_json_with_time_record(src_dir, dest_dir):
    # Ensure the destination directory exists, create if not
    os.makedirs(dest_dir, exist_ok=True)

    # Recursively walk through the source directory
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            # If the file is a JSON file and its name contains 'time_record'
            if file.endswith('.json') and 'time_record' in root:
                src_file = os.path.join(root, file)
                dest_file = os.path.join(dest_dir, file)

                # If the destination file already exists, modify the filename to avoid overwriting
                base, ext = os.path.splitext(file)
                counter = 1
                while os.path.exists(dest_file):
                    dest_file = os.path.join(dest_dir, f"{base}_{counter}{ext}")
                    counter += 1

                # Copy the file to the destination directory
                shutil.copy(src_file, dest_file)
                print(f"File {src_file} copied to {dest_file}")


DEFAULT_SRC_DIR = PROJECT_ROOT / "data"  # Path to the source directory
DEFAULT_DEST_DIR = PROJECT_ROOT / "data_ot" / "time_record"  # Path to the destination directory


if __name__ == "__main__":
    copy_json_with_time_record(str(DEFAULT_SRC_DIR), str(DEFAULT_DEST_DIR))
