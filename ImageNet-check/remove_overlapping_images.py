import os

# File paths
overlaps_file = 'overlaps.txt'
train_checksums_file = 'official_checksums.txt'

# Load overlapping SHA-1 checksums
with open(overlaps_file, 'r') as f:
    overlaps = set(f.read().splitlines())

# Find and remove overlapping files in the training dataset
with open(train_checksums_file, 'r') as f:
    for line in f:
        checksum, file_path = line.strip().split(maxsplit=1)
        if checksum in overlaps:
            try:
                os.remove(file_path)
                print(f"Removed: {file_path}")
            except Exception as e:
                print(f"Error removing {file_path}: {e}")

print("Overlapping images removed from the validation dataset.")
