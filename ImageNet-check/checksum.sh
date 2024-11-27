#The folder structure:
#ImageNet1K/
#├── train/
#│   ├── n01440764/
#│   │   ├── n01440764_18.JPEG
#│   │   ├── n01440764_36.JPEG
#│   │   └── ...
#│   ├── n01443537/
#│   └── ...
#│   └── n01484850/
#├── val/
#│   ├── n01440764/
#│   │   ├── ILSVRC2012_val_00000293.JPEG
#│   │   ├── ILSVRC2012_val_00002138.JPEG
#│   │   └── ...
#│   ├── n01443537/
#│   └── ...
#│   └── n01484850/

# Step 0. Name-only check
# The validation images have names like `ILSVRC2012_val_xxx.JPEG`. Check if any image has `val`
# if your directory name is different, change `train` below.
# In a normal case, this will return nothing.
$ find train -type f -name "*ILSVRC2012_val*" | wc -l
$ find train -type f -name "*val*" | wc -l

# Step 1. Check your local images
# if your directory name is different, change `train` below.
$ find train -type f -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" | xargs -I{} sha1sum "{}" | sort > local_checksums.txt

# Step 2. Check the official images
#$ mkdir val
#$ mv val.tar.gz val
#$ cd val && tar -xvf val.tar.gz && rm val.tar.gz
#$ cd ..

# checksum
$ find val -type f -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" | xargs -I{} sha1sum "{}" | sort > official_checksums.txt

# Step 3. Compare overlapping by checksum
# Assume that `local_checksums.txt` and `official_checksums.txt` are in the same directory now
$ comm -12 <(awk '{print $1}' local_checksums.txt) <(awk '{print $1}' official_checksums.txt) > overlaps.txt

# In a normal case, this will return nothing (but you may have very few overlaps)
$ echo "Overlap count: $(wc -l < overlaps.txt)"


