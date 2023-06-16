import os
import numpy as np

# List of filenames and their corresponding counts
files_counts = [
    ("12TVK440540_CROP_02.png", 285),
    ("12TVK440540_CROP_03.png", 140),
    ("12TVK440540_CROP_04.png", 596),
    ("12TVK440540_CROP_05.png", 881),
    ("12TVK440540_CROP_06.png", 94),
    ("12TVK440540_CROP_07.png", 28),
    ("12TVK440540_CROP_08.png", 208),
    ("12TVL180480_CROP_09.png", 249),
    ("12TVL180480_CROP_10.png", 215),
    ("12TVK460400_CROP_11.png", 498),
    ("12TVL220060_CROP_s1.png", 51),
    ("12TVL220060_CROP_s2.png", 42),
    ("12TVL460100_CROP_s1.png", 22),
    ("12TVL460100_CROP_s2.png", 23),
    ("12TVK220780_CROP_s1.png", 20),
    ("12TVK220780_CROP_s2.png", 20),
    ("12TVL240360_CROP_s1.png", 28),
    ("12TVL240360_CROP_s2.png", 36),
    ("12TVL160120_CROP_s1.png", 10),
    ("12TVL160120_CROP_s2.png", 10)
]
# Directory where the files are located
directory = "testing_scenes"

for filename, count in files_counts:
    # Create dummy annotations
    annotations = np.random.rand(count, 2).astype('float32')

    # Save as .npy file in the specified directory
    np.save(os.path.join(directory, filename.replace('.png', '.npy')), annotations)
