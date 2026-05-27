import os
import random

# --- Configuration ---
# Path to the folder containing your images AND their corresponding.txt annotation files
dataset_path = "obj"
# Percentage of images to use for the validation set (e.g., 15 means 15%)
valid_percentage = 15
# Output file paths (relative to where you run this script)
train_txt_path = "train.txt"
valid_txt_path = "valid.txt"
# --- End Configuration ---

# 1. Get a list of all image files in the dataset_path
# It looks for files ending with common image extensions.
image_files = [
    f for f in os.listdir(dataset_path) if f.endswith((".jpg", ".png", ".jpeg", ".bmp"))
]  # Added.bmp just in case

# 2. Create full paths for each image file
# os.path.join correctly creates the path like "data/obj/image1.jpg"
# These paths will be written to the output files.
image_paths = [os.path.join(dataset_path, f) for f in image_files]

# 3. Shuffle the list randomly
# This ensures that the training and validation sets are random samples
# and not biased by the original order of files (e.g., all easy ones first).
random.shuffle(image_paths)

# 4. Calculate the split point
num_images = len(image_paths)
num_valid = int(num_images * valid_percentage / 100)
num_train = num_images - num_valid

# 5. Split the list into training and validation sets
train_paths = image_paths[:num_train]
valid_paths = image_paths[num_train:]  # Corrected: Use num_train as the starting index

# 6. Write the paths to train.txt
# Opens the file in write mode ('w'), overwriting if it exists.
# Writes each path from the train_paths list, followed by a newline character ('\\n').
with open(train_txt_path, "w") as f_train:
    for path in train_paths:
        f_train.write(path.replace("\\\\", "/") + "\\n")  # Replace backslashes with forward slashes
print(f"Generated {train_txt_path} with {len(train_paths)} image paths.")

# 7. Write the paths to valid.txt
# Does the same as step 6, but for the validation paths.
with open(valid_txt_path, "w") as f_valid:
    for path in valid_paths:
        f_valid.write(path.replace("\\\\", "/") + "\\n")  # Replace backslashes with forward slashes
print(f"Generated {valid_txt_path} with {len(valid_paths)} image paths.")

print("Finished generating train.txt and valid.txt")
