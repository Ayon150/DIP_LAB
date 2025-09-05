import os
from PIL import Image

# Input and output folders
input_folder = "digit_images_jpg"
output_folder = "renamed_images"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop through all files
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".jpg"):
        base_name = os.path.splitext(filename)[0]  # remove extension
        parts = base_name.split("-")

        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            digit = int(parts[0])
            index = int(parts[1])

            # Compute new filename
            new_number = digit * 1000 + 350 + index
            new_name = f"{new_number}.jpg"

            # Load and save image with new name
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, new_name)

            try:
                img = Image.open(input_path)
                img.save(output_path, "JPEG")
                print(f"Saved: {filename} → {new_name}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
