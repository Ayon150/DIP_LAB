from PIL import Image, ImageOps
import os

# Define input and output folders
input_folder = "resized_images"  #same folder as script
output_folder = "digit_images_inverted" #same folder as script

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Process each image
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.jpg")

        try:
            # Open image and convert to grayscale
            img = Image.open(input_path).convert("L")

            # Invert colors: white ↔ black
            inverted = ImageOps.invert(img)

            # Convert to RGB and save as JPG
            inverted.convert("RGB").save(output_path, "JPEG")
            print(f"Saved inverted: {output_path}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")
