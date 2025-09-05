from PIL import Image
import os

# Input and output folder paths
input_folder = "input_images"  #same folder as script
output_folder = "resized_images"  #same folder as script

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop through all files in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
        input_path = os.path.join(input_folder, filename)
        
        # Create output filename with .jpg extension
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(output_folder, f"{base_name}.jpg")

        try:
            # Open, resize, convert to RGB (required for JPG), and save
            img = Image.open(input_path)
            img_resized = img.resize((28, 28), Image.LANCZOS)
            img_resized.convert("RGB").save(output_path, "JPEG")
            print(f"Saved: {output_path}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")
