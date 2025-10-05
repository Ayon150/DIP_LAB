import matplotlib.pyplot as plt
import numpy as np
import cv2


def main():
    #============= Read the image in grayscale ============================
    img_gray = cv2.imread(r'C:\Users\User\Desktop\CSE 4161_DIP\image\histogram.webp', cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        print("Error: Image not found!")
        return

    #============= Divide image into 8 parts ==============================
    parts = divide_image(img_gray, rows=4, cols=2)
    print(f"Divided into {len(parts)} parts")

    #============= Equalize each part independently =======================
    equalized_parts = []
    for p in parts:
        hist = calc_histogram(p)
        pdf = calc_pdf(hist)
        cdf = calc_cdf(pdf)
        new_level = np.round(cdf * 255).astype(np.uint8)
        new_img = map_intensity(p, new_level)
        equalized_parts.append(new_img)

    #============= Combine equalized parts back into one image ============
    final_equalized = combine_parts(equalized_parts, rows=4, cols=2)

    #============= Display all parts =====================================
    display_parts(parts, equalized_parts, final_equalized, img_gray)


#================= Divide image into N parts =============================
def divide_image(img, rows=4, cols=2):
    """Divide an image into rows x cols parts (default 8 parts)."""
    h, w = img.shape
    part_h, part_w = h // rows, w // cols
    parts = []
    for i in range(rows):
        for j in range(cols):
            y1, y2 = i * part_h, (i + 1) * part_h
            x1, x2 = j * part_w, (j + 1) * part_w
            part = img[y1:y2, x1:x2]
            parts.append(part)
    return parts


#================= Combine image parts ===================================
def combine_parts(parts, rows=4, cols=2):
    """Combine image parts back into one image."""
    part_h, part_w = parts[0].shape
    combined = np.zeros((rows * part_h, cols * part_w), dtype=np.uint8)
    idx = 0
    for i in range(rows):
        for j in range(cols):
            y1, y2 = i * part_h, (i + 1) * part_h
            x1, x2 = j * part_w, (j + 1) * part_w
            combined[y1:y2, x1:x2] = parts[idx]
            idx += 1
    return combined


#================= Histogram Functions ===================================
def calc_histogram(img):
    h, w = img.shape
    hist = np.zeros(256, dtype=int)
    for i in range(h):
        for j in range(w):
            pixel_value = img[i, j]
            hist[pixel_value] += 1
    return hist

def calc_pdf(hist):
    return hist / hist.sum()

def calc_cdf(pdf):
    return np.cumsum(pdf)

def map_intensity(img, new_level):
    return new_level[img]


#================= Display Function ======================================
def display_parts(original_parts, equalized_parts, final_image, img_gray):
    plt.figure(figsize=(14, 12))

    # Display original 8 parts
    for i, part in enumerate(original_parts):
        plt.subplot(5, 4, i + 1)
        plt.imshow(part, cmap='gray')
        plt.title(f"Original Part {i + 1}")
        plt.axis('off')

    # Display equalized 8 parts
    for i, part in enumerate(equalized_parts):
        plt.subplot(5, 4, i + 9)
        plt.imshow(part, cmap='gray')
        plt.title(f"Equalized Part {i + 1}")
        plt.axis('off')

    # Display combined results
    plt.subplot(5, 2, 9)
    plt.imshow(final_image, cmap='gray')
    plt.title("Final Combined Equalized Image")
    plt.axis('off')

    plt.subplot(5, 2, 10)
    plt.imshow(img_gray, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

#================= Run the Program =======================================
if __name__ == "__main__":
    main()
