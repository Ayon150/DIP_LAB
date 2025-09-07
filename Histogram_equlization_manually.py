#============== Import required libraries ================================
import matplotlib.pyplot as plt
import numpy as np
import cv2


def main():
    #============= Read the image in grayscale ============================
    img_gray = cv2.imread(r'C:\Users\User\Desktop\CSE 4161_DIP\image\histogram.webp', cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        print("Error: Image not found!")
        return

    #============= Histogram Equalization Pipeline ========================
    hist = calc_histogram(img_gray)                      # Original histogram
    pdf = calc_pdf(hist)                                 # Probability density function
    cdf = calc_cdf(pdf)                                  # Cumulative distribution function
    new_level = np.round(cdf * 255).astype(np.uint8)     # Mapping new intensity levels
    new_img = map_intensity(img_gray, new_level)         # Apply mapping to image
    hist_eq = calc_histogram(new_img)                    # Histogram of equalized image

    #============= Collect data for display ===============================
    img_set = [img_gray, new_img, hist, pdf, cdf, hist_eq]
    img_title = ["Original Image", "Equalized Image", "Original Histogram",
                 "PDF", "CDF", "Equalized Histogram"]

    #============= Display =================================================
    display(img_set, img_title)


#================= Calculate histogram (manual) ===========================
def calc_histogram(img):
    h, w = img.shape
    hist = np.zeros(256, dtype=int)
    for i in range(h):
        for j in range(w):
            pixel_value = img[i, j]
            hist[pixel_value] += 1
    return hist


#================= Calculate PDF =========================================
def calc_pdf(hist):
    return hist / hist.sum()


#================= Calculate CDF =========================================
def calc_cdf(pdf):
    return np.cumsum(pdf)


#================= Map new intensity levels ==============================
def map_intensity(img, new_level):
    return new_level[img]


#================= Display results =======================================
def display(img_set, img_title):
    plt.figure(figsize=(12, 10))
    for i in range(len(img_set)):
        plt.subplot(3, 2, i + 1)
        if i in [0, 1]:   # Images
            plt.imshow(img_set[i], cmap='gray')
            plt.axis('off')
        else:             # Histogram / PDF / CDF
            plt.plot(img_set[i])
            plt.xlim([0, 255])
        plt.title(img_title[i])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
