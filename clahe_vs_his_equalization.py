#================= Importing necessary libraries ======================
import cv2
import numpy as np
import matplotlib.pyplot as plt


#================= Helper function ======================
def plot_histogram(img, title):
    """Return histogram, PDF, and CDF for a grayscale image"""
    hist = cv2.calcHist([img], [0], None, [256], [0,256])
    hist = hist.ravel()

    pdf = hist / hist.sum()                 # Normalize to probability
    cdf = pdf.cumsum()                      # Cumulative distribution
    cdf_normalized = cdf / cdf.max()        # Normalize CDF to [0,1]

    return hist, pdf, cdf_normalized


#================= Execution workflow ======================
def main():
    # Load the grayscale image
    img_path = r"C:\Users\User\Desktop\CSE 4161_DIP\image\histogram.webp"
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("Error: Image not found or unable to load.")
        return

    # 1. Global Histogram Equalization
    hist_eq = cv2.equalizeHist(img)

    # 2. CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_eq = clahe.apply(img)

    # # Save results
    # cv2.imwrite(r"C:\Users\User\Desktop\CSE 4161_DIP\output\monarch_hist_eq.jpg", hist_eq)
    # cv2.imwrite(r"C:\Users\User\Desktop\CSE 4161_DIP\output\monarch_clahe.jpg", clahe_eq)

    # Compute histograms, pdf, cdf
    hist_orig, pdf_orig, cdf_orig = plot_histogram(img, "Original")
    hist_eq_val, pdf_eq, cdf_eq = plot_histogram(hist_eq, "Histogram Equalized")
    hist_clahe, pdf_clahe, cdf_clahe = plot_histogram(clahe_eq, "CLAHE")

    # Plot results
    plt.figure(figsize=(16,12))

    # Images
    plt.subplot(4,3,1), plt.imshow(img, cmap='gray'), plt.title("Original"), plt.axis("off")
    plt.subplot(4,3,2), plt.imshow(hist_eq, cmap='gray'), plt.title("Histogram Equalized"), plt.axis("off")
    plt.subplot(4,3,3), plt.imshow(clahe_eq, cmap='gray'), plt.title("CLAHE"), plt.axis("off")

    # Histograms
    plt.subplot(4,3,4), plt.plot(hist_orig, color='black'), plt.title("Original Histogram")
    plt.subplot(4,3,5), plt.plot(hist_eq_val, color='blue'), plt.title("Equalized Histogram")
    plt.subplot(4,3,6), plt.plot(hist_clahe, color='green'), plt.title("CLAHE Histogram")

    # PDF
    plt.subplot(4,3,7), plt.plot(pdf_orig, color='black'), plt.title("Original PDF")
    plt.subplot(4,3,8), plt.plot(pdf_eq, color='blue'), plt.title("Equalized PDF")
    plt.subplot(4,3,9), plt.plot(pdf_clahe, color='green'), plt.title("CLAHE PDF")

    # CDF
    plt.subplot(4,3,10), plt.plot(cdf_orig, color='black'), plt.title("Original CDF")
    plt.subplot(4,3,11), plt.plot(cdf_eq, color='blue'), plt.title("Equalized CDF")
    plt.subplot(4,3,12), plt.plot(cdf_clahe, color='green'), plt.title("CLAHE CDF")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
