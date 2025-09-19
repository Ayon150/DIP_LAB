#================= Importing necessary libraries ======================
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure

#================= Built-in Histogram Matching (scikit-image) ======================
def hist_matching_builtin(source, reference):
    matched = exposure.match_histograms(source, reference, channel_axis=None)
    return matched

#================= Custom Histogram Matching (CDF-based) ======================
def hist_matching_cdf(source, reference):
    # Flatten arrays
    src = source.ravel()
    ref = reference.ravel()

    # Unique pixel values and counts
    s_values, bin_idx, s_counts = np.unique(src, return_inverse=True, return_counts=True)
    r_values, r_counts = np.unique(ref, return_counts=True)

    # Compute CDFs
    s_quantiles = np.cumsum(s_counts).astype(np.float64) / src.size
    r_quantiles = np.cumsum(r_counts).astype(np.float64) / ref.size

    # Map source quantiles to reference values
    interp_r_values = np.interp(s_quantiles, r_quantiles, r_values)

    matched = interp_r_values[bin_idx].reshape(source.shape).astype(np.uint8)
    return matched

#================= Custom Histogram Matching (Histogram Specification) ======================
def hist_matching_hist_spec(source, reference, nbins=256):
    # Compute histograms
    src_hist, bins = np.histogram(source.flatten(), nbins, density=True)
    ref_hist, _ = np.histogram(reference.flatten(), nbins, density=True)

    # Compute CDFs
    src_cdf = np.cumsum(src_hist)
    src_cdf /= src_cdf[-1]
    ref_cdf = np.cumsum(ref_hist)
    ref_cdf /= ref_cdf[-1]

    # Create mapping
    mapping = np.interp(src_cdf, ref_cdf, np.linspace(0, 255, nbins))

    # Apply mapping with clipping
    src_bin_idx = np.digitize(source.flatten(), bins) - 1
    src_bin_idx = np.clip(src_bin_idx, 0, nbins-1)  # FIXED
    matched = mapping[src_bin_idx].reshape(source.shape).astype(np.uint8)
    return matched


#================= Execution workflow ======================
def main():
    # Load source and reference images (grayscale)
    src_path = r"C:\Users\User\Desktop\CSE 4161_DIP\image\landscape.png"
    ref_path = r"C:\Users\User\Desktop\CSE 4161_DIP\image\landscape.png"

    source = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    reference = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)

    if source is None or reference is None:
        print("Error: Could not load source or reference image.")
        return

    # Apply Histogram Matching
    matched_builtin = hist_matching_builtin(source, reference)
    matched_cdf = hist_matching_cdf(source, reference)
    matched_spec = hist_matching_hist_spec(source, reference)

    # Save results
    # cv2.imwrite(r"C:\Users\User\Desktop\CSE 4161_DIP\output\matched_builtin.jpg", matched_builtin)
    # cv2.imwrite(r"C:\Users\User\Desktop\CSE 4161_DIP\output\matched_cdf.jpg", matched_cdf)
    # cv2.imwrite(r"C:\Users\User\Desktop\CSE 4161_DIP\output\matched_spec.jpg", matched_spec)

    # Show results
    plt.figure(figsize=(12,8))

    plt.subplot(2,3,1)
    plt.imshow(source, cmap='gray')
    plt.title("Source Image(low contrast)")
    plt.axis("off")

    plt.subplot(2,3,2)
    plt.imshow(reference, cmap='gray')
    plt.title("Reference Image(high contrast)")
    plt.axis("off")

    plt.subplot(2,3,3)
    plt.imshow(matched_builtin, cmap='gray')
    plt.title("Built-in Matching")
    plt.axis("off")

    plt.subplot(2,3,4)
    plt.imshow(matched_cdf, cmap='gray')
    plt.title("Custom Matching (CDF)")
    plt.axis("off")

    plt.subplot(2,3,5)
    plt.imshow(matched_spec, cmap='gray')
    plt.title("Custom Matching (Hist Spec)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

#================= Main ======================
if __name__ == "__main__":
    main()
