#================= Importing necessary libraries ======================
import cv2
import matplotlib.pyplot as plt

#================= Execution workflow ======================
def main():
    # Load the image (change path as needed)
    img_path = r"C:\Users\User\Desktop\CSE 4161_DIP\image\monarch_in_may.jpg"
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale

    # Check if image loaded successfully
    if img is None:
        print("Error: Image not found or unable to load.")
        return

    # Apply Canny Edge Detection
    edges = cv2.Canny(img, threshold1=100, threshold2=200)

    # Plotting results
    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    plt.imshow(img, cmap='gray')
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(edges, cmap='gray')
    plt.title("Canny Edge Detection")
    plt.axis("off")

    plt.show()


if __name__ == "__main__":
    main()
