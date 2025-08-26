#============== Import required libraries =================================
import matplotlib.pyplot as plt
import cv2
import numpy as np


def main():
    #============= Read the image in grayscale =============================
    img_gray = cv2.imread(r'C:\Users\User\Desktop\CSE 4161_DIP\image\My_Screen.png', cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        print("Error: Image not found!")
        return

    #============= Add Gaussian noise (manual way) =========================
    row, col = img_gray.shape
    mean = 0
    var = 0.01
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col))
    noisy_img = img_gray + gauss * 255   # scale noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

    #============= Define kernels ==========================================
    # Average filter
    avg_filter = np.array([[1/9, 1/9, 1/9],
                           [1/9, 1/9, 1/9],
                           [1/9, 1/9, 1/9]])

    # Sobel-X
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    # Sobel-Y
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])

    # Prewitt-X
    prewitt_x = np.array([[-1, 0, 1],
                          [-1, 0, 1],
                          [-1, 0, 1]])

    # Prewitt-Y
    prewitt_y = np.array([[-1, -1, -1],
                          [0, 0, 0],
                          [1, 1, 1]])

    # Laplacian
    laplace = np.array([[0, -1, 0],
                        [-1, 4, -1],
                        [0, -1, 0]])

    #============= Apply filters using cv2.filter2D ========================
    avg = cv2.filter2D(noisy_img, -1, avg_filter)
    sobelx = cv2.filter2D(noisy_img, -1, sobel_x)
    sobely = cv2.filter2D(noisy_img, -1, sobel_y)
    prewittx = cv2.filter2D(noisy_img, -1, prewitt_x)
    prewitty = cv2.filter2D(noisy_img, -1, prewitt_y)
    lap = cv2.filter2D(noisy_img, -1, laplace)

    #============= Collect images for display ==============================
    img_set = [img_gray, noisy_img, avg, sobelx, sobely, prewittx, prewitty, lap]
    img_title = ['Original', 'Gaussian Noise', 'Average Filter',
                 'Sobel-X', 'Sobel-Y', 'Prewitt-X', 'Prewitt-Y', 'Laplacian']

    #============= Display =================================================
    display(img_set, img_title)


#==================== Function to display all images ======================
def display(img_set, img_title):
    plt.figure(figsize=(10, 8))
    for i in range(len(img_set)):
        plt.subplot(2, 4, i + 1)
        plt.imshow(img_set[i], cmap='gray')
        plt.title(img_title[i])
        plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
