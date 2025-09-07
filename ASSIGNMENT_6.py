#============== Import required libraries =================================
import matplotlib.pyplot as plt
import cv2
import numpy as np


#============== Custom filter2D function ==================================
def custom_filter2D(img, kernel, mode='same'):
    kernel = np.flipud(np.fliplr(kernel))  # Flip kernel for convolution
    k_h, k_w = kernel.shape
    i_h, i_w = img.shape

    if mode == 'same':
        pad_h, pad_w = k_h // 2, k_w // 2
        padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    elif mode == 'valid':
        padded = img
    else:
        raise ValueError("Mode must be 'same' or 'valid'.")

    o_h = padded.shape[0] - k_h + 1
    o_w = padded.shape[1] - k_w + 1
    output = np.zeros((o_h, o_w))

    for y in range(o_h):
        for x in range(o_w):
            region = padded[y:y+k_h, x:x+k_w]
            output[y, x] = np.sum(region * kernel)

    return np.clip(output, 0, 255).astype(np.uint8)


def main():
    #============= Read the image in grayscale =============================
    img_gray = cv2.imread(r'C:\Users\User\Desktop\CSE 4161_DIP\image\My_Screen.png', cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        print("Error: Image not found!")
        return

    #============= Define kernels ==========================================
    # Average filter
    avg_filter = np.ones((3, 3), dtype=np.float32) / 9

    # Sobel
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]], dtype=np.float32)

    # Prewitt
    prewitt_x = np.array([[-1, 0, 1],
                          [-1, 0, 1],
                          [-1, 0, 1]], dtype=np.float32)
    prewitt_y = np.array([[-1, -1, -1],
                          [0, 0, 0],
                          [1, 1, 1]], dtype=np.float32)

    # Scharr
    scharr_x = np.array([[-3, 0, 3],
                         [-10, 0, 10],
                         [ -3, 0, 3]], dtype=np.float32)
    scharr_y = np.array([[-3, -10, -3],
                         [ 0,   0,  0],
                         [ 3,  10,  3]], dtype=np.float32)

    # Laplacian
    laplace = np.array([[0, 1, 0],
                        [1, -4, 1],
                        [0, 1, 0]], dtype=np.float32)

    # Custom Kernels
    custom_k1 = np.array([[1, -1, 0],
                          [0, 1, -1],
                          [1, 0, -1]], dtype=np.float32)

    custom_k2 = np.array([[2, 0, -2],
                          [0, 0,  0],
                          [-2, 0, 2]], dtype=np.float32)

    custom_k3 = np.array([[0, 1, 0],
                          [1, -5, 1],
                          [0, 1, 0]], dtype=np.float32)

    custom_k4 = np.array([[1, 1, 1],
                          [1, -8, 1],
                          [1, 1, 1]], dtype=np.float32)

    #============= Apply filters using custom_filter2D =====================
    avg = custom_filter2D(img_gray, avg_filter)
    sobelx = custom_filter2D(img_gray, sobel_x)
    sobely = custom_filter2D(img_gray, sobel_y)
    prewittx = custom_filter2D(img_gray, prewitt_x)
    prewitty = custom_filter2D(img_gray, prewitt_y)
    scharrx = custom_filter2D(img_gray, scharr_x)
    scharry = custom_filter2D(img_gray, scharr_y)
    lap = custom_filter2D(img_gray, laplace)
    c1 = custom_filter2D(img_gray, custom_k1)
    c2 = custom_filter2D(img_gray, custom_k2)
    c3 = custom_filter2D(img_gray, custom_k3)
    c4 = custom_filter2D(img_gray, custom_k4)

    #============= Collect images for display ==============================
    img_set = [img_gray, avg, sobelx, sobely,
               prewittx, prewitty, scharrx, scharry,
               lap, c1, c2, c3, c4]

    img_title = ['Original', 'Average',
                 'Sobel-X', 'Sobel-Y',
                 'Prewitt-X', 'Prewitt-Y',
                 'Scharr-X', 'Scharr-Y',
                 'Laplacian',
                 'Custom-1', 'Custom-2', 'Custom-3', 'Custom-4']

    #============= Display =================================================
    display(img_set, img_title)


#==================== Function to display all images ======================
def display(img_set, img_title):
    plt.figure(figsize=(12, 10))
    for i in range(len(img_set)):
        plt.subplot(3, 5, i + 1)
        plt.imshow(img_set[i], cmap='gray')
        plt.title(img_title[i])
        plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
