"""
Problem statement:
    Extracts 8 bit planes from a grayscale image.
"""

#==============  Import required libraries =================================
import matplotlib.pyplot as plt  
import cv2                       
import numpy as np               


def main():
    #============= Read the image in grayscale (0 flag = grayscale) =========
    img_gray = cv2.imread("C:/Users/User/Desktop/CSE 4161_DIP/image/tulip.jpeg", 0)

    #============= List to store original image + bit planes ================
    planes = []
    planes.append(img_gray)  

    #================= Extract bit planes from the image ====================
    planes += img_plane(img_gray)

    #================= Reconstruct the original =============================
    planes += [planes[1] + planes[2] + planes[3] + planes[4] +
               planes[5] + planes[6] + planes[7] + planes[8]]

    #================== Display all images ==================================
    img_title = ['original', 'bit-1', 'bit-2', 'bit-3', 'bit-4', 'bit-5', 
                'bit-6', 'bit-7', 'bit-8', 'reconstructed']
    display(planes, img_title)  


#===================== function for extracting 8 bit plane of an image ======
def img_plane(img):
    
    planes = []

    for i in range(8):
        tmp = img & (1 << i)
        planes.append(tmp)
    
    return planes

#====================  function to display all image ========================

def display(img_set, img_title):

    for i in range(len(img_set)):
        plt.subplot(2, 5, i + 1)         
        plt.imshow(img_set[i], cmap='gray')
        plt.title(img_title[i])  
        plt.axis('off')                   
    
    plt.tight_layout() 
    plt.show()


if __name__ == "__main__":
    main()
