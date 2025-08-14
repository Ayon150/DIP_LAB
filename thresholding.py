import matplotlib.pyplot as plt
import cv2
import numpy as np 

limit1 = 127
limit2 = 200


def main():
    
    img_gray1 = cv2.imread("C:/Users/User/Desktop/CSE 4161_DIP/image/landscape.png", 0)
    img_set = [img_gray1, thresolding1(img_gray1), thresolding2(img_gray1), thresolding3(img_gray1)]
    

    display(img_set)  

    


def display(img_set):
    plt.figure(figsize=(20, 10))

    for i in range(len(img_set)):
        # Display image
        plt.subplot(2, len(img_set), i + 1)
        plt.imshow(img_set[i], cmap='gray')
        plt.axis('off')
        plt.title(f'Image {i+1}')

        # Display histogram
        plt.subplot(2, len(img_set), len(img_set) + i + 1)
        plt.hist(img_set[i].ravel(), bins=256, range=[0, 256], color='gray')
        plt.title(f'Histogram {i+1}')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()


def thresolding1(img_gray):
    img_tmp = img_gray.copy()
    
    for i in range(img_gray.shape[0]):
        for j in range(img_gray.shape[1]):
            if(img_tmp[i][j] <= limit1):
                img_tmp[i][j]    = 0
            else:
                img_tmp[i][j] = 255
    
    
    return img_tmp


def thresolding2(img_gray):
    img_tmp = img_gray.copy()
    
    for i in range(img_gray.shape[0]):
        for j in range(img_gray.shape[1]):
            if( limit1 <= img_tmp[i][j]  and img_tmp[i][j]<=limit2):
                img_tmp[i][j] = 127          
    
    
    return img_tmp

def thresolding3(img_gray):
    img_tmp = img_gray.copy()
    
    for i in range(img_gray.shape[0]):
        for j in range(img_gray.shape[1]):
            if( limit1 <= img_tmp[i][j]  and img_tmp[i][j]<=limit2):
                img_tmp[i][j] = 127 
            if (limit2 < img_tmp[i][j]):
                img_tmp[i][j] = 255
    
    
    return img_tmp

            

if __name__ == '__main__':
    main()
