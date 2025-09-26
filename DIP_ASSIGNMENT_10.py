import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure output directory exists
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
os.makedirs(output_dir, exist_ok=True)


def main():
    img = cv2.imread(r'C:\Users\User\Desktop\CSE 4161_DIP\image\homework-10.png', cv2.IMREAD_GRAYSCALE)
    img_binary = np.where(img > 127, 1, 0).astype(np.uint8)

    

    kernel = {
        'rectangular': get_kernel('rectangular'),
        'elliptical': get_kernel('elliptical'),
        'cross': get_kernel('cross'),
        'diamond': get_kernel('diamond')
    }

    operations = [ 'erosion', 'dilation' , 'opening', 'closing' , 'top_hat', 'black_hat' ]

    for op in operations: # perform all operation
        img_set = [img ]
        img_title = ['Original Image']

        for k_name, k in kernel.items(): # for built-in function
            if op == 'erosion':
                processed_img = cv2.erode(img_binary, k, iterations=1)
            elif op == 'dilation':
                processed_img = cv2.dilate(img_binary, k, iterations=1)
            elif op == 'opening':
                processed_img = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, k)
            elif op == 'closing':
                processed_img = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, k)
            elif op == 'top_hat':
                processed_img = cv2.morphologyEx(img_binary, cv2.MORPH_TOPHAT, k)
            elif op == 'black_hat':
                processed_img = cv2.morphologyEx(img_binary, cv2.MORPH_BLACKHAT, k)

            img_set.append(processed_img)
            img_title.append(f'{op} - {k_name}')
        
        for k_name, k in kernel.items(): # for user define function
            if op == 'erosion':
                processed_img = manual_erosion(img_binary, k)
            if op == 'dilation':
                processed_img = manual_dilation(img_binary, k)
            if op == 'opening':
                processed_img = manual_opening(img_binary, k)
            if op == 'closing':
                processed_img = manual_closing(img_binary, k)
            if op == 'top_hat':
                processed_img = manual_top_hat(img_binary, k)
            if op == 'black_hat':
                processed_img = manual_black_hat(img_binary, k)
            
            img_set.append(processed_img)
            img_title.append(f'{op} - {k_name} (manual)')
        
        display(img_set, img_title, op)

        # # use for save image
        # display(img_set, img_title, op) 


    input("Press Enter to exit...")



def manual_erosion(img, kernel):
    k_height, k_width = kernel.shape
    pad_h, pad_w = k_height // 2, k_width // 2
    padded_img = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    eroded_img = np.zeros_like(img)

    for i in range(eroded_img.shape[0]):
        for j in range(eroded_img.shape[1]):
            region = padded_img[i:i + k_height, j:j + k_width]
            if np.all(region[kernel == 1] == 1):
                eroded_img[i, j] = 1
            else:
                eroded_img[i, j] = 0

    return eroded_img

def manual_dilation(img, kernel):
    k_height, k_width = kernel.shape
    pad_h, pad_w = k_height // 2, k_width // 2
    padded_img = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    dilated_img = np.zeros_like(img)

    for i in range(dilated_img.shape[0]):
        for j in range(dilated_img.shape[1]):
            region = padded_img[i:i + k_height, j:j + k_width]
            if np.any(region[kernel == 1] == 1):
                dilated_img[i, j] = 1
            else:
                dilated_img[i, j] = 0

    return dilated_img

def manual_opening(img, kernel):
    eroded = manual_erosion(img, kernel)
    opened = manual_dilation(eroded, kernel)
    return opened

def manual_closing(img, kernel):
    dilated = manual_dilation(img, kernel)
    closed = manual_erosion(dilated, kernel)
    return closed

def manual_top_hat(img, kernel):
    opened = manual_opening(img, kernel)
    top_hat = img - opened
    return top_hat

def manual_black_hat(img, kernel):
    closed = manual_closing(img, kernel)
    black_hat = closed - img
    return black_hat



def display(img_set, img_title, op_name):

    for i in range(len(img_set)):        
        plt.subplot(2, 5, i + 1)  
        if(img_set[i].ndim == 2):    # show gray-scale image 
            plt.imshow(img_set[i], cmap = 'gray')
            plt.title(img_title[i])  
            plt.axis('off') 
        else:   # show histogram
            plt.bar(range(256), img_set[i])
            plt.title(img_title[i])            
    
    plt.tight_layout() 
    plt.show(block = True)

   

#---------- for save images --------------
# def display(img_set, img_title, op_name):
#     n_images = len(img_set)
#     cols = 4   # 4 images per row
#     rows = (n_images + cols - 1) // cols  # auto row calculation
    
#     plt.figure(figsize=(16, 10))  # bigger window
    
#     for i in range(n_images):        
#         plt.subplot(rows, cols, i + 1)
#         plt.imshow(img_set[i], cmap='gray')
#         plt.title(img_title[i], fontsize=9)  # smaller font
#         plt.axis('off')
    
#     plt.tight_layout(pad=2.0)  # add spacing between titles
    
#     output_path = os.path.join(output_dir, f"{op_name}_results.png")
#     plt.savefig(output_path, dpi=300, bbox_inches='tight')
#     plt.close()




def get_kernel(name):
    if name == 'rectangular':
        return np.array([[1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1]], dtype=np.uint8)
    
    elif name == 'elliptical':
        return np.array([[0, 1, 1, 1, 0],
                         [1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1],
                         [0, 1, 1, 1, 0]], dtype=np.uint8)
    
    elif name == 'cross':
        return np.array([[0, 0, 1, 0, 0],
                         [0, 0, 1, 0, 0],
                         [1, 1, 1, 1, 1],
                         [0, 0, 1, 0, 0],
                         [0, 0, 1, 0, 0]], dtype=np.uint8)
    
    elif name == 'diamond':
        return np.array([[0, 0, 1, 0, 0],
                         [0, 1, 1, 1, 0],
                         [1, 1, 1, 1, 1],
                         [0, 1, 1, 1, 0],
                         [0, 0, 1, 0, 0]], dtype=np.uint8)
    else:
        raise ValueError("Unknown kernel name")



#------------- design kernel using builtin-fumction-------------
# def get_kernel(name):
#     if name == 'rectangular':
#         return cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    
#     elif name == 'elliptical':
#         return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    
#     elif name == 'cross':
#         return cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
    
#     elif name == 'diamond':
#         return np.array([[0, 0, 1, 0, 0],
#                          [0, 1, 1, 1, 0],
#                          [1, 1, 1, 1, 1],
#                          [0, 1, 1, 1, 0],
#                          [0, 0, 1, 0, 0]], dtype=np.uint8)
#     else:
#         raise ValueError("Unknown kernel name")



if __name__ == '__main__':
    main()
