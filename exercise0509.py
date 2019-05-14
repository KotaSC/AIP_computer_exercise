import numpy as np
from PIL import Image
import random
import math
import cv2
from matplotlib import pyplot as plt


def ReadImage( _img_name ):
    # read input image
    img_BGR = cv2.imread(_img_name)

    # convert color BGR to RGB
    img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)

    return img_RGB

def main():

    d       = 5
    channel = 3
    mean = np.empty( (d, channel), float )
    cov  = np.empty( (channel, channel, d), float )

    # 画像読み込み
    for i in range(5):

        img                 = np.array(Image.open('image/class' + str(i+1) + '.bmp'))
        height, width       = img.shape[0], img.shape[1]

        sum_training_data   = np.array([0]*3)
        sum_training_data2  = np.array([0]*3)
        mean_training_data  = np.array([0]*3)
        mean_training_data2 = np.array([0]*3)

        tmp_R = []
        tmp_G = []
        tmp_B = []

        for r in range(10):

            tmp                 = np.array(img[random.randint(0, height-1), random.randint(0, width-1)], dtype='int64')
            sum_training_data  += tmp
            sum_training_data2 += tmp**2

            tmp_R.append(tmp[0])
            tmp_G.append(tmp[1])
            tmp_B.append(tmp[2])

        mean_training_data  = sum_training_data/10
        mean_training_data2 = sum_training_data2/10

        var = mean_training_data2 - mean_training_data**2

        sum_RG = 0
        sum_GB = 0
        sum_BR = 0

        for j in range(10):

            sum_RG += (tmp_R[j] - mean_training_data[0])*(tmp_G[j] - mean_training_data[1])
            sum_GB += (tmp_G[j] - mean_training_data[1])*(tmp_B[j] - mean_training_data[2])
            sum_BR += (tmp_B[j] - mean_training_data[2])*(tmp_R[j] - mean_training_data[0])

        cov_RG = sum_RG/10
        cov_GB = sum_GB/10
        cov_BR = sum_BR/10

        cov_Matrix = np.array([var[0], cov_RG, cov_BR, cov_RG, var[1], cov_GB, cov_BR, cov_GB, var[2]]).reshape((3,3))

        mean[i, :]   = mean_training_data
        cov[:, :, i] = cov_Matrix

    print("mean :\n", mean)
    print("cov :\n", cov)

    satellite_image = ReadImage("image/irabu_zhang1.bmp")
    height, width   = satellite_image.shape[0], satellite_image.shape[1]
    result_image    = np.empty( (height, width), np.uint8 )

    for h in range(height):
        for w in range(width):
            # Step 3
            x = satellite_image[h,w,:]
            # if h == 0 and w == 0:
            #     print(x)

            # Step 4
            # Calc likelihood for all class
            P = []
            for k in range(d):
                det = np.linalg.det(cov[:,:,k])
                inv = np.linalg.inv(cov[:,:,k])

                left  = ( 2*np.pi**(d/2) * det**(0.5) ) ** (-1)
                D     = np.dot( np.dot( (x-mean[k,:]).T, inv ), x-mean[k,:] )
                right = np.exp(-0.5*D)
                L     = left * right

                P.append(L)

            if h == 0 and w == 0:
                print(np.argmax(P) + 1)
            result_image[h,w] = np.argmax(P) + 1

            # Show progress
            if ((h*width+w)+1)%(height*width*0.1) == 0:
                print((h*width+w)+1, "/", height*width, " pixels done. ")
    
    result_image = (result_image / np.max(result_image)) * 255
    plt.figure(figsize=(8, 6))
    plt.imshow( result_image, cmap='viridis' )
    # plt.imshow( result_image, cmap='jet' )
    plt.colorbar()
    plt.show()

    np.savetxt("result.txt", result_image, fmt='%d')

if __name__ == "__main__":
    main()