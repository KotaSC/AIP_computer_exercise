import numpy as np
from PIL import Image
import random
import math
from matplotlib import pyplot as plt

def main():

    # Number of class
    d         = 5

    # Number of pixels randomly selected from training data
    train_num = 10

    # [R, G, B] channel
    channel   = 3

    mean      = np.empty( (d, channel), float )
    cov       = np.empty( (channel, channel, d), float )

    # 画像読み込み
    for i in range(d):

        # Read each class image
        img                 = np.array(Image.open('image/class' + str(i+1) + '.bmp'))
        height, width       = img.shape[0], img.shape[1]

        sum_training_data   = np.zeros(3)
        sum_training_data2  = np.zeros(3)

        mean_training_data  = np.zeros(3)
        mean_training_data2 = np.zeros(3)

        tmp_R = []
        tmp_G = []
        tmp_B = []

        for r in range(train_num):

            # Randomly select pixels from training data
            tmp                 = np.array(img[random.randint(0, height-1), random.randint(0, width-1)], dtype='int64')

            sum_training_data  += tmp
            sum_training_data2 += tmp**2

            tmp_R.append(tmp[0])
            tmp_G.append(tmp[1])
            tmp_B.append(tmp[2])

        # Caluculate mean vector
        mean_training_data  = sum_training_data/10
        mean_training_data2 = sum_training_data2/10

        # Cluculate variance
        var = mean_training_data2 - mean_training_data**2

        sum_RG = 0
        sum_GB = 0
        sum_BR = 0

        for j in range(train_num):

            sum_RG += (tmp_R[j] - mean_training_data[0])*(tmp_G[j] - mean_training_data[1])
            sum_GB += (tmp_G[j] - mean_training_data[1])*(tmp_B[j] - mean_training_data[2])
            sum_BR += (tmp_B[j] - mean_training_data[2])*(tmp_R[j] - mean_training_data[0])

        # Caluculate covariance
        cov_RG = sum_RG/train_num
        cov_GB = sum_GB/train_num
        cov_BR = sum_BR/train_num

        # Caluculate covariance matrix
        cov_Matrix = np.array([var[0], cov_RG, cov_BR, cov_RG, var[1], cov_GB, cov_BR, cov_GB, var[2]]).reshape((3,3))

        mean[i, :]   = mean_training_data
        cov[:, :, i] = cov_Matrix

    # Read target image
    satellite_image = np.array(Image.open('image/irabu_zhang1.bmp'))
    height, width   = satellite_image.shape[0], satellite_image.shape[1]
    result_image    = np.empty( (height, width), dtype='int64' )

    for h in range(height):
        for w in range(width):

            # Each pixle of target image
            x = satellite_image[h,w,:]

            # Caluculate likelihood for all classes
            P = []
            for k in range(d):

                # Caluculate determinant
                det = np.linalg.det(cov[:,:,k])

                # Caluculate inverse matrix
                inv = np.linalg.inv(cov[:,:,k])

                left  = ( 2*np.pi**(d/2) * det**(0.5) ) ** (-1)
                D     = np.dot( np.dot( (x-mean[k,:]).T, inv ), x-mean[k,:] )
                right = np.exp(-0.5*D)
                L     = left * right

                P.append(L)

            result_image[h,w] = np.argmax(P) + 1

    plt.figure(figsize=(8, 6))
    plt.imshow(result_image, cmap='jet')
    plt.colorbar(shrink=0.85)
    plt.savefig("0509.png")
    plt.show()

if __name__ == "__main__":
    main()