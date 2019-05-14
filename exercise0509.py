import numpy as np
from PIL import Image
import random

def main():

    # 画像読み込み
    for i in range(5):

        img                 = np.array(Image.open('image/class' + str(i+1) + '.bmp'))
        height, width       = img.shape[0], img.shape[1]
        sum_training_data   = np.array([0]*3)
        sum_training_data2  = np.array([0]*3)
        mean_training_data  = np.array([0]*3)
        mean_training_data2 = np.array([0]*3)

        for r in range(10):

            tmp                 = np.array(img[random.randint(0, height-1), random.randint(0, width-1)], dtype='int64')
            sum_training_data  += tmp
            sum_training_data2 += tmp**2

        mean_training_data  = sum_training_data/10
        mean_training_data2 = sum_training_data2/10

        variance = mean_training_data2 - mean_training_data**2

        sum_RG = 0
        sum_GB = 0
        sum_BR = 0

        for h in range(10):

            tmp   = np.array(img[random.randint(0, height-1), random.randint(0, width-1)], dtype='int64')

            tmp_R = tmp[0]
            tmp_G = tmp[1]
            tmp_B = tmp[2]

            sum_RG += (tmp_R - mean_training_data[0])*(tmp_G - mean_training_data[1])
            sum_GB += (tmp_G - mean_training_data[1])*(tmp_B - mean_training_data[2])
            sum_BR += (tmp_R - mean_training_data[0])*(tmp_B - mean_training_data[2])

        cov_RG = sum_RG/10
        cov_GB = sum_GB/10
        cov_BR = sum_BR/10

        cov_Matrix = np.array([variance[0], cov_RG, cov_BR, cov_RG, variance[1], cov_GB, cov_BR, cov_GB, variance[2]]).reshape((3,3))

        

if __name__ == "__main__":
    main()