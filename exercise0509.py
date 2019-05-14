import numpy as np
from PIL import Image
import random
import math

def main():

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

        variance = mean_training_data2 - mean_training_data**2

        sum_RG = 0
        sum_GB = 0
        sum_BR = 0

        for h in range(10):

            sum_RG += (tmp_R[h] - mean_training_data[0])*(tmp_G[h] - mean_training_data[1])
            sum_GB += (tmp_G[h] - mean_training_data[1])*(tmp_B[h] - mean_training_data[2])
            sum_BR += (tmp_R[h] - mean_training_data[0])*(tmp_B[h] - mean_training_data[2])

        cov_RG = sum_RG/10
        cov_GB = sum_GB/10
        cov_BR = sum_BR/10

        cov_Matrix     = np.array([variance[0], cov_RG, cov_BR, cov_RG, variance[1], cov_GB, cov_BR, cov_GB, variance[2]], dtype='float').reshape((3,3))
        inv_cov_Matrix = np.linalg.inv(cov_Matrix)
        det_cov_Matrix = np.linalg.det(cov_Matrix)

        # print((det_cov_Matrix)**(0.5))

        D = (np.array([100, 100, 100])-mean_training_data).T*inv_cov_Matrix*(np.array([100, 100, 100])-mean_training_data)

        # bunbo = ((2*math.pi)**(2.5)) * ((det_cov_Matrix)**(0.5) )
        # L = np.exp(D) / bunbo
        # print(L)
        # print(det_cov_Matrix**(0.5))
        # test= np.array([100, 100, 100])
        # print(test.shape)
        # print(np.array(100, 100, 100)-mean_training_data)

if __name__ == "__main__":
    main()