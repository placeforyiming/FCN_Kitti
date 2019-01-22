
# This is the first step for kitti segmentation
# In this script, we resize all images to (1242,375) and generate the class label for all the pixels
import cv2
import numpy as np
import matplotlib.pyplot as plt
import shutil
import os
Name=['road','sidewalk','building','wall','fence','pole','traffic light','traffic sign','vegetation','terrain','sky','person','rider','car','truck','bus','train','motorcycle','bicycle']
Label=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
Pixel=[(128,64,128),(244,35,232),(70,70,70),(102,102,156),(190,153,153),(153,153,153),(250,170,30),(220,220,0),(107,142,35),(152,251,152),(70,130,180),(220,20,60),(255,0,0),(0,0,142),(0,0,70),(0,60,100),(0,80,100),(0,0,230),(119,11,32)]

Img_number_dir=os.listdir('./Dataset/semantic_image/train/')
for i in range(len(Img_number_dir)):
    print (i)

    a = plt.imread('./Dataset/semantic_image/train/'+Img_number_dir[i])
    b = plt.imread('./Dataset/original_image/train/' + Img_number_dir[i])


    aa = cv2.resize(a, (1242, 375))
    bb = cv2.resize(b, (1242, 375))

    plt.imsave('./Dataset/semantic_image/train/'+Img_number_dir[i],aa)
    plt.imsave('./Dataset/original_image/train/' + Img_number_dir[i],bb)

    a = plt.imread('./Dataset/semantic_image/train/' + Img_number_dir[i])

    label_matrix=np.zeros(shape=(np.shape(a)[0],np.shape(a)[1],1))
    for m in range(np.shape(a)[0]):
        for n in range(np.shape(a)[1]):
            pixel=a[m,n,:3]*255
            backgroud=1
            for k in range(len(Pixel)):
                if abs(Pixel[k][0]-pixel[0])<2 and abs(Pixel[k][1]-pixel[1])<2 and abs(Pixel[k][2]-pixel[2])<2:
                    label_matrix[m,n,0]=k
                    backgroud=0
                    break
            if backgroud==1:
                label_matrix[m, n, 0] = 19

    np.save('./Dataset/label_image/train/'+Img_number_dir[i][:(len(Img_number_dir[i])-4)], label_matrix)

