import matplotlib.pyplot as plt
import numpy as np
a=plt.imread('./data_semantics/training/semantic_rgb/000000_10.png')
unique_0=[0]
unique_1=[0]
unique_2=[0]


for m in range(np.shape(a)[0]):
    for n in range(np.shape(a)[1]):
        count=0
        for i in range(len(unique_0)):
            if a[m,n,0]*255== unique_0[i] and a[m,n,1]*255== unique_1[i] and a[m,n,2]*255== unique_2[i]:
                count=count+1
                continue
        if count==0:
            unique_0.append(a[m, n, 0] * 255)
            unique_1.append(a[m, n, 1] * 255)
            unique_2.append(a[m, n, 2] * 255)

for m in range(len(unique_0)):
    print (unique_0[m],unique_1[m],unique_2[m])