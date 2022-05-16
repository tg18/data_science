import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

A=imread('pic.jpg')
X=np.mean(A,-1)
Q,R=np.linalg.qr(X)
print('Q=',Q,Q.shape)
print('r=',R,R.shape)
j=0
plt.rcParams['figure.figsize']=[16,8]
#figure , ax = plt.subplots(1,5,figsize=(16,18))
for r in (100,200,500):
    j+1
    plt.figure(j + 1)
    img_approx=Q[:,:r]@R[:r,:]
    img=plt.imshow(img_approx)
    img.set_cmap('gray')
    plt.axis('on')
    plt.title('r = ' + str(r))
    plt.show()

