from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams['figure.figsize']=[16,8]
A=imread("pic.jpg")
print(A.shape)
print('A FIRST VALUES',A[0,0,0],A[0,0,1],A[0,0,2])
print('a=',A,A.shape)
X=np.mean(A,-1)
#print('a=',A)
print('X=',X,X.shape)
U,S,VT = np.linalg.svd(X,full_matrices=False)
print('s before=',S,S.shape)
print('u=',U,U.shape)
print('v=',VT,VT.shape)
X=np.diag(S)
print('s=',S,S.shape)
j=0

#fig, ax1 = plt.subplots(1,2)
plt.figure()
IMG=np.zeros((2005,3*2045))
for r in (5,50,100):
    img_approx = U[:,:r] @ X[0:r,:r] @ VT[:r,:]
    #plt.figure(j+1)
    if j==0:
        IMG[:2005,:2045]=img_approx

    if j==1:
        IMG[:2005,2045:2*2045]=img_approx
    if j==2:
        IMG[:2005,2*2045:]=img_approx
    j=j+1

img = plt.imshow(IMG)
img.set_cmap('viridis') #('gray')
plt.axis('on')
plt.title('1)5 2)50 3) 100 ')
plt.show()
#ax1.plt.im

Y2=np.zeros(len(S))
for i in range(1,len(S)):
    Y2[0]=S[0]
    Y2[i]=Y2[i-1]+S[i]
Y2=Y2/Y2[len(Y2)-1]


X2=np.arange(len(S))
fig, ax= plt.subplots()
ax.plot(X2,Y2)
plt.show()

