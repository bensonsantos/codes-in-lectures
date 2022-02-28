import numpy as np
import matplotlib.pyplot as plt
a = np.array([[1,2,3], [4,5,6]])
print(a)

print(a.dtype)
print(a.shape)
print(a.ndim)

b = 2 + a
print(b)

c = np.ones((2,3),dtype=np.int32)
print(c)
d = a + c
print(d)
e = a*d
print(e)
f = np.matmul(a,np.transpose(c))
print(f)
g = a@np.transpose(c)
print(g)

img = np.random.randint(0,255,size=(96,96),dtype=np.uint8)
plt.imshow(img,cmap='gray', vmin=0,vmax=255)
plt.show()

img = np.ones((96,96), dtype=np.uint8)*255
for i in range(4):
    img[i*24:(i+1)*24, i*24:(i+1)*24] = 0
for i in range(2,4):
    img[i*24:(i+1)*24, (i-2)*24:(i-1)*24] = 0 
for i in range(0,2):
    img[i*24:(i+1)*24, (i+2)*24:(i+3)*24] = 0
plt.imshow(img,cmap='gray', vmin=0,vmax=255)
plt.show()

def chessboard(shape):
    return np.indices(shape).sum(axis=0) % 2

img = chessboard((4,4))*255
img = np.repeat(img,(24),axis=0)
img = np.repeat(img,(24),axis=1)
plt.imshow(img,cmap='gray', vmin=0,vmax=255)
plt.show()

def chess(by):
    return np.indices(by).sum(axis=0) % 2

im = chess((4,4))*255
print(im)

im = np.repeat(im, 24, axis=0)
im = np.repeat(im, 24, axis=1)
print(im)

plt.imshow(im,cmap='gray', vmin=0,vmax=255)
plt.show()