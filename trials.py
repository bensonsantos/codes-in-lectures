import numpy as np
import matplotlib.pyplot as plt

def chess(by):
    return np.indices(by).sum(axis=0) % 2

im = chess((4,4))*255
print(im)

im = np.repeat(im, 24, axis=0)
im = np.repeat(im, 24, axis=1)
print(im)

plt.imshow(im,cmap='gray', vmin=0,vmax=255)
plt.show()