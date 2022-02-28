from einops import rearrange, repeat, reduce
from matplotlib import image
import numpy as np
import matplotlib.pyplot as plt

img = image.imread("aki_dog.jpg")
print(img.shape)
plt.imshow(img)
plt.show()

img = rearrange(img, "h w c -> (h w c)")
print(img.shape)

img = image.imread("aki_dog.jpg")
img = rearrange(img, "(p1 h) (p2 w) c-> (p1 p2) h w c", p1=2, p2=2)
print(img.shape)

for i in range(4) :
    plt.imshow(img[i])
    plt.show()

img1 = image.imread("aki_dog.jpg")
img2 = image.imread("wonder_cat.jpg")

imgs = np.array([img1, img2])
print(imgs.shape)

#### Arrange as a 2D array of upper half/lower half of images
imgs = rearrange(imgs, "b (k h) w c -> k b h w c", k =2)

#### Reverse the order of the lower half of images
imgs = np.concatenate([imgs[::2], imgs[1::2,::-1]], axis=0)

#### Combine the new upper and lower half of images
imgs = rearrange(imgs, "i j h w c -> j (i h) w c")
