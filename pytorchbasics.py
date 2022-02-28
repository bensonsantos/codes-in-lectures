import torch 
import torchvision
import torchvision.transforms as transforms
import timm 
from einops import rearrange
from PIL import Image
import matplotlib.pyplot as plt
import librosa

b = torch.ones((3,3))
print(type(b))
print(b)

a = torch.tensor([[2.,2.], [4.,4.]])
print(a)

img = Image.open("wonder_cat.jpg")
plt.imshow(img)
plt.show()

words = {"hello": 0, "world": 1}
embed_len = len(words)
embed_dim = 4
embed = torch.nn.Embedding(embed_len, embed_dim)
lookup = torch.tensor([words["hello"]], dtype=torch.long)
embed(lookup)
print(embed(lookup))
