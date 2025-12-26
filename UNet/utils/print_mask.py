# from PIL import Image
# from torchvision import transforms

# m = Image.open(r"E:\code\deeplearning\deeplearning\UNet\data\train\masks\0cdf5b5d0ce1_01_mask.gif").convert("L")
# t = transforms.ToTensor()(m)
# print(t.min().item(), t.max().item())

# 简单统计前景比例
import numpy as np
from PIL import Image
from torchvision import transforms

m = Image.open(r"E:\code\deeplearning\deeplearning\UNet\data\train\masks\0cdf5b5d0ce1_01_mask.gif").convert("L")
t = transforms.ToTensor()(m)
print("mean:", t.mean().item())
