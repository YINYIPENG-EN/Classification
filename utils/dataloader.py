from random import shuffle

import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from .utils import cvtColor, preprocess_input

# class DataGenerator(Dataset):
#     def __init__(self, annotation_lines, input_shape, random=True):
#         self.annotation_lines   = annotation_lines  # 训练集长度
#         self.input_shape        = input_shape
#         self.random             = random
#
#     def __len__(self):
#         return len(self.annotation_lines)  # 返回数据集长度
#
#     def __getitem__(self, index):  # 索引每个数据
#         # 用;隔开，类别;图像路径
#         annotation_path = self.annotation_lines[index].split(';')[1].split()[0]
#         image = Image.open(annotation_path)
#         image = self.get_random_data(image, self.input_shape, random=self.random)
#         image = np.transpose(preprocess_input(np.array(image).astype(np.float32)), [2, 0, 1])  # 归一化
#
#         y = int(self.annotation_lines[index].split(';')[0])  # 获得标签0类，1类，2类.....
#         return image, y
#
#     def rand(self, a=0, b=1):
#         return np.random.rand()*(b-a) + a
#
#     def get_random_data(self, image, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
#         #   读取图像并转换成RGB图像
#         image   = cvtColor(image)
#         #   获得图像的高宽与目标高宽
#         iw, ih = image.size
#         h, w = input_shape
#
#         if not random:
#             scale = min(w/iw, h/ih)
#             nw = int(iw*scale)
#             nh = int(ih*scale)
#             dx = (w-nw)//2
#             dy = (h-nh)//2
#
#             #---------------------------------#
#             #   将图像多余的部分加上灰条
#             #---------------------------------#
#             image       = image.resize((nw,nh), Image.BICUBIC)
#             new_image   = Image.new('RGB', (w,h), (128,128,128))
#             new_image.paste(image, (dx, dy))
#             image_data  = np.array(new_image, np.float32)
#
#             return image_data
#
#         #------------------------------------------#
#         #   对图像进行缩放并且进行长和宽的扭曲
#         #------------------------------------------#
#         new_ar = w/h * self.rand(1-jitter,1+jitter)/self.rand(1-jitter,1+jitter)
#         scale = self.rand(.75, 1.25)
#         if new_ar < 1:
#             nh = int(scale*h)
#             nw = int(nh*new_ar)
#         else:
#             nw = int(scale*w)
#             nh = int(nw/new_ar)
#         image = image.resize((nw,nh), Image.BICUBIC)
#
#         #------------------------------------------#
#         #   将图像多余的部分加上灰条
#         #------------------------------------------#
#         dx = int(self.rand(0, w-nw))
#         dy = int(self.rand(0, h-nh))
#         new_image = Image.new('RGB', (w,h), (128,128,128))
#         new_image.paste(image, (dx, dy))
#         image = new_image
#
#         #------------------------------------------#
#         #   翻转图像
#         #------------------------------------------#
#         flip = self.rand()<.5
#         if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)
#
#         rotate = self.rand()<.5
#         if rotate:
#             angle = np.random.randint(-15,15)
#             a,b = w/2,h/2
#             M = cv2.getRotationMatrix2D((a,b),angle,1)
#             image = cv2.warpAffine(np.array(image), M, (w,h), borderValue=[128,128,128])
#
#         #------------------------------------------#
#         #   色域扭曲
#         #------------------------------------------#
#         hue = self.rand(-hue, hue)
#         sat = self.rand(1, sat) if self.rand()<.5 else 1/self.rand(1, sat)
#         val = self.rand(1, val) if self.rand()<.5 else 1/self.rand(1, val)
#         x = cv2.cvtColor(np.array(image,np.float32)/255, cv2.COLOR_RGB2HSV)
#         x[..., 1] *= sat
#         x[..., 2] *= val
#         x[x[:,:, 0]>360, 0] = 360
#         x[:, :, 1:][x[:, :, 1:]>1] = 1
#         x[x<0] = 0
#         image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255
#         return image_data



class CustomDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, transform=None):
        self.annotation_lines = annotation_lines
        self.input_shape = input_shape
        self.transform = transform

    def __len__(self):
        return len(self.annotation_lines)

    def __getitem__(self, index):
        # 用;隔开，类别;图像路径
        annotation_path = self.annotation_lines[index].split(';')[1].split()[0]
        image = Image.open(annotation_path)
        y = int(self.annotation_lines[index].split(';')[0])  # 获得标签0类，1类，2类.....  label
        if self.transform:
            image = self.transform(image)
        image = np.transpose(preprocess_input(np.array(image).astype(np.float32)), [2, 0, 1])  # 归一化
        return image, y



def detection_collate(batch):
    images = []
    targets = []
    for image, y in batch:
        images.append(image)
        targets.append(y)
    images = np.array(images)
    targets = np.array(targets)
    return images, targets
