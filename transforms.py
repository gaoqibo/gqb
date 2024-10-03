import numpy as np
import random

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


def pad_if_smaller(img, size, fill=0):#img表示输入的图像，size表示给定的尺寸，fill表示填充的数值（默认为0）
    # 如果图像最小边长小于给定size，则用数值fill进行padding
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img#这个函数的作用是检查图像的最小边长是否小于给定尺寸，如果小于则进行填充操作


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target#将多个图像转换操作组合在一起


class RandomResize(object):#随机调整输入图像和目标的尺寸大小
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size#如果max_size没有提供，则默认与min_size相同

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)#生成一个随机整数size在最大与最小之间
        image = F.resize(image, size)
        # 这里size传入的是int类型，所以是将图像的最小边长缩放到size大小
        target = F.resize(target, size, interpolation=T.InterpolationMode.NEAREST)
        # 这里的interpolation注意下，在torchvision(0.9.0)以后才有InterpolationMode.NEAREST
        # 如果是之前的版本需要使用PIL.Image.NEAREST
        return image, target


class RandomHorizontalFlip(object):#以给定的概率随机水平翻转输入的图像和目标
    def __init__(self, flip_prob): #flip_prob，表示水平翻转的概率
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:#是否反转
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomVerticalFlip(object):#以给定的概率随机垂直翻转输入的图像和目标
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.vflip(image)
            target = F.vflip(target)
        return image, target


class RandomCrop(object):#随机裁剪输入的图像和目标到指定的尺寸大小
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


class CenterCrop(object):#对输入的图像和目标进行中心裁剪到指定的尺寸大小
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class ToTensor(object):#将输入的图像和目标转换为张量形式
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


class Normalize(object):#对输入的图像进行归一化操作
    def __init__(self, mean, std):#两个参数mean和std，分别表示归一化的均值和标准差
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target
