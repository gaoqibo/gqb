import os
import time

import cv2
import torch
import torchvision
from torchvision import transforms
import numpy as np
from PIL import Image
from src import UNet


#
# def calculate_iou(mask1, mask2):
#     mask1 = mask1.cv2.IMREAD_GRAYSCALE
#     mask2 = mask2.cv2.IMREAD_GRAYSCALE
#     intersection = cv2.bitwise_and(mask1, mask2)
#     union = cv2.bitwise_or(mask1, mask2)
#     iou = cv2.countNonZero(intersection) / cv2.countNonZero(union)
#     return iou



def zhuanhuan(image_path):
    image = Image.open(image_path)
    pixels = image.load()
    width, height = image.size
    for x in range(width):
        for y in range(height):
            r, g, b = image.getpixel((x, y))
            if r > 100 and g < 100 and b < 100:
                pixels[x, y] = (255, 255, 255)  # 将红色部分替换为白色
    pir_zhuanhuan = image.convert('L')
    pic = np.array(pir_zhuanhuan)     #转换为numpy类型
    return pic



_CLASSNAMES = [
    "bread",
    "cracker",
    "jelly",
    "pudding",
    "soda_water",
    "yogurt",
    "yokan",
]

def retrieve(root: str):
    class_list = {}  # 创建一个字典，其中键是类名，值是列表
    num_list = {}
    pir_name_list = {}
    total_iou = {}
    data_root = os.path.join(root, "DRIVE", "test")
    for classname in _CLASSNAMES:
        total_iou[classname] = []
        class_list[classname] = []
        num_list[classname] = []  # 用来储存每个种类下的1，2，3，4等序号
        pir_name_list[classname] = {}

        data_root_class = os.path.join(data_root, classname)
        for num in os.listdir(data_root_class):
            dir_path = os.path.join(data_root_class, num)  # 包括数字的路径
            if os.path.isdir(dir_path):  # 检查dir_path是否代表一个目录（而不是文件）
                num_list[classname].append(num)
                pir_name_list[classname][num] = []  # 用来储存数字
        for num in num_list[classname]:
            data_root_num = os.path.join(data_root, classname, num)
            mask_path = os.path.join(data_root_num, "mask")

            for pic_name in os.listdir(mask_path):
                if pic_name.endswith('.png'):
                    #pir_name_list[classname][num].append(pic_name)
                    mask_pic_path = os.path.join(mask_path, pic_name)
                    mask = zhuanhuan(mask_pic_path)
                    result_path = os.path.join(data_root_num,"Predicted results",pic_name)
                    result_path = result_path.replace('.png', '.bmp')
                    result =cv2.imread(result_path, cv2.IMREAD_GRAYSCALE)
                    # print(mask_path)
                    # print(result_path)
                    # print(type(mask))
                    # print(type(result))

                    intersection = cv2.bitwise_and(mask, result)

                    intersection_count = cv2.countNonZero(intersection)
                    mask_count = cv2.countNonZero(mask)
                    result_count = cv2.countNonZero(result)
                    iou = intersection_count / float(mask_count + result_count - intersection_count)

                    total_iou[classname].append(iou)
        average_iou = sum(total_iou[classname]) / len(total_iou[classname])
        print(average_iou)

# root = "C:\\Users\\18458\\PycharmProjects\\U-net(cross-validation-tversky)"
root = "C:\\Users\\18458\\PycharmProjects\\U-net-att-tversky"
retrieve(root)