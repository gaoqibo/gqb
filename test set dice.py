import os
import cv2
import numpy as np
from PIL import Image


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
    pic = np.array(pir_zhuanhuan)  # 转换为 numpy 类型
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
    total_dice = []  # 用来存储所有类别的 Dice 系数
    data_root = os.path.join(root, "DRIVE", "test")

    for classname in _CLASSNAMES:
        data_root_class = os.path.join(data_root, classname)
        for num in os.listdir(data_root_class):
            dir_path = os.path.join(data_root_class, num)
            if os.path.isdir(dir_path):
                data_root_num = os.path.join(data_root, classname, num)
                mask_path = os.path.join(data_root_num, "mask")

                for pic_name in os.listdir(mask_path):
                    if pic_name.endswith('.png'):
                        mask_pic_path = os.path.join(mask_path, pic_name)
                        mask = zhuanhuan(mask_pic_path)
                        result_path = os.path.join(data_root_num, "Predicted results", pic_name)
                        result_path = result_path.replace('.png', '.bmp')
                        result = cv2.imread(result_path, cv2.IMREAD_GRAYSCALE)

                        intersection = cv2.bitwise_and(mask, result)
                        intersection_count = cv2.countNonZero(intersection)
                        mask_count = cv2.countNonZero(mask)
                        result_count = cv2.countNonZero(result)

                        dice = (2 * intersection_count) / float(mask_count + result_count)
                        total_dice.append(dice)

    average_dice = sum(total_dice) / len(total_dice)
    print(average_dice)


root = "C:\\Users\\18458\\PycharmProjects\\U-net-att-tversky"
retrieve(root)
