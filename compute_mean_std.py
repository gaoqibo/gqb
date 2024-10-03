import os
from PIL import Image
import numpy as np


def main():#计算给定图像数据集的通道均值和标准差
    img_channels = 3#彩色图像通道数为3
    img_dir = "./DRIVE/training/images"#表示图像文件的目录路径
    roi_dir = "./DRIVE/training/mask"  #roi_dir表示图像区域兴趣（ROI）掩码文件的目录路径。
    assert os.path.exists(img_dir), f"image dir: '{img_dir}' does not exist." #检查目录是否存在，不存在则报错
    assert os.path.exists(roi_dir), f"roi dir: '{roi_dir}' does not exist."

    img_name_list = [i for i in os.listdir(img_dir) if i.endswith(".tif")]
    #获取img_dir目录下所有以".tif"结尾的图像文件列表，并存储在img_name_list中
    cumulative_mean = np.zeros(img_channels)#创建了两个空数组，用于累计计算图像数据集的通道均值和标准差
    cumulative_std = np.zeros(img_channels)
    for img_name in img_name_list:
        img_path = os.path.join(img_dir, img_name)
        ori_path = os.path.join(roi_dir, img_name.replace(".tif", "_mask.gif"))
        img = np.array(Image.open(img_path)) / 255.
        roi_img = np.array(Image.open(ori_path).convert('L'))

        img = img[roi_img == 255]
        cumulative_mean += img.mean(axis=0)
        cumulative_std += img.std(axis=0)

    mean = cumulative_mean / len(img_name_list)
    std = cumulative_std / len(img_name_list)
    print(f"mean: {mean}")
    print(f"std: {std}")


if __name__ == '__main__':  #判断当前脚本是否被直接执行
    main()
