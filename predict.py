import os
import time

import torch
from torchvision import transforms
import numpy as np
from PIL import Image

from src import UNet


def time_synchronized():#获取当前时间
    torch.cuda.synchronize() if torch.cuda.is_available() else None#是否使用cuda加速
    return time.time()


def main():
    classes = 1  # exclude background，classes表示类别数（排除背景）
    weights_path = "./save_weights/best_model.pth"  #权重的地址
    img_path = "./DRIVE/test/images/01_test.tif"    #指向了下载的测试集第一个图片
    roi_mask_path = "./DRIVE/test/mask/01_test_mask.gif"#测试图像的ROI掩码文件路径
    assert os.path.exists(weights_path), f"weights {weights_path} not found."#检查权重文件、测试图像文件和ROI掩码文件是否存在
    assert os.path.exists(img_path), f"image {img_path} not found."
    assert os.path.exists(roi_mask_path), f"image {roi_mask_path} not found."

    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#判断是否支持cuda加速
    print("using {} device.".format(device))

    # create model
    model = UNet(in_channels=3, num_classes=classes+1, base_c=32)#创建模型

    # load weights
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])#加载预训练的权重文件
    model.to(device)

    # load roi mask
    roi_img = Image.open(roi_mask_path).convert('L')#打开ROI掩码图像文件，将图像转换为灰度图像（单通道图像）
    roi_img = np.array(roi_img)

    # load image
    original_img = Image.open(img_path).convert('RGB')#打开测试图像文件，并将图像转换为RGB模式

    # from pil image to tensor and normalize
    data_transform = transforms.Compose([transforms.ToTensor(),#转换为张量
                                         transforms.Normalize(mean=mean, std=std)])#归一化操作
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)#将图像进行扩展，从形状为(C, H, W)的张量变为形状为(1, C, H, W)的张量

    model.eval()  # 进入验证模式
    with torch.no_grad():#上下文管理器，以禁用梯度计算和内存管理，从而减少内存消耗
        # init model
        img_height, img_width = img.shape[-2:] #初始化模型
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        output = model(img.to(device))#将输入的图像传递给模型进行推理
        t_end = time_synchronized()
        print("inference time: {}".format(t_end - t_start))#计算时间

        prediction = output['out'].argmax(1).squeeze(0)#argmax(1)找到最大概率的类别索引，然后使用.squeeze(0)压缩批次维度
        prediction = prediction.to("cpu").numpy().astype(np.uint8)
        prediction[prediction == 1] = 255
        # 将前景对应的像素值改成255(白色)
        prediction[roi_img == 0] = 0
        # 将不敢兴趣的区域像素设置成0(黑色)
        mask = Image.fromarray(prediction)
        mask.save("test_result.png")   #保存图像文件


if __name__ == '__main__':
    main()
