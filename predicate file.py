import os
import time
import torch
import torchvision
from torchvision import transforms
import numpy as np
from PIL import Image
from src import UNet

def time_synchronized():#获取当前时间
    torch.cuda.synchronize() if torch.cuda.is_available() else None#是否使用cuda加速
    return time.time()



_CLASSNAMES = [
    "bread",
    "cracker",
    "jelly",
    "pudding",
    "soda_water",
    "yogurt",
    "yokan",
]
def main(img_path,output_path):
    classes = 1  # exclude background，classes表示类别数（排除背景）
    weights_path = "./save_weights/best_model.pth"  #权重的地址
    rec = Image.new('L', (580, 500), 0)  # 创建一个矩形的灰度图，只关注矩形内白色部分
    for x in range(65, 541):  # 使得中心部像素值为0
        for y in range(125, 328):
            if 0 <= x < 580 and 0 <= y < 500:
                rec.putpixel((x, y), 255)



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
    roi_img = rec.convert('L')
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
        transform = transforms.ToTensor()
        tensor_mask = transform(mask)
        torchvision.utils.save_image(tensor_mask, output_path)

_CLASSNAMES = [
    "bread",
    "cracker",
    "jelly",
    "pudding",
    "soda_water",
    "yogurt",
    "yokan",
]


# _CLASSNAMES = [
#     "empty",
# ]
def retrieve(root: str):
    class_list = {}  # 创建一个字典，其中键是类名，值是列表
    num_list = {}
    pir_name_list = {}
    data_root = os.path.join(root, "DRIVE", "test")
    for classname in _CLASSNAMES:
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
            img_path = os.path.join(data_root_num, "img")

            for pic_name in os.listdir(img_path):
                if pic_name.endswith('.bmp'):
                    pir_name_list[classname][num].append(pic_name)
                    img_path = os.path.join(data_root_num, "img", pic_name)

                    output_file_path = os.path.join(data_root,classname,num,"Predicted results")
                    if not os.path.exists(output_file_path):
                        os.makedirs(output_file_path)
                    output_path =os.path.join(output_file_path, pic_name)
                    main(img_path, output_path)

# output_path = "C:\\Users\\18458\\OneDrive\\桌面\\12\\045.bmp"
# img_path = "C:\\Users\\18458\\PycharmProjects\\U-net(bilibili)copy\\DRIVE\\test\\bread\\1\\img\\045.bmp"
# main(img_path,output_path)
root = "C:\\Users\\18458\\PycharmProjects\\U-net(att-tversky）"
retrieve(root)