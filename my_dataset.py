import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

_CLASSNAMES = [
    "bread",
    "cracker",
    "empty",
    "jelly",
    "pudding",
    "soda_water",
    "yogurt",
    "yokan",
]
class DriveDataset(Dataset):
    def __init__(self, root: str, transforms=None):
        super(DriveDataset, self).__init__()
        class_list ={}                     #创建一个字典，其中键是类名，值是列表
        num_list = {}
        pir_name_list = {}
        all_mask_path = []
        all_img_path = []
        data_root = os.path.join(root, "DRIVE", "training")
        for classname in _CLASSNAMES:
            class_list[classname] = []
            num_list[classname] = []       #用来储存每个种类下的1，2，3，4等序号
            pir_name_list[classname] = {}

            data_root_class = os.path.join(data_root, classname)
            for num in os.listdir(data_root_class):
                dir_path = os.path.join(data_root_class,num)  #包括数字的路径
                if os.path.isdir(dir_path):  #检查dir_path是否代表一个目录（而不是文件）
                    num_list[classname].append(num)
                    pir_name_list[classname][num] = []    #用来储存数字
            for num in num_list[classname] :
                data_root_num = os.path.join(data_root,classname,num)
                self.transforms = transforms
                mask_path = os.path.join(data_root_num, "mask")

                for pic_name in os.listdir(mask_path):
                    if pic_name.endswith('.png'):
                        pir_name_list[classname][num].append(pic_name)
                        all_mask_path.append(os.path.join(mask_path, pic_name))
                        img_path = os.path.join(data_root_num, "img", pic_name)
                        img_path = img_path.replace('.png', '.bmp')
                        all_img_path.append(img_path)

        self.mask  = all_mask_path
        self.img = all_img_path


    #以下为添加
    def zhuanhuan(self, image_path):
        image = Image.open(image_path)
        pixels = image.load()
        width, height = image.size
        for x in range(width):
            for y in range(height):
                r, g, b = image.getpixel((x, y))
                if r > 100 and g < 100 and b < 100:
                    pixels[x, y] = (255, 255, 255)  # 将红色部分替换为白色
        pir_zhuanhuan = image.convert('L')
        return pir_zhuanhuan


        #以上为添加

    def __getitem__(self, idx):
        rec = Image.new('L', (580, 500), 255)  #创建一个矩形的灰度图，只关注矩形内白色部分
        for x in range(100, 100):              #使得中心部像素值为0
            for y in range(100, 100):
                if 0 <= x < 580 and 0 <= y < 500:
                    rec.putpixel((x, y), 0)
        img = Image.open(self.img[idx]).convert('RGB')
        #mask = Image.open(self.mask[idx])
        mask = self.zhuanhuan(self.mask[idx])
        mask = np.array(mask) / 255
        rec = 255 - np.array(rec)
        mask = np.clip(mask + rec, a_min=0, a_max=255)
        mask = Image.fromarray(mask)

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask
    def __len__(self):
        return len(self.img)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs
