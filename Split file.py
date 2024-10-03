import os
import random
import shutil

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

def split_file(root,split_ratio = 0.8):
        class_list ={}                     #创建一个字典，其中键是类名，值是列表
        num_list = {}
        pir_name_list = {}
        for classname in _CLASSNAMES:
            class_list[classname] = []
            num_list[classname] = []       #用来储存每个种类下的1，2，3，4等序号
            pir_name_list[classname] = {}
            data_root_class = os.path.join(root, "FeedSwallowingDataset", classname)
            for num in os.listdir(data_root_class):
                dir_path = os.path.join(data_root_class,num)  #包括数字的路径
                if os.path.isdir(dir_path):  #检查dir_path是否代表一个目录（而不是文件）
                    num_list[classname].append(num)
                    pir_name_list[classname][num] = []    #用来储存数字
            for num in num_list[classname] :
                data_root_num = os.path.join(root, "FeedSwallowingDataset",classname,num)
                mask_path = os.path.join(data_root_num, "mask")

                for pic_name in os.listdir(mask_path):
                    if pic_name.endswith('.png'):
                        pir_name_list[classname][num].append(pic_name)
                # print(pir_name_list[classname][num])
                # print("1")

                total_elements = len(pir_name_list[classname][num])
                split_index = int(total_elements * split_ratio)
                random.shuffle(pir_name_list[classname][num])
                training_num = pir_name_list[classname][num][:split_index]  # 训练集图片名
                testing_num = pir_name_list[classname][num][split_index:]  # 测试集图片名

                # print(training_num)
                # print(testing_num)
                # print("1")
                #以上代码没问题


                for num1 in training_num :

                    source_path_mask = os.path.join(mask_path, num1)   #mask的路径
                    destination_mask =  os.path.join(root, "DRIVE", "training",classname,num, "mask")  #复制到的新文件夹路径
                    if not os.path.exists(destination_mask):
                        os.makedirs(destination_mask)
                    shutil.copy2(source_path_mask, destination_mask)

                    source_path_img = os.path.join(data_root_num, "img", num1)
                    source_path_img = source_path_img.replace('.png', '.bmp')
                    destination_img =  os.path.join(root, "DRIVE", "training", classname,num, "img")  #复制到的新文件夹路径
                    if not os.path.exists(destination_img):
                        os.makedirs(destination_img)
                    shutil.copy2(source_path_img, destination_img)

                for num2 in testing_num :

                            #以下为test路径
                    source_path_mask2 = os.path.join(mask_path, num2)   #mask的路径
                    destination_mask2 =  os.path.join(root, "DRIVE", "test",classname, num,"mask")  #复制到的新文件夹路径
                    if not os.path.exists(destination_mask2):
                        os.makedirs(destination_mask2)
                    shutil.copy2(source_path_mask2, destination_mask2)

                    source_path_img2 = os.path.join(data_root_num, "img", num2)
                    source_path_img2 = source_path_img2.replace('.png', '.bmp')
                    destination_img2 =  os.path.join(root, "DRIVE", "test",classname,num, "img")  #复制到的新文件夹路径
                    if not os.path.exists(destination_img2):
                        os.makedirs(destination_img2)
                    shutil.copy2(source_path_img2, destination_img2)

root = "C:\\Users\\18458\\PycharmProjects\\U-net-att-tversky"
split_file(root,split_ratio = 0.8)

