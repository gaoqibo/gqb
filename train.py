import os
import time
import datetime

import torch

from src import SegNet
from train_utils import train_one_epoch, evaluate, create_lr_scheduler
from my_dataset import DriveDataset
import transforms as T
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset

class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.5 * base_size)
        max_size = int(1.2 * base_size)

        trans = [T.RandomResize(min_size, max_size)]  #随机调整图像的尺寸
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob))
        trans.extend([
            T.RandomCrop(crop_size),#操作用于随机裁剪图像到指定的crop_size尺寸
            T.ToTensor(),#图像转换为张量形式
            T.Normalize(mean=mean, std=std),#操作对图像进行标准化
        ])
        self.transforms = T.Compose(trans)  #Compose是用来组合操作

    def __call__(self, img, target):
        return self.transforms(img, target)#训练阶段的预处理


class SegmentationPresetEval:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)#评估阶段的预处理


def get_transform(train, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    base_size = 565
    crop_size = 480

    if train:#train为布尔值，为真则是用于训练，否则是用于评估
        return SegmentationPresetTrain(base_size, crop_size, mean=mean, std=std)
    else:
        return SegmentationPresetEval(mean=mean, std=std)


def create_model(num_classes):
    # model = SegNet(in_channels=3, out_channels=2)#因为输入为彩色图片所以in_channels被指定为3
    model = SegNet()
    return model#原始的base_c为64，此处设置为了32.可以自己调整（数据集小则base_c只需要较小即可，可以提升训练速度）


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")#判断是否可用 GPU，将模型和数据存储在 CPU 还是 GPU
    batch_size = args.batch_size
    # segmentation nun_classes + background
    num_classes = args.num_classes + 1

    # using compute_mean_std.py
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)

    # 用来保存训练以及验证过程中信息
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    #通过当前的日期和时间保存训练和评估过程中的结果
    train_dataset = DriveDataset(args.data_path,#args.data_path为文件路径
                                 transforms=get_transform(train=True, mean=mean, std=std))

    val_dataset = DriveDataset(args.data_path,
                               transforms=get_transform(train=False, mean=mean, std=std))


    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    #8为预设的最小的 num_workers 值。
    train_loader = torch.utils.data.DataLoader(train_dataset,#torch.utils.data.DataLoader用于创建数据加载器
                                            batch_size=batch_size,
                                            num_workers=num_workers,#用于并行加载数据的进程数
                                            shuffle=True,
                                            pin_memory=True,#将数据加载到 CUDA 固定内存中，可以加速数据传输
                                            collate_fn=train_dataset.collate_fn)#创建训练数据加载器

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=1,
                                            num_workers=num_workers,
                                            pin_memory=True,
                                            collate_fn=val_dataset.collate_fn)#创建验证数据加载器

    model = create_model(num_classes=num_classes)#创建模型对象
    model.to(device)

    params_to_optimize = [p for p in model.parameters() if p.requires_grad]#p.requires_grad是否需要计算梯度
    #params_to_optimize 是一个列表,获取模型中需要进行梯度更新的可训练参数。

    optimizer = torch.optim.SGD(#创建一个随机梯度下降（SGD）优化器对象 optimizer
        params_to_optimize,#传入需要进行梯度更新的参数列表。
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    scaler = torch.cuda.amp.GradScaler() if args.amp else None#是否启用混合精度训练

    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)
    # 创建学习率更新策略，这里是每个step更新一次(不是每个epoch)

    if args.resume:#是否要恢复训练
        checkpoint = torch.load(args.resume, map_location='cpu')#将模型和优化器加载到 CPU
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1#更新训练的起始 epoch。
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])
                #通过恢复之前保存的检查点，可以从之前中断的训练状态继续训练，而无需重新开始。

    best_dice = 0.#初始化最佳的 Dice 系数为 0
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch, num_classes,
                                            lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)
        #train_one_epoch函数爱train_and_eval文件中

        confmat, dice = evaluate(model, val_loader, device=device, num_classes=num_classes)
        #对模型进行验证，并计算 Dice 系数 dice
        val_info = str(confmat)#将混淆矩阵转换为字符串形式
        print(val_info)
        print(f"dice coefficient: {dice:.3f}")
        # write into txt
        with open(results_file, "a") as f:
            # 记录每个epoch对应的train_loss、lr以及验证集各指标，将 train_info 和 val_info 的内容写入结果文件 results_file。
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.6f}\n" \
                         f"dice coefficient: {dice:.3f}\n"
            f.write(train_info + val_info + "\n\n")

        if args.save_best is True:
            if best_dice < dice:
                best_dice = dice# Dice 系数
            else:
                continue

        save_file = {"model": model.state_dict(),#型的状态字典
                     "optimizer": optimizer.state_dict(),#优化器的状态字典
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        if args.amp:#是否采用了混合精度训练
            save_file["scaler"] = scaler.state_dict()#将梯度缩放器的状态字典 scaler.state_dict() 加入到 save_file 字典中

        if args.save_best is True:
            torch.save(save_file, "save_weights/best_model.pth")#将文件保存为 "save_weights/best_model.pth"
        else:
            torch.save(save_file, "save_weights/model_{}.pth".format(epoch))
                #将文件保存为 "save_weights/model_{}.pth".format(epoch)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))#将时间转换为格式化的字符串形式
        print("training time {}".format(total_time_str))


def parse_args():
    import argparse#导入了 argparse 模块，用于解析命令行参数
    parser = argparse.ArgumentParser(description="pytorch unet training")

    parser.add_argument("--data-path", default="./", help="DRIVE root")
    # exclude background
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=8, type=int)
    parser.add_argument("--epochs", default=100, type=int, metavar="N",
                        help="number of total epochs to train")

    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=1, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--save-best', default=True, type=bool, help='only save best dice weights')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    return args


if __name__ == '__main__':#判断当前模块是否为主程序入口点
    args = parse_args()   #args 变量中存储了解析后的命令行参数的值。

    if not os.path.exists("./save_weights"):#是否存在"save_weights" 的文件夹，不存在则创建
        os.mkdir("./save_weights")

    main(args)#将解析后的命令行参数 args 作为参数传递给 main 函数
