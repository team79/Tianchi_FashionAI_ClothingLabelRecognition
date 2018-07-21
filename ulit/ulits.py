import torch
import torch.nn as nn
import datetime
from tensorboard_logger import configure
from config.configs import *

task_dict = {
    'coat_length_labels': 8,  # 上衣
    'collar_design_labels': 5,  # 领子
    'lapel_design_labels': 5,  # 翻领
    'neck_design_labels': 5,  # 脖颈
    'neckline_design_labels': 10,  # 颈线
    'pant_length_labels': 6,  # 裤子
    'skirt_length_labels': 6,  # 裙子
    'sleeve_length_labels': 9,  # 袖子
}

mean_dict = {
    'coat_length_labels': [0.671, 0.636, 0.624],  # 上衣
    'collar_design_labels': [0.646, 0.607, 0.592],  # 衣领
    'lapel_design_labels': [0.633, 0.595, 0.585],  # 翻领
    'neck_design_labels': [0.633, 0.588, 0.568],  # 脖子
    'neckline_design_labels': [0.643, 0.601, 0.584],  # 领口
    'pant_length_labels': [0.652, 0.627, 0.615],  # 裤子
    'skirt_length_labels': [0.648, 0.614, 0.602],  # 裙子
    'sleeve_length_labels': [0.675, 0.637, 0.622],  # 袖子
}
std_dict = {
    'coat_length_labels': [0.099, 0.106, 0.108],  # 上衣
    'collar_design_labels': [0.084, 0.087, 0.088],  # 衣领
    'lapel_design_labels': [0.089, 0.094, 0.095],  # 翻领
    'neck_design_labels': [0.081, 0.084, 0.086],  # 脖子
    'neckline_design_labels': [0.084, 0.087, 0.089],  # 领口
    'pant_length_labels': [0.101, 0.101, 0.101],  # 裤子
    'skirt_length_labels': [0.095, 0.100, 0.101],  # 裙子
    'sleeve_length_labels': [0.098, 0.104, 0.105],  # 袖子
}

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    if epoch == 0:
        lr = 1e-4
    elif epoch <= 2:
        lr = 1e-5
    elif epoch <= 4:
        lr = 1e-6
    elif epoch == 5:
        lr = 5e-7
    elif epoch == 6:
        lr = 1e-7
    elif epoch == 7:
        lr = 5e-8
    else:
        lr = 1e-8

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def myadjust_learning_rate(optimizer, epoch, isbest):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
    if 2 >= epoch:
        if epoch <= 1:
            lr = 1e-4
        elif epoch == 2:
            lr = 1e-5
    else:
        if isbest:
            lr = lr * 0.3
        else:
            lr = lr * 0.1

    if lr < 1e-8:
        lr = 1e-8

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('LR is set to {}'.format(lr))
    return lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def log_config(args):
    time = datetime.datetime.now()
    time_str = '%s' % time
    filename = args.task + '_' + args.arch + '-' + time_str
    configure("runs/" + filename, flush_secs=5)

def load_bestmodel():
    model = torch.load("checkpoint/" + "resnet152_skirt_length_labels_best.pkl")
    model.cuda()
    if curremtmachine != 0:
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
    return model
