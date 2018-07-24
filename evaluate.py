import os, argparse, time
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from ulit.ulits import *
from PIL import Image
import numpy as np
from model.model import *
import math
from config import *
from  train import validate


parser = argparse.ArgumentParser(description='PyTorch FashionAI Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet152 ')
parser.add_argument('--epochs', default=30, type=int, metavar='N')
parser.add_argument('-b', '--batch-size', default=42, type=int,
                    metavar='N', help='mini-batch size (default: 48)')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--task', default='neck_design_labels', type=str,
                    help='name of the classification task')
parser.add_argument('--usepretrain', '-u', default=1, type=int,
                    help='use pre trained model')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

CUDDEVICE=[0,1,2,3,4,5]
def progressbar(i, n, bar_len=40):
    percents = math.ceil(100.0 * i / float(n))
    filled_len = int(round(bar_len * i / float(n)))
    prog_bar = '=' * filled_len + '-' * (bar_len - filled_len)
    print('[%s] %s%s' % (prog_bar, percents, '%'), end = '\r')

def predict(arc):
    args = parser.parse_args()
    args.arch = arc
    f_out = open('submission/round2_%s_log_submission.csv' % (arc), 'w')
    for i in range(len(task_list)):
        task = task_list[i]
        args.task = task
        model, input_size = get_model(args)
        input_size = 512
        model = nn.DataParallel(model, device_ids=CUDDEVICE).cuda()

        checkpoint_name = os.path.join('./round2_checkpoint', arc + '_' + task + '_best.pkl')
        checkpoint = torch.load(checkpoint_name)

        # print(task + ' : ' + str(checkpoint['best_prec1']))
        # continue

        model.module.load_state_dict(checkpoint['state_dict'])
        #model.cpu()

        criterion = nn.CrossEntropyLoss().cuda()


        #model = model.cuda()
        model.eval()

        # if arc == 'resnet50' or arc =='inceptionv4' or arc =='resnet152':
        #    # model._modules.module.module.fc = model._modules.module.module.last_linear
        #     model.module.module.fc = model.module.module.last_linear
        # if arc == 'densenet169':
        #     model.classifier = model.last_linear
        # if arc == 'xception':
        #     model.last_linear = model.fc

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            normalize,
        ])

        # f_out = open('submission/%s_%s.csv' % (arc, task), 'w')
        with open('data/week-rank/Tests/question.csv', 'r') as f_in:
            lines = f_in.readlines()
        tokens = [l.rstrip().split(',') for l in lines]
        task_tokens = [t for t in tokens if t[1] == task]
        n = len(task_tokens)
        cnt = 0

        for path, task, _ in task_tokens:
            img_path = os.path.join('data/week-rank', path)
            img = Image.open(img_path)
            data = transform(img)
            data = data.unsqueeze_(0)
            data = data.cuda(async=True)
            data = Variable(data, volatile=True)

            output = model(data)
           # softmax = torch.nn.Softmax(dim=-1)
            softmax = torch.nn.LogSoftmax(dim=-1)
            output = softmax(output)
            output = output.view(-1)
            output = output.cpu().data.numpy().tolist()
            print(output, path)
            pred_out = ';'.join(["%.8f" % (o) for o in output])
            line_out = ','.join([path, task, pred_out])
            f_out.write(line_out + '\n')
            cnt += 1
            progressbar(cnt, n)
    f_out.close()

task_list = [
    'neck_design_labels',
    'coat_length_labels',
    'collar_design_labels',
    'lapel_design_labels',
    'neckline_design_labels',
    'pant_length_labels',
    'skirt_length_labels',
    'sleeve_length_labels'
]
arc_list = [
    ## 'xception',
   # 'densenet169',
    #'inceptionresnetv2',
    # 'inceptionv4',
    'resnet152',
    #'resnet18',
    # 'resnet50'
]


for arc in arc_list:
    predict(arc)

