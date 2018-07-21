import argparse, time
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from tensorboard_logger import log_value
from dataloaders.dataloader import *
from model.getmodel import *
from torchsample.transforms import *
from config.configs import *
import csv


parser = argparse.ArgumentParser(description='PyTorch FashionAI Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='se_resnext101_32x4d')
parser.add_argument('--epochs', default=13, type=int, metavar='N')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 8)')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--task', default='skirt_length_labels', type=str,
                    help='name of the classification task')

torch.set_default_tensor_type('torch.cuda.FloatTensor')

best_prec1 = 0
val_count = 0
is_best = 0
def main():
    global args, best_prec1, is_best
    args = parser.parse_args()
    print("task:", args.task)

    model, inputsize = get_model(args)
    model = model.cuda()
    if curremtmachine != 0:
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    log_config(args)
    csv_filename = './acc_results/' + args.arch + '_' + args.task + '.csv'
    csvFile = open(csv_filename, 'w', newline='')
    writer = csv.writer(csvFile)

    train_loader, _, train_loader3, _, _, val_loader = get_loaders(args, inputsize)
    for epoch in range(args.epochs):
        lr = myadjust_learning_rate(optimizer, epoch, is_best)

        if epoch == 7:
            train_loader = train_loader3
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-5
            model.load_state_dict(torch.load(os.path.join('./checkpoint', args.arch + '_' + args.task + '_bestparams.pkl')))
        # if epoch == 11:
        #     train_loader = train_loader4
        #     model.load_state_dict(torch.load(os.path.join('./checkpoint', args.arch + '_' + args.task + '_bestparams.pkl')))
        # if epoch == 12: #no use
        #     train_loader = train_loader5
        #     model.load_state_dict(torch.load(os.path.join('./checkpoint', args.arch + '_' + args.task + '_bestparams.pkl')))




        train(train_loader, model, criterion, optimizer, epoch)
        prec1, _ = validate(val_loader, model, criterion)

        is_best = bool(prec1 > best_prec1)
        best_prec1 = max(prec1, best_prec1)
        torch.save(model, os.path.join('./checkpoint', args.arch + '_' + args.task + '_.pkl'))
        log_value('lr', lr, step=epoch)
        if is_best:
            torch.save(model, os.path.join('./checkpoint', args.arch + '_' + args.task + '_best.pkl'))
            torch.save(model.state_dict(), os.path.join('./checkpoint', args.arch + '_' + args.task + '_bestparams.pkl'))
        writer.writerow((args.arch + '_' + args.task, prec1))
    csvFile.close()


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()

    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input = input.cuda(async=True)
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        output = model(input_var)
        loss = criterion(output, target_var)

        prec1, prec2 = accuracy(output.data, target, topk=(1, 2))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top2.update(prec2[0], input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))
    log_value('train_loss', losses.avg, step=epoch)
    log_value('train_acc', top1.avg, step=epoch)


def validate(val_loader, model, criterion):
    global val_count
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()

    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input = input.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        output = model(input_var)
        loss = criterion(output, target_var)

        prec1, prec2 = accuracy(output.data, target, topk=(1, 2))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top2.update(prec2[0], input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1))

    print(' * Acc@1 {top1.avg:.3f}'
          .format(top1=top1))
    log_value('val_loss', losses.avg, step=val_count)
    log_value('val_acc', top1.avg, step=val_count)
    val_count = val_count + 1
    return top1.avg, top2.avg


if __name__ == '__main__':
    main()
