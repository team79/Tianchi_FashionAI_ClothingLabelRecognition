import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import ImageEnhance
from torchsample.transforms import RandomAffine, RandomGamma
from dataloaders.datasets import *
from ulit.ulits import *
from config.configs import *
import pickle

lengthlist = ['coat_length_labels', 'pant_length_labels', 'skirt_length_labels', 'sleeve_length_labels']



def get_loaders(args, input_size):
    if args.task in lengthlist:
        transforms_affine0 = RandomAffine(rotation_range=10)
        transforms_affine1 = RandomAffine(rotation_range=10)
    else:
        transforms_affine0 = RandomAffine(rotation_range=15, zoom_range=(0.85, 1.05))
        transforms_affine1 = RandomAffine(rotation_range=10, zoom_range=(0.9, 1.05))
    transforms_gama = RandomGamma(0.8, 1.2)  # gama变换

    def enhance_transform_8(img):
        rate = 0.8
        brt = ImageEnhance.Brightness(img)
        random_brt = 1 + (np.random.uniform() - 0.5) * rate
        img = brt.enhance(random_brt)

        color = ImageEnhance.Color(img)
        img = color.enhance(1 + (np.random.uniform() - 0.5) * rate)

        contrast = ImageEnhance.Contrast(img)
        random_cts = 1 + (np.random.uniform() - 0.5) * rate
        img = contrast.enhance(random_cts)
        return img

    def enhance_transform_5(img):
        rate = 0.5
        brt = ImageEnhance.Brightness(img)
        random_brt = 1 + (np.random.uniform() - 0.5) * rate  # (0.6, 1.4) uniform
        img = brt.enhance(random_brt)

        color = ImageEnhance.Color(img)
        img = color.enhance(1 + (np.random.uniform() - 0.5) * rate)  # (0.6, 1.4) uniform

        contrast = ImageEnhance.Contrast(img)
        random_cts = 1 + (np.random.uniform() - 0.5) * rate  # x belongs to(0.6, 1.4), p(x<1) = p(x>1)
        img = contrast.enhance(random_cts)
        return img

    img1_list = []
    imgf_list = []
    csvpath1 = 'locate_csv/' + args.task + '_fast.csv'
    dfdata1 = pd.read_csv(csvpath1)
    csvpath2 = 'locate_csv/' + args.task + '_yolo.csv'
    dfdata2 = pd.read_csv(csvpath2)
    if os.path.exists("./imglists/%s_imgsr1.pk" % args.task):
        with open("./imglists/%s_imgsr1.pk" % args.task, 'rb') as f:
            imgsr1 = pickle.load(f)
    else:
        path = os.path.join(data1_fldir, args.task, 'train')
        classes, class_to_idx = find_classes(path)
        imgsr1 = make_datasetwithloc1(path, class_to_idx, dfdata1, dfdata2)
        with open("./imglists/%s_imgsr1.pk" % args.task, 'wb+') as f:
            pickle.dump(imgsr1, f)

    if os.path.exists("./imglists/%s_imgsr2.pk" % args.task):
        with open("./imglists/%s_imgsr2.pk" % args.task, 'rb') as f:
            imgsr2 = pickle.load(f)
    else:
        path = os.path.join(data2_tvdir, args.task, 'train')
        classes, class_to_idx = find_classes(path)
        imgsr2 = make_datasetwithloc1(path, class_to_idx, dfdata1, dfdata2)
        with open("./imglists/%s_imgsr2.pk" % args.task, 'wb+') as f:
            pickle.dump(imgsr2, f)

    img1_list.extend(imgsr1)
    imgf_list.extend(imgsr1)
    imgf_list.extend(imgsr2)


    normalize = transforms.Normalize(mean=mean_dict[args.task], std=mean_dict[args.task])
    train_dataset1 = Train_ImageFolder(
        data_list=imgf_list, args=args,
        transform0=transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.Lambda(enhance_transform_8),
            transforms.ToTensor(),
            transforms_gama,
            transforms_affine0,
            normalize,

        ]),
        transform1=transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.Lambda(enhance_transform_8),
            transforms.ToTensor(),
            transforms_gama,
            transforms_affine1,
            normalize,
        ]),
    )
    train_loader1 = torch.utils.data.DataLoader(
        train_dataset1, batch_size=args.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)

    # train_dataset2 = Train_ImageFolder(
    #     data_type=data_type, args=args,
    #     transform0=transforms.Compose([
    #         transforms.Resize((input_size, input_size)),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.Lambda(enhance_transform_5),
    #         transforms.ToTensor(),
    #         transforms_affine1,
    #         normalize,
    #     ]), )
    #
    # train_loader2 = torch.utils.data.DataLoader(
    #     train_dataset2, batch_size=args.batch_size, shuffle=True,
    #     num_workers=num_workers, pin_memory=True)

    train_dataset3 = Train_ImageFolder(
        data_list=img1_list, args=args,
        transform0=transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.Lambda(enhance_transform_8),
            transforms.ToTensor(),
            transforms_gama,
            transforms_affine0,
            normalize,
        ]),
        transform1=transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.Lambda(enhance_transform_8),
            transforms.ToTensor(),
            transforms_gama,
            transforms_affine1,
            normalize,
        ]),
    )
    train_loader3 = torch.utils.data.DataLoader(
        train_dataset3, batch_size=args.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)

    # train_dataset4 = Train_ImageFolder(
    #     data_type=0, args=args,
    #     transform0=transforms.Compose([
    #         transforms.Resize((input_size, input_size)),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.Lambda(enhance_transform_5),
    #         transforms.ToTensor(),
    #         transforms_affine0,
    #         normalize,
    #     ]),
    #     transform1=transforms.Compose([
    #         transforms.Resize((input_size, input_size)),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.Lambda(enhance_transform_5),
    #         transforms.ToTensor(),
    #         transforms_affine1,
    #         normalize,
    #     ]),
    # )
    # train_loader4 = torch.utils.data.DataLoader(
    #     train_dataset4, batch_size=args.batch_size, shuffle=True,
    #     num_workers=num_workers, pin_memory=True)
    #
    #
    # train_dataset5 = Train_ImageFolder(
    #     data_type=0, args=args,
    #     transform0=transforms.Compose([
    #         transforms.Resize((input_size, input_size)),
    #         # transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         normalize,
    #     ]))
    # train_loader5 = torch.utils.data.DataLoader(
    #     train_dataset5, batch_size=args.batch_size, shuffle=True,
    #     num_workers=num_workers, pin_memory=True)

    valdir = os.path.join(data2_tvdir, args.task, 'val')
    val_loader = torch.utils.data.DataLoader(
        Test_ImageFolder(valdir, args, transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    return train_loader1, None, train_loader3, None, None, val_loader
