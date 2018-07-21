import torch.utils.data as data
import os
import os.path
from .augs import *
from config.configs import *
from tqdm import tqdm
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


def make_mydataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    img = pil_loader(root + '/' + fname)
                    loc = (0, 0, img.width, img.height)
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target], loc)
                    images.append(item)

    return images


# 重采样数据平衡
def make_dataset1(dir, class_to_idx):
    images = []
    idx_list = []
    image_list = [[] for i in range(len(class_to_idx))]
    classcounts = [0] * len(class_to_idx)
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    image_list[class_to_idx[target]].append(item)
                    classcounts[class_to_idx[target]] += 1
                    idx_list.append(class_to_idx[target])
    for i in range(len(class_to_idx)):
        images.extend(image_list[i])

    max_classcounts = max(classcounts)
    for i in range(len(class_to_idx)):
        while classcounts[i] < max_classcounts:
            idx = np.random.randint(0, len(image_list[i]))
            item = image_list[i][idx]
            images.append(item)
            classcounts[i] += 1
            idx_list.append(class_to_idx[str(item[1])])
    return images, idx_list


# 带位置数据
def make_datasetwithloc(dir, class_to_idx, dfdata):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                get_index = list(dfdata['image_name']).index(fname)
                location = list(dfdata.loc[get_index][1:5])

                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target], location)
                    images.append(item)
    return images


# 带位置数据 yolo+fasterrcnn
def make_datasetwithloc1(dir, class_to_idx, dfdata, dfdata1):
    images = []
    dir = os.path.expanduser(dir)
    for target in tqdm(sorted(os.listdir(dir))):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in (sorted(os.walk(d))):
            for fname in sorted(fnames):
                get_index = list(dfdata['image_name']).index(fname)
                location = list(dfdata.loc[get_index][1:5])

                get_index = list(dfdata1['image_name']).index(fname)
                location1 = list(dfdata1.loc[get_index][1:5])

                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target], location, location1)
                    images.append(item)
    return images


# 带位置数据+重采样数据平衡
def make_datasetwithlocandbalance(dir, class_to_idx, dfdata):
    images = []
    classcounts = [0] * len(class_to_idx)
    image_list = [[] for i in range(len(class_to_idx))]
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                get_index = list(dfdata['image_name']).index(fname)
                location = list(dfdata.loc[get_index][1:5])

                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target], location)
                    image_list[class_to_idx[target]].append(item)
                    classcounts[class_to_idx[target]] += 1

    for i in range(len(class_to_idx)):
        images.extend(image_list[i])

    max_classcounts = max(classcounts)
    for i in range(1, len(class_to_idx)):
        while classcounts[i] < max_classcounts:
            idx = np.random.randint(0, len(image_list[i]))
            item = image_list[i][idx]
            images.append(item)
            classcounts[i] += 1
    return images


# 从数据1中获取采样数据
def make_datasetwithlocandbalance1(dir1, dir2, class_to_idx, dfdata):
    images = []
    classcounts1 = [0] * len(class_to_idx)
    image_list1 = [[] for i in range(len(class_to_idx))]
    dir1 = os.path.expanduser(dir1)
    for target in sorted(os.listdir(dir1)):
        d = os.path.join(dir1, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                get_index = list(dfdata['image_name']).index(fname)
                location = list(dfdata.loc[get_index][1:5])

                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target], location)
                    image_list1[class_to_idx[target]].append(item)
                    classcounts1[class_to_idx[target]] += 1

    classcounts2 = [0] * len(class_to_idx)
    image_list2 = [[] for i in range(len(class_to_idx))]
    dir2 = os.path.expanduser(dir2)
    for target in sorted(os.listdir(dir2)):
        d = os.path.join(dir2, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                get_index = list(dfdata['image_name']).index(fname)
                location = list(dfdata.loc[get_index][1:5])

                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target], location)
                    image_list2[class_to_idx[target]].append(item)
                    classcounts2[class_to_idx[target]] += 1

    for i in range(len(class_to_idx)):
        images.extend(image_list1[i])

    max_classcounts = max(classcounts1)
    for i in range(1, len(class_to_idx)):
        if classcounts1[i] < max_classcounts:
            if (classcounts1[i] + classcounts2[i]) < max_classcounts:
                images.extend(image_list2[i])
            else:
                imgs = random.shuffle(image_list2[i])
                images.extend(image_list2[i][:(max_classcounts - classcounts1[i])])
    return images


def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class Train_ImageFolder(data.Dataset):
    def __init__(self, data_list=None, args=None, transform0=None, transform1=None, loader=default_loader):
        # csvpath1 = 'locate_csv/' + args.task + '_fast.csv'
        # dfdata1 = pd.read_csv(csvpath1)
        # csvpath2 = 'locate_csv/' + args.task + '_yolo.csv'
        # dfdata2 = pd.read_csv(csvpath2)
        # imgs = []
        # if data_type == 0:  # data2
        #     path = os.path.join(data2_tvdir, args.task, 'train')
        #     classes, class_to_idx = find_classes(path)
        #     imgs0 = make_datasetwithloc1(path, class_to_idx, dfdata1, dfdata2)
        #     imgs.extend(imgs0)
        # elif data_type == 1:  # data12
        #     path = os.path.join(data2_tvdir, args.task, 'train')
        #     classes, class_to_idx = find_classes(path)
        #     imgs0 = make_datasetwithloc1(path, class_to_idx, dfdata1, dfdata2)
        #     imgs.extend(imgs0)
        #     path = os.path.join(data1_fldir, args.task, 'train')
        #     classes, class_to_idx = find_classes(path)
        #     imgs1 = make_datasetwithloc1(path, class_to_idx, dfdata1, dfdata2)
        #     imgs.extend(imgs1)

        self.imgs = data_list
        self.transform0 = transform0
        self.transform1 = transform1
        self.loader = loader
        self.data_type = data_type
        self.task = args.task

    def __getitem__(self, index):
        path, target, loc1, loc2 = self.imgs[index]
        img = self.loader(path)

        img_noneloc = (0, 0, img.width, img.height)

        if img_noneloc == loc1 and img_noneloc == loc2:
            pass
        if img_noneloc == loc1 and img_noneloc != loc2:
            loc = loc2
        elif img_noneloc != loc1 and img_noneloc == loc2:
            loc = loc1
        else:
            loc = [min(loc1[0], loc2[0]), min(loc1[1], loc2[1]), max(loc1[2], loc2[2]), max(loc1[3], loc2[3])]

        loc = loc_(loc, img)

        if self.transform1 is None:
            if loc == img_noneloc:
                img = padwithnolocrnd(img, self.task)
                img = self.transform0(img)
            else:
                img = padwithlocrnd(img, loc, self.task)
                img = self.transform0(img)
        else:
            if loc == img_noneloc:
                img = padwithnolocrnd(img, self.task)
                img = self.transform0(img)
            else:
                img = padwithlocrnd(img, loc, self.task)
                img = self.transform1(img)

        return img, target

    def __len__(self):
        return len(self.imgs)


class Test_ImageFolder(data.Dataset):
    def __init__(self, root, args, transform=None, loader=default_loader):
        csvpath1 = 'locate_csv/' + args.task + '_fast.csv'
        dfdata1 = pd.read_csv(csvpath1)
        csvpath2 = 'locate_csv/' + args.task + '_yolo.csv'
        dfdata2 = pd.read_csv(csvpath2)
        classes, class_to_idx = find_classes(root)
        imgs = make_datasetwithloc1(root, class_to_idx, dfdata1, dfdata2)

        self.imgs = imgs
        self.transform = transform
        self.loader = loader
        self.task = args.task

    def __getitem__(self, index):
        path, target, loc1, loc2 = self.imgs[index]
        img = self.loader(path)

        img_noneloc = (0, 0, img.width, img.height)

        if img_noneloc == loc1 and img_noneloc == loc2:
            pass
        if img_noneloc == loc1 and img_noneloc != loc2:
            loc = loc2
        elif img_noneloc != loc1 and img_noneloc == loc2:
            loc = loc1
        else:
            loc = [min(loc1[0], loc2[0]), min(loc1[1], loc2[1]), max(loc1[2], loc2[2]), max(loc1[3], loc2[3])]

        loc = loc_(loc, img)

        if loc == img_noneloc:
            img = padwithnoloc_T(img, self.task)
            img = self.transform(img)
        else:
            img = padwithloc_T(img, loc, self.task)
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.imgs)