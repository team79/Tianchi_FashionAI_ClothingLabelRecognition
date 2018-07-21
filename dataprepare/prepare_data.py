# coding=utf-8
import os, shutil, random

data_home = '/home/kaka/FASHIONAI_DATA'

# 生成文件夹
# 第一轮数据_所有
trainandvaldir_1_full = data_home + '/train_valid_1_full'
# 第二轮数据_所有
trainandvaldir_2_full = data_home + '/train_valid_2_full'
# 第一轮数据_已分
trainandvaldir_1_trainval = data_home + '/train_valid_1_trainval'
# 第二轮数据_已分
trainandvaldir_2_trainval = data_home + '/train_valid_2_trainval'


# 第一轮数据
basedir = data_home + '/base'
label1_dir = data_home + '/base/Annotations/label.csv'
# 第二轮数据
round2imagedir = data_home + '/round2_data/train'
label2_dir = data_home + '/round2_data/train/Annotations/label.csv'
# 热身数据
warmup_label_dir = data_home + '/web/Annotations/skirt_length_labels.csv'
webdir = data_home + '/web'
# 评测1数据
test_a = data_home + '/rank'
testb_label_dir = data_home + '/z_rank/Annotations/fashionAI_attributes_answer_b_20180428.csv'
# 评测2数据
test_b = data_home + '/z_rank'
testa_label_dir = data_home + '/rank/Annotations/fashionAI_attributes_answer_a_20180428.csv'

def mkdir_if_not_exist(path):
    if not os.path.exists(os.path.join(*path)):
        os.makedirs(os.path.join(*path))

def gen_1_full():
    # round 1 data
    trainandvaldir = trainandvaldir_1_full
    label_dict = {'coat_length_labels': [],
                  'lapel_design_labels': [],
                  'neckline_design_labels': [],
                  'skirt_length_labels': [],
                  'collar_design_labels': [],
                  'neck_design_labels': [],
                  'pant_length_labels': [],
                  'sleeve_length_labels': []}

    task_list = label_dict.keys()

    mkdir_if_not_exist([trainandvaldir])

    with open(label1_dir, 'r') as f:
        lines = f.readlines()
        tokens = [l.rstrip().split(',') for l in lines]
        for path, task, label in tokens:
            label_dict[task].append((path, label))

    for task, path_label in label_dict.items():
        mkdir_if_not_exist([trainandvaldir, task])
        train_count = 0
        n = len(path_label)
        m = len(list(path_label[0][1]))

        for mm in range(m):
            mkdir_if_not_exist([trainandvaldir, task, 'train', str(mm)])
            mkdir_if_not_exist([trainandvaldir, task, 'val', str(mm)])

        random.shuffle(path_label)
        for path, label in path_label:
            label_index = list(label).index('y')
            src_path = os.path.join(basedir, path)
            shutil.copy(src_path,
                            os.path.join(trainandvaldir, task, 'train', str(label_index)))
            train_count += 1
    print(1, train_count)

    # Add warmup data to skirt task
    label_dict = {'skirt_length_labels': []}

    with open(warmup_label_dir, 'r') as f:
        lines = f.readlines()
        tokens = [l.rstrip().split(',') for l in lines]
        for path, task, label in tokens:
            label_dict[task].append((path, label))

    for task, path_label in label_dict.items():
        train_count = 0
        n = len(path_label)
        m = len(list(path_label[0][1]))

        random.shuffle(path_label)
        for path, label in path_label:
            label_index = list(label).index('y')
            src_path = os.path.join(webdir, path)

            shutil.copy(src_path,
                            os.path.join(trainandvaldir, task, 'train', str(label_index)))

            train_count += 1
    print(3, train_count)

    # test a
    label_dict = {'coat_length_labels': [],
                  'lapel_design_labels': [],
                  'neckline_design_labels': [],
                  'skirt_length_labels': [],
                  'collar_design_labels': [],
                  'neck_design_labels': [],
                  'pant_length_labels': [],
                  'sleeve_length_labels': []}

    task_list = label_dict.keys()

    with open(testa_label_dir, 'r') as f:
        lines = f.readlines()
        tokens = [l.rstrip().split(',') for l in lines]
        for path, task, label in tokens:
            label_dict[task].append((path, label))

    for task, path_label in label_dict.items():
        mkdir_if_not_exist([trainandvaldir, task])
        train_count = 0
        n = len(path_label)
        m = len(list(path_label[0][1]))

        for mm in range(m):
            mkdir_if_not_exist([trainandvaldir, task, 'train', str(mm)])
            mkdir_if_not_exist([trainandvaldir, task, 'val', str(mm)])

        random.shuffle(path_label)
        for path, label in path_label:
            label_index = list(label).index('y')
            src_path = os.path.join(test_a, path)
            #if train_count < n * 0.95:
            shutil.copy(src_path,
                            os.path.join(trainandvaldir, task, 'train', str(label_index)))
            #else:
            #    shutil.copy(src_path,
             #               os.path.join(trainandvaldir, task, 'val', str(label_index)))
            train_count += 1
    print(4, train_count)

    # test b
    label_dict = {'coat_length_labels': [],
                  'lapel_design_labels': [],
                  'neckline_design_labels': [],
                  'skirt_length_labels': [],
                  'collar_design_labels': [],
                  'neck_design_labels': [],
                  'pant_length_labels': [],
                  'sleeve_length_labels': []}

    task_list = label_dict.keys()

    with open(testb_label_dir, 'r') as f:
        lines = f.readlines()
        tokens = [l.rstrip().split(',') for l in lines]
        for path, task, label in tokens:
            label_dict[task].append((path, label))

    for task, path_label in label_dict.items():
        mkdir_if_not_exist([trainandvaldir, task])
        train_count = 0
        n = len(path_label)
        m = len(list(path_label[0][1]))

        for mm in range(m):
            mkdir_if_not_exist([trainandvaldir, task, 'train', str(mm)])
            mkdir_if_not_exist([trainandvaldir, task, 'val', str(mm)])

        random.shuffle(path_label)
        for path, label in path_label:
            label_index = list(label).index('y')
            src_path = os.path.join(test_b, path)
            shutil.copy(src_path,
                            os.path.join(trainandvaldir, task, 'train', str(label_index)))

            train_count += 1
    print(5, train_count)

def gen_1_trainval():
    # round 1 data
    trainandvaldir = trainandvaldir_1_trainval
    label_dict = {'coat_length_labels': [],
                  'lapel_design_labels': [],
                  'neckline_design_labels': [],
                  'skirt_length_labels': [],
                  'collar_design_labels': [],
                  'neck_design_labels': [],
                  'pant_length_labels': [],
                  'sleeve_length_labels': []}

    task_list = label_dict.keys()

    mkdir_if_not_exist([trainandvaldir])

    with open(label1_dir, 'r') as f:
        lines = f.readlines()
        tokens = [l.rstrip().split(',') for l in lines]
        for path, task, label in tokens:
            label_dict[task].append((path, label))

    for task, path_label in label_dict.items():
        mkdir_if_not_exist([trainandvaldir, task])
        train_count = 0
        n = len(path_label)
        m = len(list(path_label[0][1]))

        for mm in range(m):
            mkdir_if_not_exist([trainandvaldir, task, 'train', str(mm)])
            mkdir_if_not_exist([trainandvaldir, task, 'val', str(mm)])

        random.shuffle(path_label)
        for path, label in path_label:
            label_index = list(label).index('y')
            src_path = os.path.join(basedir, path)
            if train_count < n * 0.95:
                shutil.copy(src_path,
                            os.path.join(trainandvaldir, task, 'train', str(label_index)))
            else:
                shutil.copy(src_path,
                            os.path.join(trainandvaldir, task, 'val', str(label_index)))
            train_count += 1
    print(1, train_count)

    # Add warmup data to skirt task
    label_dict = {'skirt_length_labels': []}

    with open(warmup_label_dir, 'r') as f:
        lines = f.readlines()
        tokens = [l.rstrip().split(',') for l in lines]
        for path, task, label in tokens:
            label_dict[task].append((path, label))

    for task, path_label in label_dict.items():
        train_count = 0
        n = len(path_label)
        m = len(list(path_label[0][1]))

        random.shuffle(path_label)
        for path, label in path_label:
            label_index = list(label).index('y')
            src_path = os.path.join(webdir, path)
            if train_count < n * 0.95:
                shutil.copy(src_path,
                            os.path.join(trainandvaldir, task, 'train', str(label_index)))
            else:
                shutil.copy(src_path,
                            os.path.join(trainandvaldir, task, 'val', str(label_index)))
            train_count += 1
    print(3, train_count)

    # test a
    label_dict = {'coat_length_labels': [],
                  'lapel_design_labels': [],
                  'neckline_design_labels': [],
                  'skirt_length_labels': [],
                  'collar_design_labels': [],
                  'neck_design_labels': [],
                  'pant_length_labels': [],
                  'sleeve_length_labels': []}

    task_list = label_dict.keys()

    with open(testa_label_dir, 'r') as f:
        lines = f.readlines()
        tokens = [l.rstrip().split(',') for l in lines]
        for path, task, label in tokens:
            label_dict[task].append((path, label))

    for task, path_label in label_dict.items():
        mkdir_if_not_exist([trainandvaldir, task])
        train_count = 0
        n = len(path_label)
        m = len(list(path_label[0][1]))

        for mm in range(m):
            mkdir_if_not_exist([trainandvaldir, task, 'train', str(mm)])
            mkdir_if_not_exist([trainandvaldir, task, 'val', str(mm)])

        random.shuffle(path_label)
        for path, label in path_label:
            label_index = list(label).index('y')
            src_path = os.path.join(test_a, path)
            if train_count < n * 0.95:
                shutil.copy(src_path,
                            os.path.join(trainandvaldir, task, 'train', str(label_index)))
            else:
                shutil.copy(src_path,
                            os.path.join(trainandvaldir, task, 'val', str(label_index)))
            train_count += 1
    print(4, train_count)

    # test b
    label_dict = {'coat_length_labels': [],
                  'lapel_design_labels': [],
                  'neckline_design_labels': [],
                  'skirt_length_labels': [],
                  'collar_design_labels': [],
                  'neck_design_labels': [],
                  'pant_length_labels': [],
                  'sleeve_length_labels': []}

    task_list = label_dict.keys()

    with open(testb_label_dir, 'r') as f:
        lines = f.readlines()
        tokens = [l.rstrip().split(',') for l in lines]
        for path, task, label in tokens:
            label_dict[task].append((path, label))

    for task, path_label in label_dict.items():
        mkdir_if_not_exist([trainandvaldir, task])
        train_count = 0
        n = len(path_label)
        m = len(list(path_label[0][1]))

        for mm in range(m):
            mkdir_if_not_exist([trainandvaldir, task, 'train', str(mm)])
            mkdir_if_not_exist([trainandvaldir, task, 'val', str(mm)])

        random.shuffle(path_label)
        for path, label in path_label:
            label_index = list(label).index('y')
            src_path = os.path.join(test_b, path)
            if train_count < n * 0.95:
                shutil.copy(src_path,
                            os.path.join(trainandvaldir, task, 'train', str(label_index)))
            else:
                shutil.copy(src_path,
                            os.path.join(trainandvaldir, task, 'val', str(label_index)))
            train_count += 1
    print(5, train_count)

def gen_2_full():
    # round 2 data
    trainandvaldir = trainandvaldir_2_full
    label_dict = {'coat_length_labels': [],
                  'lapel_design_labels': [],
                  'neckline_design_labels': [],
                  'skirt_length_labels': [],
                  'collar_design_labels': [],
                  'neck_design_labels': [],
                  'pant_length_labels': [],
                  'sleeve_length_labels': []}

    task_list = label_dict.keys()

    with open(label2_dir, 'r') as f:
        lines = f.readlines()
        tokens = [l.rstrip().split(',') for l in lines]
        for path, task, label in tokens:
            label_dict[task].append((path, label))

    for task, path_label in label_dict.items():
        mkdir_if_not_exist([trainandvaldir, task])
        train_count = 0
        n = len(path_label)
        m = len(list(path_label[0][1]))

        for mm in range(m):
            mkdir_if_not_exist([trainandvaldir, task, 'train', str(mm)])
            mkdir_if_not_exist([trainandvaldir, task, 'val', str(mm)])

        random.shuffle(path_label)
        for path, label in path_label:
            label_index = list(label).index('y')
            src_path = os.path.join(round2imagedir, path)
            shutil.copy(src_path,
                            os.path.join(trainandvaldir, task, 'train', str(label_index)))
            train_count += 1
    print(2, train_count)

def gen_2_trainval():
    # round 2 data
    trainandvaldir = trainandvaldir_2_trainval
    label_dict = {'coat_length_labels': [],
                  'lapel_design_labels': [],
                  'neckline_design_labels': [],
                  'skirt_length_labels': [],
                  'collar_design_labels': [],
                  'neck_design_labels': [],
                  'pant_length_labels': [],
                  'sleeve_length_labels': []}

    task_list = label_dict.keys()

    with open(label2_dir, 'r') as f:
        lines = f.readlines()
        tokens = [l.rstrip().split(',') for l in lines]
        for path, task, label in tokens:
            label_dict[task].append((path, label))

    for task, path_label in label_dict.items():
        mkdir_if_not_exist([trainandvaldir, task])
        train_count = 0
        n = len(path_label)
        m = len(list(path_label[0][1]))

        for mm in range(m):
            mkdir_if_not_exist([trainandvaldir, task, 'train', str(mm)])
            mkdir_if_not_exist([trainandvaldir, task, 'val', str(mm)])

        random.shuffle(path_label)
        for path, label in path_label:
            label_index = list(label).index('y')
            src_path = os.path.join(round2imagedir, path)
            if train_count < n * 0.95:
                shutil.copy(src_path,
                            os.path.join(trainandvaldir, task, 'train', str(label_index)))
            else:
                shutil.copy(src_path,
                            os.path.join(trainandvaldir, task, 'val', str(label_index)))
            train_count += 1
    print(2, train_count)


gen_1_trainval()
gen_1_full()
gen_2_trainval()
gen_2_full()
