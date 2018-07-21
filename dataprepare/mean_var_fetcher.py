from PIL import Image
import numpy as np
import config


def get_files(dir):
    import os
    # if not os.path.exists(dir):
    #     return []
    # if os.path.isfile(dir):
    #     return [dir]
    # result = []
    # for subdir in os.listdir(dir):
    #     sub_path = os.path.join(dir, subdir)
    #     result += get_files(sub_path)
    import glob
    imgs = glob.glob(dir + '/*.jpg')
    return imgs


r = 0  # r mean
g = 0  # g mean
b = 0  # b mean

r_2 = 0  # r^2
g_2 = 0  # g^2
b_2 = 0  # b^2

total = 0
task_list = [
    'coat_length_labels',
    'collar_design_labels',
    'lapel_design_labels',
    'neck_design_labels',
    'neckline_design_labels',
    'pant_length_labels',
    'skirt_length_labels',
    'sleeve_length_labels'
]
import glob
dir = '/home/kaka/FASHIONAI_DATA/train_valid_2_full/' + task_list[7] + '/train'


def listdir(path, list_name):  # 传入存储的list
    import os
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        else:
            list_name.append(file_path)

imgs = []
listdir(dir, imgs)
i=0
count=len(imgs)
for image_file in imgs:
    print('Process: %d/%d' % (i, count))
    img = Image.open(image_file)
    # img = img.resize((299, 299))
    img = np.asarray(img)
    img = img.astype('float32') / 255.
    total += img.shape[0] * img.shape[1]

    r += img[:, :, 0].sum()
    g += img[:, :, 1].sum()
    b += img[:, :, 2].sum()

    r_2 += (img[:, :, 0] ** 2).sum()
    g_2 += (img[:, :, 1] ** 2).sum()
    b_2 += (img[:, :, 2] ** 2).sum()
    i += 1

r_mean = r / total
g_mean = g / total
b_mean = b / total

r_var = r_2 / total - r_mean ** 2
g_var = g_2 / total - g_mean ** 2
b_var = b_2 / total - b_mean ** 2

print('Mean is %s' % ([r_mean, g_mean, b_mean]))
print('Var is %s' % ([r_var, g_var, b_var]))