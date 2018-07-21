import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

downlist = ['coat_length_labels', 'pant_length_labels', 'skirt_length_labels']
uplist = ['collar_design_labels', 'lapel_design_labels', 'neck_design_labels', 'neckline_design_labels']


def loc_(loc, img):
    width = img.size[0]
    height = img.size[1]

    if loc[0] < 0:
        loc[0] = 0
    if loc[1] < 0:
        loc[1] = 0
    if loc[2] > width:
        loc[2] = width
    if loc[3] > height:
        loc[3] = height

    return (loc[0], loc[1], loc[2], loc[3])


def Normalrandom():
    return (np.random.rand() + np.random.rand()) / 2


def padImage(img):
    max_hw = max(img.size)

    newimg = Image.new("RGB", (max_hw, max_hw), (255, 255, 255))
    if max_hw == img.size[1]:
        l = int((max_hw - img.size[0]) / 2)
        r = l + img.size[0]
        box = (l, 0, r, max_hw)
    else:
        u = int((max_hw - img.size[1]) / 2)
        l = u + img.size[1]
        box = (0, u, max_hw, l)
    newimg.paste(img, box)
    return newimg


def padImagewithtask(img, task):
    max_hw = max(img.size)

    newimg = Image.new("RGB", (max_hw, max_hw), (255, 255, 255))
    if max_hw == img.size[1]:
        l = int((max_hw - img.size[0]) / 2)
        r = l + img.size[0]
        box = (l, 0, r, max_hw)
    else:
        if task in uplist:
            u = 0
            l = u + img.size[1]
            box = (0, u, max_hw, l)
            # u = int((max_hw - img.size[1]) / 2)
            # l = u + img.size[1]
            # box = (0, u, max_hw, l)
        elif task in downlist:
            u = max_hw - img.size[1]
            l = max_hw
            box = (0, u, max_hw, l)
        else:
            u = int((max_hw - img.size[1]) / 2)
            l = u + img.size[1]
            box = (0, u, max_hw, l)
    newimg.paste(img, box)
    return newimg


def padImagewithrnd(img):
    max_hw = max(img.size)
    newimg = Image.new("RGB", (max_hw, max_hw), (255, 255, 255))
    if max_hw == img.size[1]:
        l = int((max_hw - img.size[0]) * random.random())
        r = l + img.size[0]
        box = (l, 0, r, max_hw)
    else:
        u = int((max_hw - img.size[1]) * random.random())
        d = u + img.size[1]
        box = (0, u, max_hw, d)
    newimg.paste(img, box)
    return newimg


def padImagewithrndandloc(img, loc):
    max_hw = max(img.size)
    loc_l, loc_u, loc_r, loc_d = loc
    newimg = Image.new("RGB", (max_hw, max_hw), (255, 255, 255))
    if max_hw == img.size[1]:
        l = int((max_hw - img.size[0]) * random.random())
        r = l + img.size[0]
        box = (l, 0, r, max_hw)
        loc_l = loc_l + l
        loc_r = loc_r + l
    else:
        u = int((max_hw - img.size[1]) * random.random())
        d = u + img.size[1]
        box = (0, u, max_hw, d)
        loc_u = loc_u + u
        loc_d = loc_d + u
    newimg.paste(img, box)
    return newimg, [loc_l, loc_u, loc_r, loc_d]


def padImagewithrndcrop(img, loc, task):
    srcwidth = img.width
    srcheight = img.height

    loc_l, loc_u, loc_r, loc_d = loc

    if task in downlist:
        loc_d = srcheight
    else:
        loc_d += 5
    if task in uplist:
        loc_u = 0
    else:
        loc_u -= 5
    loc_l -= 8
    loc_r += 8

    loc_l, loc_u, loc_r, loc_d = loc_([loc_l, loc_u, loc_r, loc_d], img)

    loc_l = loc_l * Normalrandom()
    loc_r = loc_r + (srcwidth - loc_r) * Normalrandom()
    loc_u = loc_u * Normalrandom()
    loc_d = loc_d + (srcheight - loc_d) * Normalrandom()

    loc_l, loc_u, loc_r, loc_d = loc_([loc_l, loc_u, loc_r, loc_d], img)

    cropimg = img.crop((int(loc_l), int(loc_u), int(loc_r), int(loc_d)))
    max_hw = max(cropimg.size)
    padimg = Image.new("RGB", (max_hw, max_hw), (255, 255, 255))
    if max_hw == cropimg.size[1]:
        l = int((max_hw - cropimg.size[0]) * Normalrandom())
        r = l + cropimg.size[0]
        box = (l, 0, r, max_hw)
    else:
        u = int((max_hw - cropimg.size[1]) * Normalrandom())
        d = u + cropimg.size[1]
        box = (0, u, max_hw, d)
    padimg.paste(cropimg, box)
    return padimg


def padImagewithrndcrop_test(img, loc, task):
    srcwidth = img.width
    srcheight = img.height

    loc_l, loc_u, loc_r, loc_d = loc

    if task in downlist:
        loc_d = srcheight
    else:
        loc_d += 5
    if task in uplist:
        loc_u = 0
    else:
        loc_u -= 5
    loc_l -= 8
    loc_r += 8

    loc_l, loc_u, loc_r, loc_d = loc_([loc_l, loc_u, loc_r, loc_d], img)

    loc_l = loc_l * 0.5
    loc_r = loc_r + (srcwidth - loc_r) * 0.5
    loc_u = loc_u * 0.5
    loc_d = loc_d + (srcheight - loc_d) * 0.5

    loc_l, loc_u, loc_r, loc_d = loc_([loc_l, loc_u, loc_r, loc_d], img)

    cropimg = img.crop((int(loc_l), int(loc_u), int(loc_r), int(loc_d)))
    max_hw = max(cropimg.size)
    padimg = Image.new("RGB", (max_hw, max_hw), (255, 255, 255))
    if max_hw == cropimg.size[1]:
        l = int((max_hw - cropimg.size[0]) * 0.5)
        r = l + cropimg.size[0]
        box = (l, 0, r, max_hw)
    else:
        u = int((max_hw - cropimg.size[1]) * 0.5)
        d = u + cropimg.size[1]
        box = (0, u, max_hw, d)
    padimg.paste(cropimg, box)
    return padimg


def stretchingimg(img):
    if np.random.uniform() > 0.5:
        rnd = np.random.uniform(0.9, 1.0)
        img = img.resize((int(img.width * rnd), img.height))
        img = padImagewithrnd(img)
    else:
        rnd = np.random.uniform(0.9, 1.0)
        img = img.resize((img.width, int(img.height * rnd)))
        img = padImagewithrnd(img)
    return img


def cropwithloc(img, loc, task):
    height = img.height
    width = img.width
    x1, y1, x2, y2 = loc

    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x2 > width:
        x2 = width
    if y2 > height:
        y2 = height

    if task == 'coat_length_labels' \
            or task == 'collar_design_labels' \
            or task == 'collar_design_labels' \
            or task == 'neck_design_labels' \
            or task == 'neckline_design_labels':
        y2 = height
    else:
        y1 = 0

    person_height = y2 - y1
    person_width = x2 - x1
    if person_width > person_height:

        left = x1 * (1 - random.uniform(0, 1))
        right = x2 + (width - x2) * random.uniform(0, 1)

        person_width = right - left
        # 补上
        if task == 'coat_length_labels' \
                or task == 'collar_design_labels' \
                or task == 'collar_design_labels' \
                or task == 'neck_design_labels' \
                or task == 'neckline_design_labels':
            y1 = height - person_width

            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0
            if x2 > width:
                x2 = width
            if y2 > height:
                y2 = height
            loc_1 = loc[1] - y1
            loc_3 = loc[3] - y1
            loc_0 = loc[0] - left
            loc_2 = loc[2] - left
            if (y2 - y1) / height > 0.99:
                return img.crop((int(left), int(y1), int(right), int(y2))), 2, (loc_0, loc_1, loc_2, loc_3)
            else:
                return img.crop((int(left), int(y1), int(right), int(y2))), 1, (loc_0, loc_1, loc_2, loc_3)
        # 补下
        else:
            y2 = y1 + person_width
            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0
            if x2 > width:
                x2 = width
            if y2 > height:
                y2 = height
            loc_1 = loc[1] - y1
            loc_3 = loc[3] - y1
            loc_0 = loc[0] - left
            loc_2 = loc[2] - left
        if (y2 - y1) / height > 0.99:
            return img.crop((int(left), int(y1), int(right), int(y2))), 2, (loc_0, loc_1, loc_2, loc_3)
        else:
            return img.crop((int(left), int(y1), int(right), int(y2))), 1, (loc_0, loc_1, loc_2, loc_3)
    else:
        if y1 == 0:
            return img, 2, loc
        else:
            if task == 'coat_length_labels' \
                    or task == 'collar_design_labels' \
                    or task == 'collar_design_labels' \
                    or task == 'neck_design_labels' \
                    or task == 'neckline_design_labels':
                y1 = random.uniform(0, 1) * y1
                # y2 = y2 + (height - y2) * random.uniform(0, 1)
                person_height = y2 - y1

                left1 = x2 - person_height
                left2 = x1
                if (left2 + person_height) > width:
                    left2 = width - person_height
                if left1 < 0:
                    left1 = 0
                if left2 > width:
                    left2 = width
                left = left1 + (left2 - left1) * random.uniform(0, 1)

                if x1 < 0:
                    x1 = 0
                if y1 < 0:
                    y1 = 0
                if x2 > width:
                    x2 = width
                if y2 > height:
                    y2 = height
                loc_0 = loc[0] - left
                loc_2 = loc[2] - left
                loc_1 = loc[1] - y1
                loc_3 = loc[3] - y1
                if (y2 - y1) / height > 0.99:
                    return img.crop((int(left), int(y1), int(left + person_height), int(y2))), 2, (
                        loc_0, loc_1, loc_2, loc_3)
                else:
                    return img.crop((int(left), int(y1), int(left + person_height), int(y2))), 0, (
                        loc_0, loc_1, loc_2, loc_3)
            else:
                # y1 = random.uniform(0, 1) * y1
                y1 = 0
                y2 = y2 + (height - y2) * random.uniform(0, 1)
                person_height = y2 - y1

                left1 = x2 - person_height
                left2 = x1
                if (left2 + person_height) > width:
                    left2 = width - person_height
                if left1 < 0:
                    left1 = 0
                if left2 > width:
                    left2 = width
                left = left1 + (left2 - left1) * random.uniform(0, 1)

                if x1 < 0:
                    x1 = 0
                if y1 < 0:
                    y1 = 0
                if x2 > width:
                    x2 = width
                if y2 > height:
                    y2 = height
                loc_0 = loc[0] - left
                loc_2 = loc[2] - left
                loc_1 = loc[1] - y1
                loc_3 = loc[3] - y1
                if (y2 - y1) / height > 0.99:
                    return img.crop((int(left), int(y1), int(left + person_height), int(y2))), 2, (
                        loc_0, loc_1, loc_2, loc_3)
                else:
                    return img.crop((int(left), int(y1), int(left + person_height), int(y2))), 0, (
                        loc_0, loc_1, loc_2, loc_3)


def padwithnolocrnd(img, task):
    width, height = img.size
    max_hw = max(img.size)
    if width == height or (1.0 < (width / height) < 1.0):
        return img
    else:
        if max_hw == height:
            if task == 'sleeve_length_labels':  # white pad
                newimg = Image.new("RGB", (max_hw, max_hw), (255, 255, 255))
                l = int((max_hw - img.size[0]) * random.random())
                r = l + img.size[0]
                box = (l, 0, r, max_hw)
                newimg.paste(img, box)
            else:  # same pad
                newimg = Image.new("RGB", (max_hw, max_hw), (255, 255, 255))
                l = int((max_hw - width) * random.random())
                r = l + width
                box = (l, 0, r, max_hw)
                newimg.paste(img, box)

                if task in uplist:
                    col = newimg.crop((box[0], 0, box[0] + 1, max_hw))
                    for i in (range(l)):
                        box1 = (i, 0, i + 1, max_hw)
                        newimg.paste(col, box1)
                    col = newimg.crop((box[2] - 1, 0, box[2], max_hw))
                    for i in range(r, max_hw):
                        box1 = (i, 0, i + 1, max_hw)
                        newimg.paste(col, box1)
        else:
            if task in uplist:
                newimg = Image.new("RGB", (max_hw, max_hw), (255, 255, 255))
                u = 0
                d = u + img.size[1]
                box = (0, u, max_hw, d)
                newimg.paste(img, box)

                if task in uplist:
                    col = newimg.crop((0, d - 1, max_hw, d))
                    for i in (range(d, max_hw)):
                        box1 = (0, i, max_hw, i + 1)
                        newimg.paste(col, box1)
            elif task in downlist:
                newimg = Image.new("RGB", (max_hw, max_hw), (255, 255, 255))
                u = max_hw - img.size[1]
                d = max_hw
                box = (0, u, max_hw, d)
                newimg.paste(img, box)

                if task in uplist:
                    col = newimg.crop((0, u, max_hw, u + 1))
                    for i in (range(u)):
                        box1 = (0, i, max_hw, i + 1)
                        newimg.paste(col, box1)
            else:
                newimg = Image.new("RGB", (max_hw, max_hw), (255, 255, 255))
                u = int((max_hw - img.size[1]) * random.random())
                d = u + img.size[1]
                box = (0, u, max_hw, d)
                newimg.paste(img, box)

                if task in uplist:
                    col = newimg.crop((0, u, max_hw, u + 1))
                    for i in (range(u)):
                        box1 = (0, i, max_hw, i + 1)
                        newimg.paste(col, box1)
                    col = newimg.crop((0, d - 1, max_hw, d))
                    for i in (range(d, max_hw)):
                        box1 = (0, i, max_hw, i + 1)
                        newimg.paste(col, box1)
    return newimg


def padwithnolocrnd1(img, task):
    width, height = img.size
    if height == 0:
        print(1)
    max_hw = max(img.size)
    if width == height or (0.75 < (width / height) < 1.33):
        return img
    else:
        if max_hw == height:
            if task == 'sleeve_length_labels':  # white pad
                newimg = Image.new("RGB", (max_hw, max_hw), (255, 255, 255))
                l = int((max_hw - img.size[0]) * random.random())
                r = l + img.size[0]
                box = (l, 0, r, max_hw)
                newimg.paste(img, box)
            else:  # same pad
                newimg = Image.new("RGB", (max_hw, max_hw), (255, 255, 255))
                l = int((max_hw - width) * random.random())
                r = l + width
                box = (l, 0, r, max_hw)
                newimg.paste(img, box)

                col = newimg.crop((box[0], 0, box[0] + 1, max_hw))
                for i in (range(l)):
                    box1 = (i, 0, i + 1, max_hw)
                    newimg.paste(col, box1)
                col = newimg.crop((box[2] - 1, 0, box[2], max_hw))
                for i in range(r, max_hw):
                    box1 = (i, 0, i + 1, max_hw)
                    newimg.paste(col, box1)
        else:
            if task in uplist:
                newimg = Image.new("RGB", (max_hw, max_hw), (255, 255, 255))
                u = 0
                d = u + img.size[1]
                box = (0, u, max_hw, d)
                newimg.paste(img, box)

                col = newimg.crop((0, d - 1, max_hw, d))
                for i in (range(d, max_hw)):
                    box1 = (0, i, max_hw, i + 1)
                    newimg.paste(col, box1)
            elif task in downlist:
                newimg = Image.new("RGB", (max_hw, max_hw), (255, 255, 255))
                u = max_hw - img.size[1]
                d = max_hw
                box = (0, u, max_hw, d)
                newimg.paste(img, box)

                col = newimg.crop((0, u, max_hw, u + 1))
                for i in (range(u)):
                    box1 = (0, i, max_hw, i + 1)
                    newimg.paste(col, box1)
            else:
                newimg = Image.new("RGB", (max_hw, max_hw), (255, 255, 255))
                u = int((max_hw - img.size[1]) * random.random())
                d = u + img.size[1]
                box = (0, u, max_hw, d)
                newimg.paste(img, box)

                col = newimg.crop((0, u, max_hw, u + 1))
                for i in (range(u)):
                    box1 = (0, i, max_hw, i + 1)
                    newimg.paste(col, box1)
                col = newimg.crop((0, d - 1, max_hw, d))
                for i in (range(d, max_hw)):
                    box1 = (0, i, max_hw, i + 1)
                    newimg.paste(col, box1)
    return newimg



def padwithnoloc_T(img, task):
    width, height = img.size
    max_hw = max(img.size)
    if width == height or (1.0 < (width / height) < 1.0):
        return img
    else:
        if max_hw == height:
            if task == 'sleeve_length_labels':  # white pad
                newimg = Image.new("RGB", (max_hw, max_hw), (255, 255, 255))
                l = int((max_hw - img.size[0]) * 0.5)
                r = l + img.size[0]
                box = (l, 0, r, max_hw)
                newimg.paste(img, box)
            else:  # same pad
                newimg = Image.new("RGB", (max_hw, max_hw), (255, 255, 255))
                l = int((max_hw - width) * 0.5)
                r = l + width
                box = (l, 0, r, max_hw)
                newimg.paste(img, box)

                col = newimg.crop((box[0], 0, box[0] + 1, max_hw))
                for i in (range(l)):
                    box1 = (i, 0, i + 1, max_hw)
                    newimg.paste(col, box1)
                col = newimg.crop((box[2] - 1, 0, box[2], max_hw))
                for i in range(r, max_hw):
                    box1 = (i, 0, i + 1, max_hw)
                    newimg.paste(col, box1)
        else:
            if task in uplist:
                newimg = Image.new("RGB", (max_hw, max_hw), (255, 255, 255))
                u = 0
                d = u + img.size[1]
                box = (0, u, max_hw, d)
                newimg.paste(img, box)

                col = newimg.crop((0, d - 1, max_hw, d))
                for i in (range(d, max_hw)):
                    box1 = (0, i, max_hw, i + 1)
                    newimg.paste(col, box1)
            elif task in downlist:
                newimg = Image.new("RGB", (max_hw, max_hw), (255, 255, 255))
                u = max_hw - img.size[1]
                d = max_hw
                box = (0, u, max_hw, d)
                newimg.paste(img, box)

                col = newimg.crop((0, u, max_hw, u + 1))
                for i in (range(u)):
                    box1 = (0, i, max_hw, i + 1)
                    newimg.paste(col, box1)
            else:
                newimg = Image.new("RGB", (max_hw, max_hw), (255, 255, 255))
                u = int((max_hw - img.size[1]) * 0.5)
                d = u + img.size[1]
                box = (0, u, max_hw, d)
                newimg.paste(img, box)

                col = newimg.crop((0, u, max_hw, u + 1))
                for i in (range(u)):
                    box1 = (0, i, max_hw, i + 1)
                    newimg.paste(col, box1)
                col = newimg.crop((0, d - 1, max_hw, d))
                for i in (range(d, max_hw)):
                    box1 = (0, i, max_hw, i + 1)
                    newimg.paste(col, box1)
    return newimg

def padwithnoloc_T1(img, task):
    width, height = img.size
    max_hw = max(img.size)
    if width == height or (1.0 < (width / height) < 1.0):
        return img
    else:
        if max_hw == height:
            if task == 'sleeve_length_labels':  # white pad
                newimg = Image.new("RGB", (max_hw, max_hw), (255, 255, 255))
                l = int((max_hw - img.size[0]) * 0.5)
                r = l + img.size[0]
                box = (l, 0, r, max_hw)
                newimg.paste(img, box)
            else:  # same pad
                newimg = Image.new("RGB", (max_hw, max_hw), (255, 255, 255))
                l = int((max_hw - width) * 0.5)
                r = l + width
                box = (l, 0, r, max_hw)
                newimg.paste(img, box)

                col = newimg.crop((box[0], 0, box[0] + 1, max_hw))
                for i in (range(l)):
                    box1 = (i, 0, i + 1, max_hw)
                    newimg.paste(col, box1)
                col = newimg.crop((box[2] - 1, 0, box[2], max_hw))
                for i in range(r, max_hw):
                    box1 = (i, 0, i + 1, max_hw)
                    newimg.paste(col, box1)
        else:
            if task in uplist:
                newimg = Image.new("RGB", (max_hw, max_hw), (255, 255, 255))
                u = 0
                d = u + img.size[1]
                box = (0, u, max_hw, d)
                newimg.paste(img, box)

                col = newimg.crop((0, d - 1, max_hw, d))
                for i in (range(d, max_hw)):
                    box1 = (0, i, max_hw, i + 1)
                    newimg.paste(col, box1)
            elif task in downlist:
                newimg = Image.new("RGB", (max_hw, max_hw), (255, 255, 255))
                u = max_hw - img.size[1]
                d = max_hw
                box = (0, u, max_hw, d)
                newimg.paste(img, box)

                col = newimg.crop((0, u, max_hw, u + 1))
                for i in (range(u)):
                    box1 = (0, i, max_hw, i + 1)
                    newimg.paste(col, box1)
            else:
                newimg = Image.new("RGB", (max_hw, max_hw), (255, 255, 255))
                u = int((max_hw - img.size[1]) * 0.5)
                d = u + img.size[1]
                box = (0, u, max_hw, d)
                newimg.paste(img, box)

                col = newimg.crop((0, u, max_hw, u + 1))
                for i in (range(u)):
                    box1 = (0, i, max_hw, i + 1)
                    newimg.paste(col, box1)
                col = newimg.crop((0, d - 1, max_hw, d))
                for i in (range(d, max_hw)):
                    box1 = (0, i, max_hw, i + 1)
                    newimg.paste(col, box1)
    return newimg

def padwithnoloc_T2(img, task):
    width, height = img.size
    max_hw = max(img.size)
    if width == height or (1.0 < (width / height) < 1.0):
        return img
    else:
        if max_hw == height:
            if task == 'sleeve_length_labels':  # white pad
                newimg = Image.new("RGB", (max_hw, max_hw), (255, 255, 255))
                l = int((max_hw - img.size[0]) * 0.5)
                r = l + img.size[0]
                box = (l, 0, r, max_hw)
                newimg.paste(img, box)
            else:  # same pad
                newimg = Image.new("RGB", (max_hw, max_hw), (255, 255, 255))
                l = int((max_hw - width) * 0.5)
                r = l + width
                box = (l, 0, r, max_hw)
                newimg.paste(img, box)
        else:
            if task in uplist:
                newimg = Image.new("RGB", (max_hw, max_hw), (255, 255, 255))
                u = 0
                d = u + img.size[1]
                box = (0, u, max_hw, d)
                newimg.paste(img, box)

            elif task in downlist:
                newimg = Image.new("RGB", (max_hw, max_hw), (255, 255, 255))
                u = max_hw - img.size[1]
                d = max_hw
                box = (0, u, max_hw, d)
                newimg.paste(img, box)
            else:
                newimg = Image.new("RGB", (max_hw, max_hw), (255, 255, 255))
                u = int((max_hw - img.size[1]) * 0.5)
                d = u + img.size[1]
                box = (0, u, max_hw, d)
                newimg.paste(img, box)

    return newimg

def padwithlocrnd(img, loc, task):
    width, height = img.size

    loc_l, loc_u, loc_r, loc_d = loc

    if task in downlist:
        loc_d = height
    else:
        loc_d += 10
    if task in uplist:
        loc_u = 0
    else:
        loc_u -= 10
    if task == 'sleeve_length_labels' \
            or task == 'skirt_length_labels' \
            or task == 'coat_length_labels':
        loc_l -= 20
        loc_r += 20
    else:
        loc_l -= 10
        loc_r += 10

    loc_l, loc_u, loc_r, loc_d = loc_([loc_l, loc_u, loc_r, loc_d], img)

    # loc_height = loc_d - loc_u
    # loc_width = loc_r - loc_l
    loc_l = loc_l * random.random()
    loc_r = loc_r + (width - loc_r) * random.random()
    loc_u = loc_u * random.random()
    loc_d = loc_d + (height - loc_d) * random.random()

    c_height = loc_d - loc_u
    c_width = loc_r - loc_l
    if c_height > c_width:
        while (loc_l >= 0 or loc_r <= width) and c_height > c_width:
            if loc_l >= 0:
                loc_l -= 1
            if loc_r <= width:
                loc_r += 1
    else:
        while (loc_u >= 0 or loc_d <= height) and c_width > c_height:
            if loc_u >= 0:
                loc_u -= 1
            if loc_d <= height:
                loc_d += 1

    loc_l, loc_u, loc_r, loc_d = loc_([loc_l, loc_u, loc_r, loc_d], img)
    img = img.crop((int(loc_l), int(loc_u), int(loc_r), int(loc_d)))

    img = padwithnolocrnd(img, task)
    return img


def padwithloc_T(img, loc, task):
    width, height = img.size

    loc_l, loc_u, loc_r, loc_d = loc

    if task in downlist:
        loc_d = height
    else:
        loc_d += 10
    if task in uplist:
        loc_u = 0
    else:
        loc_u -= 10
    if task == 'sleeve_length_labels' \
            or task == 'skirt_length_labels' \
            or task == 'coat_length_labels':
        loc_l -= 25
        loc_r += 25
    else:
        loc_l -= 10
        loc_r += 10

    loc_l, loc_u, loc_r, loc_d = loc_([loc_l, loc_u, loc_r, loc_d], img)

    # loc_height = loc_d - loc_u
    # loc_width = loc_r - loc_l

    loc_l = loc_l * 0.5
    loc_r = loc_r + (width - loc_r) * 0.5
    loc_u = loc_u * 0.5
    loc_d = loc_d + (height - loc_d) * 0.5

    c_height = loc_d - loc_u
    c_width = loc_r - loc_l
    if c_height > c_width:
        while (loc_l >= 0 or loc_r <= width) and c_height > c_width:
            if loc_l >= 0:
                loc_l -= 1
            if loc_r <= width:
                loc_r += 1
    else:
        while (loc_u >= 0 or loc_d <= height) and c_width > c_height:
            if loc_u >= 0:
                loc_u -= 1
            if loc_d <= height:
                loc_d += 1

    loc_l, loc_u, loc_r, loc_d = loc_([loc_l, loc_u, loc_r, loc_d], img)
    img = img.crop((int(loc_l), int(loc_u), int(loc_r), int(loc_d)))

    img = padwithnoloc_T(img, task)
    return img


def padwithloc_T1(img, loc, task):
    width, height = img.size

    loc_l, loc_u, loc_r, loc_d = loc

    if task in downlist:
        loc_d = height
    else:
        loc_d += 10
    if task in uplist:
        loc_u = 0
    else:
        loc_u -= 10
    if task == 'sleeve_length_labels' \
            or task == 'skirt_length_labels' \
            or task == 'coat_length_labels':
        loc_l -= 25
        loc_r += 25
    else:
        loc_l -= 10
        loc_r += 10

    loc_l, loc_u, loc_r, loc_d = loc_([loc_l, loc_u, loc_r, loc_d], img)

    # loc_height = loc_d - loc_u
    # loc_width = loc_r - loc_l

    loc_l = loc_l * 0.5
    loc_r = loc_r + (width - loc_r) * 0.5
    loc_u = loc_u * 0.5
    loc_d = loc_d + (height - loc_d) * 0.5

    c_height = loc_d - loc_u
    c_width = loc_r - loc_l
    if c_height > c_width:
        while (loc_l >= 0 or loc_r <= width) and c_height > c_width:
            if loc_l >= 0:
                loc_l -= 1
            if loc_r <= width:
                loc_r += 1
    else:
        while (loc_u >= 0 or loc_d <= height) and c_width > c_height:
            if loc_u >= 0:
                loc_u -= 1
            if loc_d <= height:
                loc_d += 1

    loc_l, loc_u, loc_r, loc_d = loc_([loc_l, loc_u, loc_r, loc_d], img)
    img = img.crop((int(loc_l), int(loc_u), int(loc_r), int(loc_d)))

    img = padwithnoloc_T(img, task)
    return img

def padwithloc_T2(img, loc, task):
    width, height = img.size

    loc_l, loc_u, loc_r, loc_d = loc

    if task in downlist:
        loc_d = height
    else:
        loc_d += 10
    if task in uplist:
        loc_u = 0
    else:
        loc_u -= 10
    if task == 'sleeve_length_labels' \
            or task == 'skirt_length_labels' \
            or task == 'coat_length_labels':
        loc_l -= 25
        loc_r += 25
    else:
        loc_l -= 10
        loc_r += 10

    loc_l, loc_u, loc_r, loc_d = loc_([loc_l, loc_u, loc_r, loc_d], img)

    # loc_height = loc_d - loc_u
    # loc_width = loc_r - loc_l

    loc_l = loc_l * 0.5
    loc_r = loc_r + (width - loc_r) * 0.5
    loc_u = loc_u * 0.5
    loc_d = loc_d + (height - loc_d) * 0.5

    c_height = loc_d - loc_u
    c_width = loc_r - loc_l
    if c_height > c_width:
        while (loc_l >= 0 or loc_r <= width) and c_height > c_width:
            if loc_l >= 0:
                loc_l -= 1
            if loc_r <= width:
                loc_r += 1
    else:
        while (loc_u >= 0 or loc_d <= height) and c_width > c_height:
            if loc_u >= 0:
                loc_u -= 1
            if loc_d <= height:
                loc_d += 1

    loc_l, loc_u, loc_r, loc_d = loc_([loc_l, loc_u, loc_r, loc_d], img)
    img = img.crop((int(loc_l), int(loc_u), int(loc_r), int(loc_d)))

    img = padwithnoloc_T1(img, task)
    return img

def padwithloc_T3(img, loc, task):
    p=1.0
    width, height = img.size

    loc_l, loc_u, loc_r, loc_d = loc

    if task in downlist:
        loc_d = height
    else:
        loc_d += 10
    if task in uplist:
        loc_u = 0
    else:
        loc_u -= 10
    if task == 'sleeve_length_labels' \
            or task == 'skirt_length_labels' \
            or task == 'coat_length_labels':
        loc_l -= 25
        loc_r += 25
    else:
        loc_l -= 10
        loc_r += 10

    loc_l, loc_u, loc_r, loc_d = loc_([loc_l, loc_u, loc_r, loc_d], img)

    # loc_height = loc_d - loc_u
    # loc_width = loc_r - loc_l

    loc_l = loc_l * p
    loc_r = loc_r + (width - loc_r) * (1-p)
    loc_u = loc_u * p
    loc_d = loc_d + (height - loc_d) * (1-p)

    c_height = loc_d - loc_u
    c_width = loc_r - loc_l
    if c_height > c_width:
        while (loc_l >= 0 or loc_r <= width) and c_height > c_width:
            if loc_l >= 0:
                loc_l -= 1
            if loc_r <= width:
                loc_r += 1
    else:
        while (loc_u >= 0 or loc_d <= height) and c_width > c_height:
            if loc_u >= 0:
                loc_u -= 1
            if loc_d <= height:
                loc_d += 1

    loc_l, loc_u, loc_r, loc_d = loc_([loc_l, loc_u, loc_r, loc_d], img)
    img = img.crop((int(loc_l), int(loc_u), int(loc_r), int(loc_d)))

    img = padwithnoloc_T(img, task)
    return img


def cropwithloc_test(img, loc, task):
    p1 = 0.5
    p2 = 0.5
    p3 = 0.5
    height = img.height
    width = img.width
    x1, y1, x2, y2 = loc

    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x2 > width:
        x2 = width
    if y2 > height:
        y2 = height

    if task == 'coat_length_labels' \
            or task == 'collar_design_labels' \
            or task == 'collar_design_labels' \
            or task == 'neck_design_labels' \
            or task == 'neckline_design_labels':
        y2 = height
    else:
        y1 = 0

    person_height = y2 - y1
    person_width = x2 - x1
    if person_width > person_height:

        left = x1 * (1 - p1)
        right = x2 + (width - x2) * p1

        person_width = right - left
        # 补上
        if task == 'coat_length_labels' \
                or task == 'collar_design_labels' \
                or task == 'collar_design_labels' \
                or task == 'neck_design_labels' \
                or task == 'neckline_design_labels':
            y1 = height - person_width

            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0
            if x2 > width:
                x2 = width
            if y2 > height:
                y2 = height
            loc_1 = loc[1] - y1
            loc_3 = loc[3] - y1
            loc_0 = loc[0] - left
            loc_2 = loc[2] - left
            if (y2 - y1) / height > 0.99:
                return img.crop((int(left), int(y1), int(right), int(y2))), 2, (loc_0, loc_1, loc_2, loc_3)
            else:
                return img.crop((int(left), int(y1), int(right), int(y2))), 1, (loc_0, loc_1, loc_2, loc_3)
        # 补下
        else:
            y2 = y1 + person_width
            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0
            if x2 > width:
                x2 = width
            if y2 > height:
                y2 = height
            loc_1 = loc[1] - y1
            loc_3 = loc[3] - y1
            loc_0 = loc[0] - left
            loc_2 = loc[2] - left
        if (y2 - y1) / height > 0.99:
            return img.crop((int(left), int(y1), int(right), int(y2))), 2, (loc_0, loc_1, loc_2, loc_3)
        else:
            return img.crop((int(left), int(y1), int(right), int(y2))), 1, (loc_0, loc_1, loc_2, loc_3)
    else:
        if y1 == 0:
            return img, 2, loc
        else:
            if task == 'coat_length_labels' \
                    or task == 'collar_design_labels' \
                    or task == 'collar_design_labels' \
                    or task == 'neck_design_labels' \
                    or task == 'neckline_design_labels':
                y1 = (1 - p2) * y1
                # y2 = y2 + (height - y2) * random.uniform(0, 1)
                person_height = y2 - y1

                left1 = x2 - person_height
                left2 = x1
                if (left2 + person_height) > width:
                    left2 = width - person_height
                if left1 < 0:
                    left1 = 0
                if left2 > width:
                    left2 = width
                left = left1 + (left2 - left1) * 0.5

                if x1 < 0:
                    x1 = 0
                if y1 < 0:
                    y1 = 0
                if x2 > width:
                    x2 = width
                if y2 > height:
                    y2 = height
                loc_0 = loc[0] - left
                loc_2 = loc[2] - left
                loc_1 = loc[1] - y1
                loc_3 = loc[3] - y1
                if (y2 - y1) / height > 0.99:
                    return img.crop((int(left), int(y1), int(left + person_height), int(y2))), 2, (
                        loc_0, loc_1, loc_2, loc_3)
                else:
                    return img.crop((int(left), int(y1), int(left + person_height), int(y2))), 0, (
                        loc_0, loc_1, loc_2, loc_3)
            else:
                # y1 = random.uniform(0, 1) * y1
                y1 = 0
                y2 = y2 + (height - y2) * p3
                person_height = y2 - y1

                left1 = x2 - person_height
                left2 = x1
                if (left2 + person_height) > width:
                    left2 = width - person_height
                if left1 < 0:
                    left1 = 0
                if left2 > width:
                    left2 = width
                left = left1 + (left2 - left1) * 0.5

                if x1 < 0:
                    x1 = 0
                if y1 < 0:
                    y1 = 0
                if x2 > width:
                    x2 = width
                if y2 > height:
                    y2 = height
                loc_0 = loc[0] - left
                loc_2 = loc[2] - left
                loc_1 = loc[1] - y1
                loc_3 = loc[3] - y1
                if (y2 - y1) / height > 0.99:
                    return img.crop((int(left), int(y1), int(left + person_height), int(y2))), 2, (
                        loc_0, loc_1, loc_2, loc_3)
                else:
                    return img.crop((int(left), int(y1), int(left + person_height), int(y2))), 0, (
                        loc_0, loc_1, loc_2, loc_3)


def enhanced_pad(img, loc, task):
    width, height = img.size
    max_hw = max(img.size)
    newimg = Image.new("RGB", (max_hw, max_hw), (0, 0, 0))
    if width == height or ((width / height) > 0.95 and (width / height) < 1.05):
        return img, loc
    elif max_hw == height:  # shu
        img_l = loc[0]
        img_r = loc[2]
        mid_lr = (img_l + img_r) / 2

        if abs(width / 2 - mid_lr) / width < 0.1:
            l = (int)((max_hw - width) / 2)
            r = l + width
            box = (l, 0, r, max_hw)
            newimg.paste(img, box)

            col = newimg.crop((box[0], 0, box[0] + 1, max_hw))
            for i in (range(box[0])):
                box1 = (i, 0, i + 1, max_hw)
                newimg.paste(col, box1)
            col = newimg.crop((box[2] - 1, 0, box[2], max_hw))
            for i in range(box[2], max_hw):
                box1 = (i, 0, i + 1, max_hw)
                newimg.paste(col, box1)
                # loc[0]+=l
                # loc[2]+=l
            loc = (loc[0] + l, loc[1], loc[2] + l, loc[3])
        elif mid_lr < (width / 2):
            l = 0
            r = l + width
            box = (l, 0, r, max_hw)
            newimg.paste(img, box)

            col = newimg.crop((box[2] - 1, 0, box[2], max_hw))
            for i in range(box[2], max_hw):
                box1 = (i, 0, i + 1, max_hw)
                newimg.paste(col, box1)
            # print(1)
            # loc[0] += l
            # loc[2] += l
            loc = (loc[0] + l, loc[1], loc[2] + l, loc[3])
        else:  # 1
            l = max_hw - width
            r = max_hw
            box = (l, 0, r, max_hw)
            newimg.paste(img, box)

            col = newimg.crop((box[0], 0, box[0] + 1, max_hw))
            for i in range(0, l):
                box1 = (i, 0, i + 1, max_hw)
                newimg.paste(col, box1)
            # loc[0] += l
            # loc[2] += l
            loc = (loc[0] + l, loc[1], loc[2] + l, loc[3])
    else:
        if task == 'coat_length_labels' \
                or task == 'collar_design_labels' \
                or task == 'collar_design_labels' \
                or task == 'neck_design_labels' \
                or task == 'neckline_design_labels':
            u = 0
            l = u + img.size[1]
            box = (0, u, max_hw, l)
            newimg.paste(img, box)

            col = newimg.crop((0, l - 1, max_hw, l))
            for i in (range(l, max_hw)):
                box1 = (0, i, max_hw, i + 1)
                newimg.paste(col, box1)

        else:
            u = max_hw - img.size[1]
            l = max_hw
            box = (0, u, max_hw, l)
            newimg.paste(img, box)

            col = newimg.crop((0, u, max_hw, u + 1))
            for i in (range(u)):
                box1 = (0, i, max_hw, i + 1)
                newimg.paste(col, box1)
            # loc[1] += u
            # loc[3] += u
            loc = (loc[0], loc[1] + u, loc[2], loc[3] + u)
    # plt.imshow(newimg)
    # plt.show()
    return newimg, loc

# def enhanced_pad(img, loc, task):
#     width, height = img.size
#     max_hw = max(img.size)
#     newimg = Image.new("RGB", (max_hw, max_hw), (0, 0, 0))
#     if width == height or ((width / height) > 0.95 and (width / height) < 1.05):
#         return img, loc
#     elif max_hw == height:  # shu
#         img_l = loc[0]
#         img_r = loc[2]
#         mid_lr = (img_l + img_r) / 2
#
#         if abs(width / 2 - mid_lr) / width < 0.1:
#             l = (int)((max_hw - width) / 2)
#             r = l + width
#             box = (l, 0, r, max_hw)
#             newimg.paste(img, box)
#
#             col = newimg.crop((box[0], 0, box[0] + 1, max_hw))
#             for i in (range(box[0])):
#                 box1 = (i, 0, i + 1, max_hw)
#                 newimg.paste(col, box1)
#             col = newimg.crop((box[2] - 1, 0, box[2], max_hw))
#             for i in range(box[2], max_hw):
#                 box1 = (i, 0, i + 1, max_hw)
#                 newimg.paste(col, box1)
#                 # loc[0]+=l
#                 # loc[2]+=l
#             loc = (loc[0] + l, loc[1], loc[2] + l, loc[3])
#         elif mid_lr < (width / 2):
#             l = 0
#             r = l + width
#             box = (l, 0, r, max_hw)
#             newimg.paste(img, box)
#
#             col = newimg.crop((box[2] - 1, 0, box[2], max_hw))
#             for i in range(box[2], max_hw):
#                 box1 = (i, 0, i + 1, max_hw)
#                 newimg.paste(col, box1)
#             # print(1)
#             # loc[0] += l
#             # loc[2] += l
#             loc = (loc[0] + l, loc[1], loc[2] + l, loc[3])
#         else: #1
#             l = max_hw - width
#             r = max_hw
#             box = (l, 0, r, max_hw)
#             newimg.paste(img, box)
#
#             col = newimg.crop((box[0], 0, box[0] + 1, max_hw))
#             for i in range(0, l):
#                 box1 = (i, 0, i + 1, max_hw)
#                 newimg.paste(col, box1)
#             # loc[0] += l
#             # loc[2] += l
#             loc = (loc[0] + l, loc[1], loc[2] + l, loc[3])
#     else:
#         if task == 'coat_length_labels' \
#                 or task == 'collar_design_labels' \
#                 or task == 'collar_design_labels' \
#                 or task == 'neck_design_labels' \
#                 or task == 'neckline_design_labels':
#             u = 0
#             l = u + img.size[1]
#             box = (0, u, max_hw, l)
#             newimg.paste(img, box)
#
#             col = newimg.crop((0, l - 1, max_hw, l))
#             for i in (range(l, max_hw)):
#                 box1 = (0, i, max_hw, i + 1)
#                 newimg.paste(col, box1)
#
#         else:
#             u = max_hw - img.size[1]
#             l = max_hw
#             box = (0, u, max_hw, l)
#             newimg.paste(img, box)
#
#             col = newimg.crop((0, u, max_hw, u + 1))
#             for i in (range(u)):
#                 box1 = (0, i, max_hw, i + 1)
#                 newimg.paste(col, box1)
#             # loc[1] += u
#             # loc[3] += u
#             loc = (loc[0], loc[1] + u, loc[2], loc[3] + u)
#     # plt.imshow(newimg)
#     # plt.show()
#     return newimg, loc
#
# def enhanced_padwithoutloc(img, task):
#     width, height = img.size
#     max_hw = max(img.size)
#     newimg = Image.new("RGB", (max_hw, max_hw), (0, 0, 0))
#
#     if max_hw == height:  # shu
#         l = (int)((max_hw - width) / 2)
#         r = l + width
#         box = (l, 0, r, max_hw)
#         newimg.paste(img, box)
#
#         col = newimg.crop((box[0], 0, box[0] + 1, max_hw))
#         for i in (range(box[0])):
#             box1 = (i, 0, i + 1, max_hw)
#             newimg.paste(col, box1)
#         col = newimg.crop((box[2] - 1, 0, box[2], max_hw))
#         for i in range(box[2], max_hw):
#             box1 = (i, 0, i + 1, max_hw)
#             newimg.paste(col, box1)
#     else:
#         if task == 'coat_length_labels' \
#                 or task == 'collar_design_labels' \
#                 or task == 'collar_design_labels' \
#                 or task == 'neck_design_labels' \
#                 or task == 'neckline_design_labels':
#             u = 0
#             l = u + img.size[1]
#             box = (0, u, max_hw, l)
#             newimg.paste(img, box)
#
#             col = newimg.crop((0, l - 1, max_hw, l))
#             for i in (range(l, max_hw)):
#                 box1 = (0, i, max_hw, i + 1)
#                 newimg.paste(col, box1)
#
#         else:
#             u = max_hw - img.size[1]
#             l = max_hw
#             box = (0, u, max_hw, l)
#             newimg.paste(img, box)
#
#             col = newimg.crop((0, u, max_hw, u + 1))
#             for i in (range(u)):
#                 box1 = (0, i, max_hw, i + 1)
#                 newimg.paste(col, box1)
#     return newimg
#
# def stretchingimg(img, task):
#     if np.random.uniform() > 0.5:
#         rnd = np.random.uniform(0.9, 1.0)
#         img = img.resize((int(img.width * rnd), img.height))
#         img = enhanced_padwithoutloc(img, task)
#     else:
#         rnd = np.random.uniform(0.9, 1.0)
#         img = img.resize((img.width, int(img.height * rnd)))
#         img = enhanced_padwithoutloc(img, task)
#     return img
#
# def cropwithloc(img, loc, task):
#     height = img.height
#     width = img.width
#     x1, y1, x2, y2 = loc
#
#     if x1 < 0:
#         x1 = 0
#     if y1 < 0:
#         y1 = 0
#     if x2 > width:
#         x2 = width
#     if y2 > height:
#         y2 = height
#
#     if task == 'coat_length_labels' \
#             or task == 'collar_design_labels' \
#             or task == 'collar_design_labels' \
#             or task == 'neck_design_labels' \
#             or task == 'neckline_design_labels':
#         y2 = height
#     else:
#         y1 = 0
#
#
#     person_height = y2 - y1
#     person_width = x2 - x1
#     if person_width > person_height:
#
#         left = x1 * (1-random.uniform(0, 1))
#         right = x2 + (width - x2) * random.uniform(0, 1)
#
#         person_width = right - left
#         #补上
#         if task == 'coat_length_labels' \
#                 or task == 'collar_design_labels' \
#                 or task == 'collar_design_labels' \
#                 or task == 'neck_design_labels' \
#                 or task == 'neckline_design_labels':
#             y1 = height - person_width
#
#             if x1 < 0:
#                 x1 = 0
#             if y1 < 0:
#                 y1 = 0
#             if x2 > width:
#                 x2 = width
#             if y2 > height:
#                 y2 = height
#             loc_1 = loc[1] - y1
#             loc_3 = loc[3] - y1
#             loc_0 = loc[0] - left
#             loc_2 = loc[2] - left
#             if (y2-y1)/height>0.99:
#                 return img.crop((int(left), int(y1), int(right), int(y2))), 2, (loc_0, loc_1, loc_2, loc_3)
#             else:
#                 return img.crop((int(left), int(y1), int(right), int(y2))), 1, (loc_0, loc_1, loc_2, loc_3)
#         #补下
#         else:
#             y2 = y1 + person_width
#             if x1 < 0:
#                 x1 = 0
#             if y1 < 0:
#                 y1 = 0
#             if x2 > width:
#                 x2 = width
#             if y2 > height:
#                 y2 = height
#             loc_1 = loc[1] - y1
#             loc_3 = loc[3] - y1
#             loc_0 = loc[0] - left
#             loc_2 = loc[2] - left
#         if (y2 - y1) / height > 0.99:
#             return img.crop((int(left), int(y1), int(right), int(y2))), 2, (loc_0, loc_1, loc_2, loc_3)
#         else:
#             return img.crop((int(left), int(y1), int(right), int(y2))), 1, (loc_0, loc_1, loc_2, loc_3)
#     else:
#         if y1 == 0:
#             return img, 2, loc
#         else:
#             if task == 'coat_length_labels' \
#                     or task == 'collar_design_labels' \
#                     or task == 'collar_design_labels' \
#                     or task == 'neck_design_labels' \
#                     or task == 'neckline_design_labels':
#                 y1 = random.uniform(0, 1) * y1
#                 # y2 = y2 + (height - y2) * random.uniform(0, 1)
#                 person_height = y2 - y1
#
#                 left1 = x2 - person_height
#                 left2 = x1
#                 if (left2 + person_height) > width:
#                     left2 = width - person_height
#                 if left1 < 0:
#                     left1 = 0
#                 if left2 > width:
#                     left2 = width
#                 left = left1 + (left2 - left1) * random.uniform(0, 1)
#
#                 if x1 < 0:
#                     x1 = 0
#                 if y1 < 0:
#                     y1 = 0
#                 if x2 > width:
#                     x2 = width
#                 if y2 > height:
#                     y2 = height
#                 loc_0 = loc[0] - left
#                 loc_2 = loc[2] - left
#                 loc_1 = loc[1] - y1
#                 loc_3 = loc[3] - y1
#                 if (y2 - y1) / height > 0.99:
#                     return img.crop((int(left), int(y1), int(left + person_height), int(y2))), 2, (loc_0, loc_1, loc_2, loc_3)
#                 else:
#                     return img.crop((int(left), int(y1), int(left + person_height), int(y2))), 0, (loc_0, loc_1, loc_2, loc_3)
#             else:
#                 # y1 = random.uniform(0, 1) * y1
#                 y1 = 0
#                 y2 = y2 + (height - y2) * random.uniform(0, 1)
#                 person_height = y2 - y1
#
#                 left1 = x2 - person_height
#                 left2 = x1
#                 if (left2 + person_height) > width:
#                     left2 = width - person_height
#                 if left1 < 0:
#                     left1 = 0
#                 if left2 > width:
#                     left2 = width
#                 left = left1 + (left2 - left1) * random.uniform(0, 1)
#
#                 if x1 < 0:
#                     x1 = 0
#                 if y1 < 0:
#                     y1 = 0
#                 if x2 > width:
#                     x2 = width
#                 if y2 > height:
#                     y2 = height
#                 loc_0 = loc[0] - left
#                 loc_2 = loc[2] - left
#                 loc_1 = loc[1] - y1
#                 loc_3 = loc[3] - y1
#                 if (y2 - y1) / height > 0.99:
#                     return img.crop((int(left), int(y1), int(left + person_height), int(y2))), 2, (
#                     loc_0, loc_1, loc_2, loc_3)
#                 else:
#                     return img.crop((int(left), int(y1), int(left + person_height), int(y2))), 0, (
#                     loc_0, loc_1, loc_2, loc_3)
#
#
# def cropwithloc_test(img, loc, task):
#     p1 = 0.5
#     p2 = 0.5
#     p3 = 0.5
#     height = img.height
#     width = img.width
#     x1, y1, x2, y2 = loc
#
#     if x1 < 0:
#         x1 = 0
#     if y1 < 0:
#         y1 = 0
#     if x2 > width:
#         x2 = width
#     if y2 > height:
#         y2 = height
#
#     if task == 'coat_length_labels' \
#             or task == 'collar_design_labels' \
#             or task == 'collar_design_labels' \
#             or task == 'neck_design_labels' \
#             or task == 'neckline_design_labels':
#         y2 = height
#     else:
#         y1 = 0
#
#     person_height = y2 - y1
#     person_width = x2 - x1
#     if person_width > person_height:
#
#         left = x1 * (1 - p1)
#         right = x2 + (width - x2) * p1
#
#         person_width = right - left
#         # 补上
#         if task == 'coat_length_labels' \
#                 or task == 'collar_design_labels' \
#                 or task == 'collar_design_labels' \
#                 or task == 'neck_design_labels' \
#                 or task == 'neckline_design_labels':
#             y1 = height - person_width
#
#             if x1 < 0:
#                 x1 = 0
#             if y1 < 0:
#                 y1 = 0
#             if x2 > width:
#                 x2 = width
#             if y2 > height:
#                 y2 = height
#             loc_1 = loc[1] - y1
#             loc_3 = loc[3] - y1
#             loc_0 = loc[0] - left
#             loc_2 = loc[2] - left
#             if (y2 - y1) / height > 0.99:
#                 return img.crop((int(left), int(y1), int(right), int(y2))), 2, (loc_0, loc_1, loc_2, loc_3)
#             else:
#                 return img.crop((int(left), int(y1), int(right), int(y2))), 1, (loc_0, loc_1, loc_2, loc_3)
#         # 补下
#         else:
#             y2 = y1 + person_width
#             if x1 < 0:
#                 x1 = 0
#             if y1 < 0:
#                 y1 = 0
#             if x2 > width:
#                 x2 = width
#             if y2 > height:
#                 y2 = height
#             loc_1 = loc[1] - y1
#             loc_3 = loc[3] - y1
#             loc_0 = loc[0] - left
#             loc_2 = loc[2] - left
#         if (y2 - y1) / height > 0.99:
#             return img.crop((int(left), int(y1), int(right), int(y2))), 2, (loc_0, loc_1, loc_2, loc_3)
#         else:
#             return img.crop((int(left), int(y1), int(right), int(y2))), 1, (loc_0, loc_1, loc_2, loc_3)
#     else:
#         if y1 == 0:
#             return img, 2, loc
#         else:
#             if task == 'coat_length_labels' \
#                     or task == 'collar_design_labels' \
#                     or task == 'collar_design_labels' \
#                     or task == 'neck_design_labels' \
#                     or task == 'neckline_design_labels':
#                 y1 = (1-p2) * y1
#                 # y2 = y2 + (height - y2) * random.uniform(0, 1)
#                 person_height = y2 - y1
#
#                 left1 = x2 - person_height
#                 left2 = x1
#                 if (left2 + person_height) > width:
#                     left2 = width - person_height
#                 if left1 < 0:
#                     left1 = 0
#                 if left2 > width:
#                     left2 = width
#                 left = left1 + (left2 - left1) * 0.5
#
#                 if x1 < 0:
#                     x1 = 0
#                 if y1 < 0:
#                     y1 = 0
#                 if x2 > width:
#                     x2 = width
#                 if y2 > height:
#                     y2 = height
#                 loc_0 = loc[0] - left
#                 loc_2 = loc[2] - left
#                 loc_1 = loc[1] - y1
#                 loc_3 = loc[3] - y1
#                 if (y2 - y1) / height > 0.99:
#                     return img.crop((int(left), int(y1), int(left + person_height), int(y2))), 2, (
#                     loc_0, loc_1, loc_2, loc_3)
#                 else:
#                     return img.crop((int(left), int(y1), int(left + person_height), int(y2))), 0, (
#                     loc_0, loc_1, loc_2, loc_3)
#             else:
#                 # y1 = random.uniform(0, 1) * y1
#                 y1 = 0
#                 y2 = y2 + (height - y2) * p3
#                 person_height = y2 - y1
#
#                 left1 = x2 - person_height
#                 left2 = x1
#                 if (left2 + person_height) > width:
#                     left2 = width - person_height
#                 if left1 < 0:
#                     left1 = 0
#                 if left2 > width:
#                     left2 = width
#                 left = left1 + (left2 - left1) * 0.5
#
#                 if x1 < 0:
#                     x1 = 0
#                 if y1 < 0:
#                     y1 = 0
#                 if x2 > width:
#                     x2 = width
#                 if y2 > height:
#                     y2 = height
#                 loc_0 = loc[0] - left
#                 loc_2 = loc[2] - left
#                 loc_1 = loc[1] - y1
#                 loc_3 = loc[3] - y1
#                 if (y2 - y1) / height > 0.99:
#                     return img.crop((int(left), int(y1), int(left + person_height), int(y2))), 2, (
#                         loc_0, loc_1, loc_2, loc_3)
#                 else:
#                     return img.crop((int(left), int(y1), int(left + person_height), int(y2))), 0, (
#                         loc_0, loc_1, loc_2, loc_3)
#
#
# def bubian(img):
#     max_hw = max(img.size)
#     nimg = np.asarray(img)
#     col = nimg[:, 0]
#     ncol = np.asarray(col)
#
#     r = ncol[:, 0]
#     g = ncol[:, 1]
#     b = ncol[:, 2]
#     r = np.median(r).astype(int)
#     g = np.median(g).astype(int)
#     b = np.median(b).astype(int)
#
#     newimg = Image.new("RGB", (max_hw, max_hw), (r, g, b))
#     if max_hw == img.size[1]:
#         l = (int)((max_hw - img.size[0]) / 2)
#         r = l + img.size[0]
#         box = (l, 0, r, max_hw)
#     else:
#         u = (int)((max_hw - img.size[1]) / 2)
#         l = u + img.size[1]
#         box = (0, u, max_hw, l)
#     newimg.paste(img, box)
#     return newimg
#
#
# def randomlc(img):
#     rate = random.uniform(0.9, 1.1)
#     h = (int)(img.size[1] * rate)
#     img = img.resize((img.size[0], h))
#
#     return img
#
#
# def test_cropperson(img, loc):
#     height = img.height
#     width = img.width
#     x1, y1, x2, y2 = loc
#
#     if x1 < 0:
#         x1 = 0
#     if y1 < 0:
#         y1 = 0
#     if x2 > width:
#         x2 = width
#     if y2 > height:
#         y2 = height
#
#     personw = (x2 - x1)
#     wl = 0.1 * personw
#     wr = 0.1 * personw
#     x1 = x1 - wl
#     x2 = x2 + wr
#     y1 -= 20
#     y2 = height
#
#     if x1 < 0:
#         x1 = 0
#     if y1 < 0:
#         y1 = 0
#     if x2 > width:
#         x2 = width
#     if y2 > height:
#         y2 = height
#
#     img = img.crop((x1, y1, x2, y2))
#
#     return img
#
#
# def test_cropperson1(img, loc, mean):
#     height = img.height
#     width = img.width
#     x1, y1, x2, y2 = loc
#
#     if x1 < 0:
#         x1 = 0
#     if y1 < 0:
#         y1 = 0
#     if x2 > width:
#         x2 = width
#     if y2 > height:
#         y2 = height
#
#     personw = (x2 - x1)
#     wl = 0.2 * personw
#     wr = 0.2 * personw
#     x1 = x1 - wl
#     x2 = x2 + wr
#     y2 = height
#
#     if x1 < 0:
#         x1 = 0
#     if y1 < 0:
#         y1 = 0
#     if x2 > width:
#         x2 = width
#     if y2 > height:
#         y2 = height
#
#     x1 = (int)(x1)
#     x2 = (int)(x2)
#     y1 = (int)(y1)
#     y2 = (int)(y2)
#     cropimg = img.crop((x1, y1, x2, y2))
#
#     newimg = Image.new("RGB", (img.width, img.height),
#                        ((int)(mean[0] * 255), (int)(mean[1] * 255), (int)(mean[2] * 255)))
#     newimg.paste(cropimg, (x1, y1, x2, y2))
#
#     return newimg
#
#
# def cropperson(img, loc):
#     height = img.height
#     width = img.width
#     x1, y1, x2, y2 = loc
#
#     if x1 < 0:
#         x1 = 0
#     if y1 < 0:
#         y1 = 0
#     if x2 > width:
#         x2 = width
#     if y2 > height:
#         y2 = height
#
#     personw = (x2 - x1)
#     wl = random.uniform(0.0, 0.2) * personw
#     wr = random.uniform(0.0, 0.2) * personw
#     x1 = x1 - wl
#     x2 = x2 + wr
#     y1 -= 20
#     y2 = height
#
#     if x1 < 0:
#         x1 = 0
#     if y1 < 0:
#         y1 = 0
#     if x2 > width:
#         x2 = width
#     if y2 > height:
#         y2 = height
#
#     img = img.crop((x1, y1, x2, y2))
#
#     # max_hw = max(img.size)
#     #
#     # newimg = Image.new("RGB", (max_hw, max_hw), ((int)(mean[0] * 255), (int)(mean[1] * 255), (int)(mean[2] * 255)))
#     # if max_hw == img.size[1]:
#     #     l = (int)((max_hw - cropimg.size[0]) / 2)
#     #     r = l + img.size[0]
#     #     box = (l, 0, r, max_hw)
#     # else:
#     #     u = (int)((max_hw - cropimg.size[1]) / 2)
#     #     l = u + img.size[1]
#     #     box = (0, u, max_hw, l)
#     # newimg.paste(img, box)
#
#     return img


# def skirt_cropperson(img, loc):
#     height = img.height
#     width = img.width
#     x1, y1, x2, y2 = loc
#
#     if x1 < 0:
#         x1 = 0
#     if y1 < 0:
#         y1 = 0
#     if x2 > width:
#         x2 = width
#     if y2 > height:
#         y2 = height
#
#     personw = (x2 - x1)
#     x1 = x1 - personw*0.1
#     x2 = x2 + personw * 0.1
#     y1 -=20
#
#     if x1 < 0:
#         x1 = 0
#     if y1 < 0:
#         y1 = 0
#     if x2 > width:
#         x2 = width
#     if y2 > height:
#         y2 = height
#
#     x1 = x1 * random.uniform(0, 1)
#     x2 = x2 + (width-x2) * random.uniform(0, 1)
#     y1 = y1 - (height-y1) * random.uniform(0, 1)
#     y2 = height
#
#     if x1 < 0:
#         x1 = 0
#     if y1 < 0:
#         y1 = 0
#     if x2 > width:
#         x2 = width
#     if y2 > height:
#         y2 = height
#     x1 = int(x1)
#     x2 = int(x2)
#     y1 = int(y1)
#     y2 = int(y2)
#     img = img.crop((x1, y1, x2, y2))
#
#     return img
#
#
# def skirt_Testcropperson(img, loc):
#     height = img.height
#     width = img.width
#     x1, y1, x2, y2 = loc
#
#     if x1 < 0:
#         x1 = 0
#     if y1 < 0:
#         y1 = 0
#     if x2 > width:
#         x2 = width
#     if y2 > height:
#         y2 = height
#
#     personw = (x2 - x1)
#     x1 = x1 - personw * 0.1
#     x2 = x2 + personw * 0.1
#     y1 -= 20
#
#     if x1 < 0:
#         x1 = 0
#     if y1 < 0:
#         y1 = 0
#     if x2 > width:
#         x2 = width
#     if y2 > height:
#         y2 = height
#
#     x1 = x1 * 0.5
#     x2 = x2 + (width - x2) * 0.5
#     y1 = y1 - (height - y1) * 0.5
#     y2 = height
#
#     if x1 < 0:
#         x1 = 0
#     if y1 < 0:
#         y1 = 0
#     if x2 > width:
#         x2 = width
#     if y2 > height:
#         y2 = height
#     x1 = int(x1)
#     x2 = int(x2)
#     y1 = int(y1)
#     y2 = int(y2)
#     img = img.crop((x1, y1, x2, y2))
#
#     return img
# def skirt_cropperson(img, loc):
#     height = img.height
#     width = img.width
#     x1, y1, x2, y2 = loc
#
#     if x1 < 0:
#         x1 = 0
#     if y1 < 0:
#         y1 = 0
#     if x2 > width:
#         x2 = width
#     if y2 > height:
#         y2 = height
#
#     personw = (x2 - x1)
#     x1 = x1 - personw * 0.1
#     x2 = x2 + personw * 0.1
#     y1 -= 20
#
#     if x1 < 0:
#         x1 = 0
#     if y1 < 0:
#         y1 = 0
#     if x2 > width:
#         x2 = width
#     if y2 > height:
#         y2 = height
#
#     x1 = x1 * random.uniform(0.8, 1.0)
#     x2 = x2 + (width - x2) * random.uniform(0, 0.2)
#     y1 = y1 - (height - y1) * random.uniform(0, 0.2)
#     x1 = x1
#     x2 = x2
#     y1 = y1
#     y2 = height
#
#     if x1 < 0:
#         x1 = 0
#     if y1 < 0:
#         y1 = 0
#     if x2 > width:
#         x2 = width
#     if y2 > height:
#         y2 = height
#     x1 = int(x1)
#     x2 = int(x2)
#     y1 = int(y1)
#     y2 = int(y2)
#     img = img.crop((x1, y1, x2, y2))
#
#     return img
#
#
# def skirt_Testcropperson(img, loc):
#     height = img.height
#     width = img.width
#     x1, y1, x2, y2 = loc
#
#     if x1 < 0:
#         x1 = 0
#     if y1 < 0:
#         y1 = 0
#     if x2 > width:
#         x2 = width
#     if y2 > height:
#         y2 = height
#
#     personw = (x2 - x1)
#     x1 = x1 - personw * 0.1
#     x2 = x2 + personw * 0.1
#     y1 -= 20
#
#     if x1 < 0:
#         x1 = 0
#     if y1 < 0:
#         y1 = 0
#     if x2 > width:
#         x2 = width
#     if y2 > height:
#         y2 = height
#
#     x1 = x1 * 0.9
#     x2 = x2 + (width - x2) * 0.1
#     y1 = y1 - (height - y1) * 0.1
#     y2 = height
#
#     if x1 < 0:
#         x1 = 0
#     if y1 < 0:
#         y1 = 0
#     if x2 > width:
#         x2 = width
#     if y2 > height:
#         y2 = height
#     x1 = int(x1)
#     x2 = int(x2)
#     y1 = int(y1)
#     y2 = int(y2)
#     img = img.crop((x1, y1, x2, y2))
#
#     return img
#
#
# def cropperson1(img, loc, mean):
#     height = img.height
#     width = img.width
#     x1, y1, x2, y2 = loc
#
#     if x1 < 0:
#         x1 = 0
#     if y1 < 0:
#         y1 = 0
#     if x2 > width:
#         x2 = width
#     if y2 > height:
#         y2 = height
#
#     personw = (x2 - x1)
#     wl = random.uniform(0.1, 0.3) * personw
#     wr = random.uniform(0.1, 0.3) * personw
#     x1 = x1 - wl
#     x2 = x2 + wr
#     y1 -= 20
#     y2 = height
#
#     if x1 < 0:
#         x1 = 0
#     if y1 < 0:
#         y1 = 0
#     if x2 > width:
#         x2 = width
#     if y2 > height:
#         y2 = height
#     x1 = (int)(x1)
#     x2 = (int)(x2)
#     y1 = (int)(y1)
#     y2 = (int)(y2)
#     cropimg = img.crop((x1, y1, x2, y2))
#
#     newimg = Image.new("RGB", (img.width, img.height),
#                        ((int)(mean[0] * 255), (int)(mean[1] * 255), (int)(mean[2] * 255)))
#     newimg.paste(cropimg, (x1, y1, x2, y2))
#     # max_hw = max(img.size)
#     #
#     # newimg = Image.new("RGB", (max_hw, max_hw), ((int)(mean[0] * 255), (int)(mean[1] * 255), (int)(mean[2] * 255)))
#     # if max_hw == img.size[1]:
#     #     l = (int)((max_hw - cropimg.size[0]) / 2)
#     #     r = l + img.size[0]
#     #     box = (l, 0, r, max_hw)
#     # else:
#     #     u = (int)((max_hw - cropimg.size[1]) / 2)
#     #     l = u + img.size[1]
#     #     box = (0, u, max_hw, l)
#     # newimg.paste(img, box)
#
#     return newimg
#
#
# def pad(img, mean):
#     max_hw = max(img.size)
#
#     newimg = Image.new("RGB", (max_hw, max_hw), ((int)(mean[0] * 255), (int)(mean[1] * 255), (int)(mean[2] * 255)))
#     if max_hw == img.size[1]:
#         l = (int)((max_hw - img.size[0]) / 2)
#         r = l + img.size[0]
#         box = (l, 0, r, max_hw)
#     else:
#         u = (int)((max_hw - img.size[1]) / 2)
#         l = u + img.size[1]
#         box = (0, u, max_hw, l)
#     newimg.paste(img, box)
#     return newimg
#
#
# def skirt_Testcropperson1(img, loc):
#     height = img.height
#     width = img.width
#     x1, y1, x2, y2 = loc
#
#     if x1 < 0:
#         x1 = 0
#     if y1 < 0:
#         y1 = 0
#     if x2 > width:
#         x2 = width
#     if y2 > height:
#         y2 = height
#
#     personw = (x2 - x1)
#     x1 = x1 - personw * 0.1
#     x2 = x2 + personw * 0.1
#     y1 -= 20
#
#     if x1 < 0:
#         x1 = 0
#     if y1 < 0:
#         y1 = 0
#     if x2 > width:
#         x2 = width
#     if y2 > height:
#         y2 = height
#
#     x1 = x1
#     x2 = x2 + (width - x2) * 0
#     y1 = y1 - (height - y1) * 0
#     y2 = height
#
#     if x1 < 0:
#         x1 = 0
#     if y1 < 0:
#         y1 = 0
#     if x2 > width:
#         x2 = width
#     if y2 > height:
#         y2 = height
#     x1 = int(x1)
#     x2 = int(x2)
#     y1 = int(y1)
#     y2 = int(y2)
#     img = img.crop((x1, y1, x2, y2))
#
#     return img

# if __name__ == '__main__':
#     srcimg = Image.open('a08a7f1fc1695630f790b579e837bb58.jpg')
#     enhanced_pad(srcimg, (131.43889911358173,68.99142661461462,388.5497577373798,788.2641454843374), 'sleeve_length_labels')
