import csv
import numpy as np
import time

def average(arclist):
    list_reader = []

    name = ''
    # for arc in arclist:
    #     name =name + arc
    #     name = name + '_'
    name = time.strftime("%Y-%m-%d-%H-%M", time.localtime(time.time()))
    f_out = open('submission/%smean.csv' % name, 'w')
    for arc in arclist:
        with open("submission/%s_submission.csv" % arc, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = [row for row in reader]
            list_reader.append(rows)

    for i in range(len(list_reader[0])):
        list_sum = []
        for j in range(len(list_reader)):
            #             for row in list_reader[j]:
            rowlist = list_reader[j][i]
            list_sum.append(rowlist[2].split(";"))
        #         print(list_sum)
        np_rowlist = np.asarray(list_sum, dtype=np.float32)
        np_row_mean = np.mean(np_rowlist, axis=0)
        list_row_mean = np_row_mean.tolist()
        pred_out = ';'.join(["%.8f" % (o) for o in list_row_mean])
        line_out = ','.join([list_reader[j][i][0], list_reader[j][i][1], pred_out])
        f_out.write(line_out + '\n')
    f_out.close()


arc_list = {
    '2_inceptionresnetv2_flip',
    '2_inceptionresnetv2_original',
    '2_inceptionresnetv2_padwithnoloc_T1_f',
    '2_inceptionresnetv2_padwithnoloc_T1',
    'inceptionv4_flip',
    'inceptionv4_original',
    'inceptionv4_padwithnoloc_T1_f',
    'inceptionv4_padwithnoloc_T1',
    'nasnetalarge_flip',
    'nasnetalarge_original',
    'nasnetalarge_padwithnoloc_T1_f',
    'nasnetalarge_padwithnoloc_T1',
    # 'inceptionresnetv2_flip',
    # 'inceptionresnetv2_original',
    # 'inceptionresnetv2_padwithnoloc_T1_f',
    # 'inceptionresnetv2_padwithnoloc_T1',
    # 'se_resnext50_32x4d_flip',
    # 'se_resnext50_32x4d_original',
    # 'se_resnext50_32x4d_padwithnoloc_T1_f',
    # 'se_resnext50_32x4d_padwithnoloc_T1',
}

average(arc_list)
