#!/usr/bin/env bash
num_epochs=13
model="se_resnext101_32x4d"
batchsize=32

python train.py --task skirt_length_labels --arch $model --epochs $num_epochs -b $batchsize
python train.py --task collar_design_labels --arch $model --epochs $num_epochs -b $batchsize
python train.py --task coat_length_labels --arch $model --epochs $num_epochs -b $batchsize
python train.py --task pant_length_labels --arch $model --epochs $num_epochs -b $batchsize
python train.py --task sleeve_length_labels --arch $model --epochs $num_epochs -b $batchsize
python train.py --task neckline_design_labels --arch $model --epochs $num_epochs -b $batchsize
python train.py --task neck_design_labels --arch $model --epochs $num_epochs -b $batchsize
python train.py --task lapel_design_labels --arch $model --epochs $num_epochs -b $batchsize
