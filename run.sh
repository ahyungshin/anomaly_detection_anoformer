#!/bin/bash

cd experiments/

test=1  # 0 means train the model, 1 means evaluate the model
threshold=0.02
fold_cnt=1

dataroot="/path/to/dataset/ann_gun_CentroidA"
outf="./output"
model="anoformer"
dataset="ann_gun_CentroidA" #"neurips_ts" or "power_data" or "mit_bih" or "ann_gun_CentroidA"

ngpu=1
gpu_ids='1'
n_aug=0
bs=64
niter=500
lr=0.0001
isize=200 # 100 for neurips_ts and power / 320 for mit_bih / 200 for 2d-gesture
mask_rate=0.5
mask_len=10
ntoken=200
emsize=128
nhead=8
nlayer_g=9
nlayer_d=6

for (( i=0; i<$fold_cnt; i+=1))
do
    echo "#################################"
    echo "########  Folder $i  ############"
    if [ $test = 0 ]; then
	    python -u main.py  \
            --dataroot $dataroot \
            --dataset $dataset \
            --model $model \
            --niter $niter \
            --lr $lr \
            --outf  $outf \
            --folder $i \
            --batchsize $bs \
            --n_aug $n_aug \
            --ngpu $ngpu \
            --gpu_ids $gpu_ids \
            --isize $isize \
            --mask_rate $mask_rate \
            --mask_len $mask_len \
            --ntoken $ntoken \
            --emsize $emsize \
            --nhead $nhead \
            --nlayer_g $nlayer_g \
            --nlayer_d $nlayer_d \

	else
	    python -u main.py  \
            --istest  \
            --dataroot $dataroot \
            --dataset $dataset \
            --model $model \
            --niter $niter \
            --lr $lr \
            --outf  $outf \
            --folder $i  \
            --batchsize $bs \
            --threshold $threshold \
            --isize $isize \
            --mask_rate $mask_rate \
            --ntoken $ntoken \
            --mask_len $mask_len \
            --nlayer_g $nlayer_g \
            --nlayer_d $nlayer_d \

    fi

done
