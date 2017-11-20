#!/bin/bash
python ../gen_patches.py --data_dir /home/amax/data/SPCup2018 --out_dir ../data --meta_dir ../data/meta --out_file gen_patches
# 8 layers resnet
python ../train_new_resnet.py --gpu 3 --data_dir data --ckpt_dir 8lres/ckpt --log_dir 8lres/logs --blocks 1 --out_file 8lres/train_log
python ../test_img_level.py --gpu 3 --data_dir /home/amax/data/SPCup2018 --meta_dir ../data/meta --ckpt_dir 8lrs/ckpt --blocks 1 --out_file 8lres/valid_log
# 20 layers resnet
python ../train_new_resnet.py --gpu 3 --data_dir data --ckpt_dir 20lres/ckpt --log_dir 20lres/logs --blocks 3 --out_file 20lres/train_log
python ../test_img_level.py --gpu 3 --data_dir /home/amax/data/SPCup2018 --meta_dir ../data/meta --ckpt_dir 20lrs/ckpt --blocks 3 --out_file 20lres/valid_log
# 32 layers resnet
python ../train_new_resnet.py --gpu 3 --data_dir data --ckpt_dir 32lres/ckpt --log_dir 32lres/logs --blocks 5 --out_file 32lres/train_log
python ../test_img_level.py --gpu 3 --data_dir /home/amax/data/SPCup2018 --meta_dir ../data/meta --ckpt_dir 32lrs/ckpt --blocks 5 --out_dir 32lres/valid_log
