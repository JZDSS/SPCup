#!/bin/bash

root=../spcup
meta_dir=../pre/meta
out_dir=../pre
count=0
for file in `ls $root`
do
    python gen_patches.py --data_dir $root/$file  --meta_dir $meta_dir --out_dir $out_dir/$file --patch_size 64 \
    --max_patches 100 --out_file out_$file --all_label $count
    let count++
done