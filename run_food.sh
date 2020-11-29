#!/bin/bash
initial_pool='200'
budget='500'
split='0'
# declare -a arr=("rnd" "us" "us_grad" "us_geometry" "us_ecl" "us_centroid")

for n in $initial_pool
do
    for p in $budget
    do
        for s in "${arr[@]}"
        do
            for sp in $split
            do
                # echo $n $p $s $sp $(($sp+1))
                CUDA_VISIBLE_DEVICES=0 python3 food.py --initial_pool=$n --budget=$p --sampling=$s --m=$sp --n=$(($sp+1))
            done
        done
    done
done
