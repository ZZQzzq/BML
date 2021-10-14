#!/bin/bash
# ----------------- eval ----------------- #
for seed_id in 0 1 2 3 4 5 6 7 8 9
do
  python3  -m torch.distributed.launch --nproc_per_node 1 --master_port 12347 main_multi_gpu.py -s params --dataset $1 --transform $2 --ckp_path $3 --is_eval --seed $seed_id --n_shots $4
done

# ----------------- mean score ----------------- #
python3  utils/util.py --dataset $1
