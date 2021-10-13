# ----------------- miniImageNet ----------------- #
python3  -m torch.distributed.launch --nproc_per_node 2 --master_port 12345 main_multi_gpu.py -s params --dataset miniImageNet --transform A

# ----------------- tieredImageNet ----------------- #
python3  -m torch.distributed.launch --nproc_per_node 2 --master_port 12345 main_multi_gpu.py -s params --dataset tieredImageNet --transform B --epochs 200

# ----------------- CIFAR-FS ----------------- #
python3  -m torch.distributed.launch --nproc_per_node 2 --master_port 12345 main_multi_gpu.py -s params --dataset CIFAR-FS --transform D

# ----------------- FC100 ----------------- #
python3  -m torch.distributed.launch --nproc_per_node 2 --master_port 12345 main_multi_gpu.py -s params --dataset FC100 --transform C

# ----------------- CUB ----------------- #
python3  -m torch.distributed.launch --nproc_per_node 2 --master_port 12345 main_multi_gpu.py -s params --dataset CUB --transform E
