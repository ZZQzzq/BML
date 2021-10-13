# ----------------- miniImageNet ----------------- #
python3  -m torch.distributed.launch --nproc_per_node 2 --master_port 12346 main_multi_gpu.py --mode local -s params_baseline --dataset miniImageNet --transform A
python3  -m torch.distributed.launch --nproc_per_node 2 --master_port 12346 main_multi_gpu.py --mode global -s params_baseline --dataset miniImageNet --transform A

# ----------------- tieredImageNet ----------------- #
python3  -m torch.distributed.launch --nproc_per_node 2 --master_port 12342 main_multi_gpu.py --mode local -s params_baseline --dataset tieredImageNet --transform B
python3  -m torch.distributed.launch --nproc_per_node 2 --master_port 12342 main_multi_gpu.py --mode global -s params_baseline --dataset tieredImageNet --transform B

# ----------------- CIFAR-FS ----------------- #
python3  -m torch.distributed.launch --nproc_per_node 2 --master_port 12342 main_multi_gpu.py --mode local -s params_baseline --dataset CIFAR-FS --transform D
python3  -m torch.distributed.launch --nproc_per_node 2 --master_port 12342 main_multi_gpu.py --mode global -s params_baseline --dataset CIFAR-FS --transform D

# ----------------- FC100 ----------------- #
python3  -m torch.distributed.launch --nproc_per_node 2 --master_port 12342 main_multi_gpu.py --mode local -s params_baseline --dataset FC100 --transform C
python3  -m torch.distributed.launch --nproc_per_node 2 --master_port 12342 main_multi_gpu.py --mode global -s params_baseline --dataset FC100 --transform C

# ----------------- CUB ----------------- #
python3  -m torch.distributed.launch --nproc_per_node 2 --master_port 12342 main_multi_gpu.py --mode local -s params_baseline --dataset CUB --transform E
python3  -m torch.distributed.launch --nproc_per_node 2 --master_port 12342 main_multi_gpu.py --mode global -s params_baseline --dataset CUB --transform E
