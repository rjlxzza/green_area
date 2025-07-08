#export NCCL_NET=IB
deepspeed --hostfile hostfile train.py
