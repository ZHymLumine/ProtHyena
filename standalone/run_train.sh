python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr="192.168.0.59" \
    --master_port=1234 \
    train.py

# 在第二个节点
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr="192.168.0.59" \
    --master_port=1234 \
    train.py

torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr="192.168.0.59" --master_port=8888 train_standalone2.py --do_train --do_eval --overwrite_cache