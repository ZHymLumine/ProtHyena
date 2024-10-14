python -m train wandb=null experiment=hg38/genomic_benchmark_scratch

# for pretrain
/home/lr/zym/.pyenv/versions/anaconda3-2023.09-0/envs/hyena-dna/bin/python -m train wandb=null experiment=hg38/hg38_hyena model.d_model=128 model.n_layer=2 dataset.batch_size=256 train.global_batch_size=256 dataset.max_length=1024 optimizer.lr=6e-4 trainer.devices=1

CUDA_VISIBLE_DEVICES=2,3 python -m train experiment=hg38/hg38_hyena model.d_model=128 model.n_layer=2 dataset.batch_size=256 train.global_batch_size=256 dataset.max_length=1024 optimizer.lr=6e-4 trainer.devices=2
CUDA_VISIBLE_DEVICES=0,1 python -m train experiment=prot14m/prot14m_hyena trainer.devices=2
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr="192.168.0.47" \
    --master_port=8888 \
    train.py --config_path=configs/wandb=null experiment=rna32m/rna32m_hyena model.d_model=128 model.n_layer=2 dataset.batch_size=256 train.global_batch_size=256 dataset.max_length=1024 optimizer.lr=6e-4 trainer.devices=2

torchrun --nproc_per_node=2 --nnodes=2 --node_rank=0 --master_addr="192.168.0.47" --master_port=8888 train.py --config_path=configs/wandb=null experiment=rna32m/rna32m_hyena model.d_model=128 model.n_layer=2 dataset.batch_size=256 train.global_batch_size=256 dataset.max_length=1024 optimizer.lr=6e-4

python -m train experiment=prot14m/location trainer.devices=1

CUDA_VISIBLE_DEVICES=0 python -m train experiment=prot14m/disorder trainer.devices=1
CUDA_VISIBLE_DEVICES=1 python -m train experiment=prot14m/cleavage_attn trainer.devices=1 optimizer.lr=6e-4

CUDA_VISIBLE_DEVICES=0 python -m train experiment=prot14m/prot14m_large trainer.devices=1

CUDA_VISIBLE_DEVICES=0 python -m train experiment=prot14m/fold_class trainer.devices=1
CUDA_VISIBLE_DEVICES=0 python -m train experiment=prot14m/signalP trainer.devices=1

CUDA_VISIBLE_DEVICES=0 python -m train wandb=null experiment=prot14m/stability trainer.devices=1 dataset.batch_size=128 dataset.batch_size_eval=128

CUDA_VISIBLE_DEVICES=1 python -m train experiment=prot14m/fluorescence trainer.devices=1 dataset.batch_size=128 dataset.batch_size_eval=128
