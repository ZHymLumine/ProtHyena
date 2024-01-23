from datetime import datetime
import logging
import torch.optim as optim
import time
import torch
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange
from typing import Optional
from functools import partial, reduce
from torch import Tensor
from torchvision.ops import StochasticDepth
from collections import namedtuple
import json
import os
import subprocess
import transformers
import argparse
import torch.multiprocessing as mp
from transformers import PreTrainedModel, AutoModelForCausalLM, PretrainedConfig
from standalone_hyenadna import HyenaDNAModel, CharacterTokenizer, HyenaDNAPreTrainedModel
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from utils import *
from dataset.language_modeling import (
    LineByLineTextDatasetCached, LineByLineTextDatasetChunksCached, LineByLineTextDataset, TextDataset
)

"""
We provide simple training code for the GenomicBenchmark datasets.
"""

#@title GenomicBenchmark dataset

"""
The GenomicBenchmarks dataset will automatically download to /contents on colab.
There are 8 datasets to choose from.

"""

from random import random
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

# 配置 Logger
logging.basicConfig(filename='training_log.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


RANDOM_TOKEN_PROB = 0.10
MASK_TOKEN_ID = 3
PAD_TOKEN_ID = 4
MASK_IGNORE_TOKEN_IDS = [0, 1, 2, 3, 4, 5, 6]

# get the random prob matrix and True means smaller than prob threshold
def prob_mask_like(t, prob):
    return torch.zeros_like(t).float().uniform_(0, 1) < prob

# get the mask matrix which cannot be masked
def mask_with_tokens(t, token_ids):
    init_no_mask = torch.full_like(t, False, dtype=torch.bool)
    mask = reduce(lambda acc, el: acc | (t == el), token_ids, init_no_mask)
    return mask

def get_mask_subset_with_prob(mask, prob):
    batch, seq_len, device = *mask.shape, mask.device
    max_masked = math.ceil(prob * seq_len)      # num of mask of a single sequence in average
    num_tokens = mask.sum(dim=-1, keepdim=True)     # num of pure tokens of each sequence except special tokens
    mask_excess = torch.cat((torch.zeros(0), torch.arange(mask.size(-1)).repeat(mask.size(0)))).reshape(mask.size(0),mask.size(-1)).to(device)
    mask_excess = (mask_excess >= (num_tokens * prob).ceil())        # only 15% of pure tokens can be masked
    mask_excess = mask_excess[:, :max_masked]       # get difference between 15% of pure tokens and 15% of all tokens
    rand = torch.rand((batch, seq_len), device=device).masked_fill(~mask, -1e9)     # rand (0-1) as prob, special token use -1e9
    _, sampled_indices = rand.topk(max_masked, dim=-1)      # get index of topk prob to mask
    sampled_indices = (sampled_indices + 1).masked_fill_(mask_excess, 0)        # delete difference of mask not pure
    new_mask = torch.zeros((batch, seq_len + 1), device=device)     # get (batch, seq_len) shape zero matrix
    new_mask.scatter_(-1, sampled_indices, 1)       # set masks in zero matrix as 1
    return new_mask[:, 1:].bool()       # the final mask, True is mask

def data_mask(data,
    mask_prob = 0.15,
    replace_prob = 0.8,
    num_tokens = 22,
    random_token_prob = 0.1,
    mask_token_id = 3,
    pad_token_id = 4,
    mask_ignore_token_ids = [0, 1, 2, 3, 4, 5, 6]
):
    mask_ignore_token_ids = set([*mask_ignore_token_ids, pad_token_id])
    # do not mask [pad] tokens, or any other tokens in the tokens designated to be excluded ([cls], [sep])
    # also do not include these special tokens in the tokens chosen at random
    # print("before mask: ", data.shape, " \n", data)
    no_mask = mask_with_tokens(data, mask_ignore_token_ids)   # ignore_token as True, will not be masked later
    mask = get_mask_subset_with_prob(~no_mask, mask_prob)      # ~no_mask: 为True：可以mask， False：不能mask get the True/False mask matrix
    # print("no mask: \n", no_mask)
    # print("mask: \n", mask)
    # get mask indices
    ## mask_indices = torch.nonzero(mask, as_tuple=True)   # get the index of mask(nonzero value of mask matrix)
    # mask input with mask tokens with probability of `replace_prob` (keep tokens the same with probability 1 - replace_prob)
    masked_input = data.clone().detach()
    
    # if random token probability > 0 for mlm
    if random_token_prob > 0:
        assert num_tokens is not None, 'num_tokens keyword must be supplied when instantiating MLM if using random token replacement'
        random_token_prob = prob_mask_like(mask.float(), random_token_prob)       # get the mask matrix of random token replace
        random_tokens = torch.randint(7, 7+num_tokens, data.shape, device=data.device)     # generate random token matrix with the same shape as input
        random_no_mask = mask_with_tokens(random_tokens, mask_ignore_token_ids)        # not masked matrix for the random token matrix
        random_token_prob &= ~random_no_mask        # get the pure mask matrix of random token replace
        random_indices = torch.nonzero(random_token_prob, as_tuple=True)        # index of random token replace
        masked_input[random_indices] = random_tokens[random_indices]        # replace some tokens by random token

   
    # [mask] input
    replace_prob = prob_mask_like(data, replace_prob)     # get the mask matrix of token being masked
    masked_input = masked_input.masked_fill(mask * replace_prob, mask_token_id)        # get the data has been masked by mask_token
    # mask out any tokens to padding tokens that were not originally going to be masked
    labels = data.masked_fill(~mask, pad_token_id)        # the label of masked tokens

    # Mask positions for loss computation
    # mask_positions = mask & replace_prob
    # random_positions = random_token_prob
    # combined_mask_positions = mask_positions | random_positions
    # print("masked input:\n", masked_input)
    # print("mask_positions:\n ", mask_positions)
    # print("random_positions:\n ", random_token_prob)
    # print(labels)
    return masked_input, labels



# helper functions
def exists(val):
    return val is not None

def coin_flip():
    return random() > 0.5


string_complement_map = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'a': 't', 'c': 'g', 'g': 'c', 't': 'a'}
# augmentation
def string_reverse_complement(seq):
    rev_comp = ''
    for base in seq[::-1]:
        if base in string_complement_map:
            rev_comp += string_complement_map[base]
        # if bp not complement map, use the same bp
        else:
            rev_comp += base
    return rev_comp

def get_dataset(args, 
                tokenizer, 
                file_path):
    if args.line_by_line:
        # return LineByLineTextDatasetChunksCached(
        return LineByLineTextDatasetChunksCached(
            tokenizer=tokenizer,
            file_path=file_path, 
            block_size=args.block_size,
            overwrite_cache=args.overwrite_cache,
            chunk_length=args.chunk_length,
        )
    else:
        return TextDataset(
            tokenizer=tokenizer, file_path=file_path, 
            block_size=args.block_size, 
            overwrite_cache=args.overwrite_cache
        )
    


def run_train(args, is_master, world_size):
    # experiment settings:
    num_epochs = args.epochs  # ~100 seems fine
    max_length = 512  # max len of sequence of dataset (of what you want)
    use_padding = True
    dataset_name = 'human_enhancers_cohn'
    batch_size = args.batch_size
    learning_rate = 6e-4  # good default for Hyena
    rc_aug = True  # reverse complement augmentation
    add_eos = False  # add end of sentence token
    weight_decay = 0.1
    model_name = args.model_name
    ckpt_dir = args.ckpt_dir

    GRADIENT_ACCUMULATION = args.grad_acc
    VALIDATE_EVERY = args.valid_every
    ######################################################################
    #rank = args.nr * args.gpus +u	    
    local_rank = args.local_rank                      
    # dist.init_process_group(                                   
    # 	backend='nccl',                                         
   	# 	init_method='env://',                                   
    # 	world_size=args.world_size,                              
    # 	rank=rank                                               
    # )                                                          
    ######################################################################
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    # create tokenizer
    tokenizer = CharacterTokenizer(
        characters=['D', 'N', 'E', 'K', 'V', 'Y', 'A', 'Q', 'M', 'I', 'T', 
                    'L', 'R', 'F', 'G', 'C', 'S', 'P', 'H', 'W', 'X', 'U'],  # add DNA characters, N is uncertain
        model_max_length=max_length + 2,  # to account for special tokens, like EOS
        add_special_tokens=False,  # we handle special tokens elsewhere
        padding_side='right', # since HyenaDNA is causal, we pad on the left
    )


    ## Set the model
    pretrained_model_name = None
    # we need these for the decoder head, if using
    use_head = False
    n_classes = tokenizer.vocab_size
    use_mlm = True
    # you can override with your own backbone config here if you want,
    # otherwise we'll load the HF one by default
    backbone_cfg = {
        'd_model': 128, 
        'n_layer': 2, 
        'd_inner': 512, 
        'vocab_size': 29, 
        'resid_dropout': 0.0, 
        'embed_dropout': 0.1, 
        'fused_mlp': False, 
        'fused_dropout_add_ln': True, 
        'residual_in_fp32': True, 
        'pad_vocab_size_multiple': 8, 
        'return_hidden_state': True, 
        'layer': {'_name_': 'hyena', 
                  'emb_dim': 5, 
                  'filter_order': 64, 
                  'local_order': 3, 
                  'l_max': 1026, 
                  'modulate': True, 
                  'w': 10, 
                  'lr': 0.0006, 
                  'wd': 0.0, 
                  'lr_pos_emb': 0.0}
    }

    backbone_large_cfg = {
        "d_model": 256,
        "n_layer": 8,
        "d_inner": 1024,
        "vocab_size": 12,
        "resid_dropout": 0.0,
        "embed_dropout": 0.1,
        "fused_mlp": False,
        "fused_dropout_add_ln": True,
        "checkpoint_mixer": True,
        "checkpoint_mlp": True,
        "residual_in_fp32": True,
        "pad_vocab_size_multiple": 8,
        "return_hidden_state": True,
        "layer": {
            "_name_": "hyena",
            "emb_dim": 5,
            "filter_order": 64,
            "local_order": 3,
            "l_max": 1000002,
            "modulate": True,
            "w": 10,
            "lr": 6e-4,
            "wd": 0.0,
            "lr_pos_emb": 0.0
        }
    }

    # instantiate the model (pretrained here)
    if pretrained_model_name in ['hyenadna-tiny-1k-seqlen']:
        # use the pretrained Huggingface wrapper instead
        model = HyenaDNAPreTrainedModel.from_pretrained(
            './checkpoints',
            pretrained_model_name,
            download=True,
            config=backbone_cfg,
            device=device,
            use_head=use_head,
            n_classes=n_classes,
        )
    # from scratch
    else:
        model = HyenaDNAModel(**backbone_large_cfg, use_head=use_head, n_classes=n_classes, use_mlm=use_mlm)
    model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # loss function
    loss_fn = nn.CrossEntropyLoss(ignore_index = PAD_TOKEN_ID, reduction='mean').to(local_rank)
    softmax = nn.Softmax(dim=-1)
    # create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # learning rate scheduler
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=15,
        cycle_mult=2,
        max_lr=learning_rate,
        min_lr=1e-6,
        warmup_steps=5,
        gamma=0.9
    )


    start_time = time.time()
    # create datasets
    train_dataset = get_dataset(args, tokenizer=tokenizer, file_path=args.train_path) if args.do_train else None
    eval_dataset = get_dataset(args, tokenizer=tokenizer, file_path=args.eval_path) if args.do_eval else None

    print(len(train_dataset))
    print(len(eval_dataset))
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"getdataset cost : {elapsed_time} secs")
    ######################################################################
    # train_sampler = torch.utils.data.distributed.DistributedSampler(
    # 	train_dataset,
    # 	num_replicas=args.world_size,
    # 	rank=rank
    # )

    train_sampler = DistributedSampler(train_dataset)
    eval_sampler = SequentialDistributedSampler(eval_dataset, batch_size=batch_size, world_size=world_size)
    # eval_sampler = torch.utils.data.distributed.DistributedSampler(
    # 	eval_dataset,
    # 	num_replicas=args.world_size,
    # 	rank=rank
    # )
    ######################################################################
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               pin_memory=True,
                                               sampler=train_sampler)
    
    eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=0,
                                               sampler=eval_sampler)
    

    
    ######################################################################
    # Wrap the model
    # model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    ######################################################################
    # start = datetime.now()

    best_loss = 1e9
    best_accu = 0.

    dist.barrier()
    for epoch in range(num_epochs):
        train_loader.sampler.set_epoch(epoch)
        model.train()
        dist.barrier()
        running_loss = 0.0
        cum_acc = 0.0
        for index, data in enumerate(train_loader):
            index += 1
            data = data.to(device)
            data, labels = data_mask(data)
            if index % GRADIENT_ACCUMULATION != 0:
                with model.no_sync():
                    logits = model(data)
                    loss = loss_fn(logits.transpose(1, 2), labels) / GRADIENT_ACCUMULATION
                    loss.backward()
            if index % GRADIENT_ACCUMULATION == 0:
                logits = model(data)
                loss = loss_fn(logits.transpose(1, 2), labels) / GRADIENT_ACCUMULATION
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), int(1e2))
                optimizer.step()
                optimizer.zero_grad()
            running_loss += loss.item()
            final = softmax(logits)[..., 1:-1]
            final = final.argmax(dim=-1)
            pred_num = (labels != PAD_TOKEN_ID).sum(dim=-1)
            correct_num = ((labels != PAD_TOKEN_ID) * (final == labels)).sum(dim=-1)
            cum_acc += torch.true_divide(correct_num, pred_num).mean().item()
        epoch_loss = running_loss / index
        epoch_acc = 100 * cum_acc / index
        epoch_loss = get_reduced(epoch_loss, local_rank, 0, world_size)
        epoch_acc = get_reduced(epoch_acc, local_rank, 0, world_size)
        
        if is_master:
            log_message = f'    ==  Epoch: {epoch} | Training Loss: {epoch_loss:.6f} | Accuracy: {epoch_acc:6.4f}%  =='
            print(log_message)
            logging.info(log_message)  # 添加日志记录
            # print(f'    ==  Epoch: {epoch} | Training Loss: {epoch_loss:.6f} | Accuracy: {epoch_acc:6.4f}%  ==')
        dist.barrier()
        scheduler.step()


        if epoch % VALIDATE_EVERY == 0:
            model.eval()
            dist.barrier()
            running_loss = 0.0
            running_error = 0.0
            predictions = []
            truths = []
            with torch.no_grad():
                for index, data in enumerate(eval_loader):
                    index += 1
                    data = data.to(device)
                    data, labels = data_mask(data)
                    logits = model(data)
                    loss = loss_fn(logits.transpose(1, 2), labels)
                    running_loss += loss.item()
                    softmax = nn.Softmax(dim=-1)
                    final = softmax(logits)[..., 1:-1]
                    # final = final.argmax(dim=-1) + 1
                    final = final.argmax(dim=-1)
                    predictions.append(final)
                    truths.append(labels)
                del data, labels, logits, final
                # gather
                predictions = distributed_concat(torch.cat(predictions, dim=0), len(eval_sampler.dataset), world_size)
                truths = distributed_concat(torch.cat(truths, dim=0), len(eval_sampler.dataset), world_size)
                correct_num = ((truths != PAD_TOKEN_ID) * (predictions == truths)).sum(dim=-1)[0].item()
                val_num = (truths != PAD_TOKEN_ID).sum(dim=-1)[0].item()
                val_loss = running_loss / index
                val_loss = get_reduced(val_loss, local_rank, 0, world_size)
                perplexity = math.exp(val_loss)
            if is_master:
                val_acc = 100 * correct_num / val_num
                val_log_message = f'    ==  Epoch: {epoch} | Validation Loss: {val_loss:.6f} | Accuracy: {val_acc:6.4f} | Perplexity: {perplexity:.4f} ==%'
                print(val_log_message)
                logging.info(val_log_message)  # 添加验证日志记录
                # print(f'    ==  Epoch: {epoch} | Validation Loss: {val_loss:.6f} | Accuracy: {val_acc:6.4f}%  ==')
        del predictions, truths

        if is_master:
            if val_loss < best_loss:
                state = "best_val_loss"
                save_ckpt(epoch, model, optimizer, scheduler, epoch_loss, model_name, ckpt_dir, state)
            if val_acc > best_accu:
                state = "best_acc"
                save_ckpt(epoch, model, optimizer, scheduler, epoch_loss, model_name, ckpt_dir, state)


    # if gpu == 0:
    #     print("Training complete in: " + str(datetime.now() - start))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0, help='Local process rank.')
    parser.add_argument("--bin_num", type=int, default=5, help='Number of bins.')
    parser.add_argument("--epochs", type=int, default=100, help='Number of epochs.')
    parser.add_argument("--seed", type=int, default=2021, help='Random seed.')
    parser.add_argument("--batch_size", type=int, default=3, help='Number of batch size.')
    parser.add_argument("--learning_rate", type=float, default=1e-4, help='Learning rate.')
    parser.add_argument("--grad_acc", type=int, default=60, help='Number of gradient accumulation.')
    parser.add_argument("--valid_every", type=int, default=1, help='Number of training epochs between twice validation.')
    parser.add_argument("--mask_prob", type=float, default=0.15, help='Probability of masking.')
    parser.add_argument("--replace_prob", type=float, default=0.9, help='Probability of replacing with [MASK] token for masking.')
    parser.add_argument("--pos_embed", type=bool, default=True, help='Using Gene2vec encoding or not.')
    parser.add_argument("--train_path", type=str, default='/raid_elmo/home/lr/zym/data/protein_data/pretraining/pfam/pfam_holdout.txt', help='Path of data for pretraining.')
    parser.add_argument("--eval_path", type=str, default='/raid_elmo/home/lr/zym/data/protein_data/pretraining/pfam/pfam_holdout.txt', help='Path of data for pretraining.')
    parser.add_argument("--ckpt_dir", type=str, default='./ckpts/', help='Directory of checkpoint to save.')
    parser.add_argument("--model_name", type=str, default='hyenapro_pretrain', help='Pretrained model name.')
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--line_by_line", action="store_true", help="Whether to use line_by_line dataset.")
    parser.add_argument("--overwrite_cache", action="store_true", help="Whether to overwrite cached dataset.")
    parser.add_argument("--chunk_length", type=int, default=1000000, help="Length of chunks when batch tokenizing the dataset.")
    parser.add_argument("--block_size", type=int, default=512, help="Optional input sequence length after tokenization. The training dataset will be truncated in block of this size for training. Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    args = parser.parse_args()
    # local_rank = args.local_rank
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    args.local_rank = local_rank
    rank = int(os.environ["RANK"])
    is_master = rank == 0

    global SEED
    SEED = args.seed
    
    LEARNING_RATE = args.learning_rate
    # SEQ_LEN = args.gene_num + 1
    CLASS = args.bin_num + 2
    MASK_PROB = args.mask_prob
    REPLACE_PROB = args.replace_prob
    POS_EMBED_USING = args.pos_embed

    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    world_size = torch.distributed.get_world_size()
    seed_all(SEED + torch.distributed.get_rank())
    
    run_train(args, is_master, world_size)
    # args.world_size = args.gpus * args.nodes
    # os.environ['MASTER_ADDR'] = '192.168.0.59'
    # os.environ['MASTER_PORT'] = '8888'
    # mp.spawn(run_train, nprocs=args.gpus, args=(args,))


if __name__ == "__main__":
    main()