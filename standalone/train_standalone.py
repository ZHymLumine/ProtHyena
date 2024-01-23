from datetime import datetime
import torch.optim as optim
import torch
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange
from typing import Optional
from functools import partial
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

from genomic_benchmarks.loc2seq import download_dataset
from genomic_benchmarks.data_check import is_downloaded



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
    replace_prob = 0.9,
    num_tokens = None,
    random_token_prob = 0.,
    mask_token_id = 20,
    pad_token_id = 20,
    mask_ignore_token_ids = [0, 1, 2, 3, 4, 5, 6]
):
    mask_ignore_token_ids = set([*mask_ignore_token_ids, pad_token_id])
    # do not mask [pad] tokens, or any other tokens in the tokens designated to be excluded ([cls], [sep])
    # also do not include these special tokens in the tokens chosen at random
    no_mask = mask_with_tokens(data, mask_ignore_token_ids)   # ignore_token as True, will not be masked later
    mask = get_mask_subset_with_prob(~no_mask, mask_prob)      # get the True/False mask matrix
    # get mask indices
    ## mask_indices = torch.nonzero(mask, as_tuple=True)   # get the index of mask(nonzero value of mask matrix)
    # mask input with mask tokens with probability of `replace_prob` (keep tokens the same with probability 1 - replace_prob)
    masked_input = data.clone().detach()
    # if random token probability > 0 for mlm
    if random_token_prob > 0:
        assert num_tokens is not None, 'num_tokens keyword must be supplied when instantiating MLM if using random token replacement'
        random_token_prob = prob_mask_like(data, random_token_prob)       # get the mask matrix of random token replace
        random_tokens = torch.randint(0, num_tokens, data.shape, device=data.device)     # generate random token matrix with the same shape as input
        random_no_mask = mask_with_tokens(random_tokens, mask_ignore_token_ids)        # not masked matrix for the random token matrix
        random_token_prob &= ~random_no_mask        # get the pure mask matrix of random token replace
        random_indices = torch.nonzero(random_token_prob, as_tuple=True)        # index of random token replace
        masked_input[random_indices] = random_tokens[random_indices]        # replace some tokens by random token
    # [mask] input
    replace_prob = prob_mask_like(data, replace_prob)     # get the mask matrix of token being masked
    masked_input = masked_input.masked_fill(mask * replace_prob, mask_token_id)        # get the data has been masked by mask_token
    # mask out any tokens to padding tokens that were not originally going to be masked
    labels = data.masked_fill(~mask, pad_token_id)        # the label of masked tokens
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


class GenomicBenchmarkDataset(torch.utils.data.Dataset):

    '''
    Loop thru bed file, retrieve (chr, start, end), query fasta file for sequence.
    Returns a generator that retrieves the sequence.

    Genomic Benchmarks Dataset, from:
    https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks


    '''

    def __init__(
        self,
        split,
        max_length,
        dataset_name='human_enhancers_cohn',
        d_output=2, # default binary classification
        dest_path="../data", 
        tokenizer=None,
        tokenizer_name=None,
        use_padding=None,
        add_eos=False,
        rc_aug=False,
        return_augs=False,
    ):

        self.max_length = max_length
        self.use_padding = use_padding
        self.tokenizer_name = tokenizer_name
        self.tokenizer = tokenizer
        self.return_augs = return_augs
        self.add_eos = add_eos
        self.d_output = d_output  # needed for decoder to grab
        self.rc_aug = rc_aug

        if not is_downloaded(dataset_name, cache_path=dest_path):
            print("downloading {} to {}".format(dataset_name, dest_path))
            download_dataset(dataset_name, version=0, dest_path=dest_path)
        else:
            print("already downloaded {}-{}".format(split, dataset_name))

        # use Path object
        base_path = Path(dest_path) / dataset_name / split

        self.all_paths = []
        self.all_labels = []
        label_mapper = {}

        for i, x in enumerate(base_path.iterdir()):
            label_mapper[x.stem] = i

        for label_type in label_mapper.keys():
            for x in (base_path / label_type).iterdir():
                self.all_paths.append(x)
                self.all_labels.append(label_mapper[label_type])

    def __len__(self):
        return len(self.all_paths)

    def __getitem__(self, idx):
        txt_path = self.all_paths[idx]
        with open(txt_path, "r") as f:
            content = f.read()
        x = content
        y = self.all_labels[idx]

        # apply rc_aug here if using
        if self.rc_aug and coin_flip():
            x = string_reverse_complement(x)

        seq = self.tokenizer(x,
            add_special_tokens=False,
            padding="max_length" if self.use_padding else None,
            max_length=self.max_length,
            truncation=True,
        )  # add cls and eos token (+2)
        seq = seq["input_ids"]  # get input_ids

        # need to handle eos here
        if self.add_eos:
            # append list seems to be faster than append tensor
            seq.append(self.tokenizer.sep_token_id)

        # convert to tensor
        seq = torch.LongTensor(seq)

        # need to wrap in list
        target = torch.LongTensor([y])

        return seq, target
    

def train(gpu, model, device, train_loader, optimizer, epoch, loss_fn, log_interval=10):
    """Training loop."""
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        data, labels = data_mask(data)
        print(data)
        optimizer.zero_grad()
        
        output = model(data)
        loss = loss_fn(output, target.squeeze())
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0 and gpu == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader, loss_fn):
    """Test loop."""
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target.squeeze()).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

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
    


def run_train(args):
    # experiment settings:
    num_epochs = 100  # ~100 seems fine
    max_length = 512  # max len of sequence of dataset (of what you want)
    use_padding = True
    dataset_name = 'human_enhancers_cohn'
    batch_size = 256
    learning_rate = 6e-4  # good default for Hyena
    rc_aug = True  # reverse complement augmentation
    add_eos = False  # add end of sentence token
    weight_decay = 0.1

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
    
    
    # for fine-tuning, only the 'tiny' model can fit on colab
    # pretrained_model_name = 'hyenadna-tiny-1k-seqlen'  # use None if training from scratch
    pretrained_model_name = None
    # we need these for the decoder head, if using
    use_head = True
    n_classes = 2

    # you can override with your own backbone config here if you want,
    # otherwise we'll load the HF one by default
    backbone_cfg = {
        'd_model': 128, 
        'n_layer': 2, 
        'd_inner': 512, 
        'vocab_size': 12, 
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


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

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
        model = HyenaDNAModel(**backbone_cfg, use_head=use_head, n_classes=n_classes)

    # create tokenizer
    tokenizer = CharacterTokenizer(
        characters=['D', 'N', 'E', 'K', 'V', 'Y', 'A', 'Q', 'M', 'I', 'T', 
                    'L', 'R', 'F', 'G', 'C', 'S', 'P', 'H', 'W', 'X', 'U'],  # add DNA characters, N is uncertain
        model_max_length=max_length + 2,  # to account for special tokens, like EOS
        add_special_tokens=False,  # we handle special tokens elsewhere
        padding_side='right', # since HyenaDNA is causal, we pad on the left
    )

    # create datasets
    train_dataset = get_dataset(args, tokenizer=tokenizer, file_path=args.train_path) if args.do_train else None
    # eval_dataset = get_dataset(args, tokenizer=tokenizer, file_path=args.eval_path) if args.do_eval else None

    print(len(train_dataset))
    ######################################################################
    train_sampler = torch.utils.data.distributed.DistributedSampler(
    	train_dataset,
    	num_replicas=args.world_size,
    	rank=rank
    )

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
                                               num_workers=0,
                                               pin_memory=True,
                                               sampler=train_sampler)
    
    # eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset,
    #                                            batch_size=batch_size,
    #                                            shuffle=False,
    #                                            num_workers=0,
    #                                            sampler=eval_sampler)


    # loss function
    loss_fn = nn.CrossEntropyLoss()

    # create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    torch.cuda.set_device(gpu)
    model.cuda(gpu)

    ######################################################################
    # Wrap the model
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    ######################################################################
    # start = datetime.now()

    for epoch in range(num_epochs):
        train(gpu, model, device, train_loader, optimizer, epoch, loss_fn)
        # test(model, device, eval_loader, loss_fn)
        optimizer.step()

    # if gpu == 0:
    #     print("Training complete in: " + str(datetime.now() - start))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1, help='Local process rank.')
    parser.add_argument("--bin_num", type=int, default=5, help='Number of bins.')
    parser.add_argument("--epoch", type=int, default=100, help='Number of epochs.')
    parser.add_argument("--seed", type=int, default=2021, help='Random seed.')
    parser.add_argument("--batch_size", type=int, default=3, help='Number of batch size.')
    parser.add_argument("--learning_rate", type=float, default=1e-4, help='Learning rate.')
    parser.add_argument("--grad_acc", type=int, default=60, help='Number of gradient accumulation.')
    parser.add_argument("--valid_every", type=int, default=1, help='Number of training epochs between twice validation.')
    parser.add_argument("--mask_prob", type=float, default=0.15, help='Probability of masking.')
    parser.add_argument("--replace_prob", type=float, default=0.9, help='Probability of replacing with [MASK] token for masking.')
    parser.add_argument("--pos_embed", type=bool, default=True, help='Using Gene2vec encoding or not.')
    parser.add_argument("--train_path", type=str, default='/home/lr/zym/research/data/protein_data/pretraining/pfam/pfam_holdout.txt', help='Path of data for pretraining.')
    parser.add_argument("--eval_path", type=str, default='/home/lr/zym/research/data/protein_data/pretraining/pfam/pfam_valid.txt', help='Path of data for pretraining.')
    parser.add_argument("--ckpt_dir", type=str, default='./ckpts/', help='Directory of checkpoint to save.')
    parser.add_argument("--model_name", type=str, default='panglao_pretrain', help='Pretrained model name.')
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
    local_rank = args.local_rank
    # rank = int(os.environ["RANK"])
    rank  = 0
    is_master = rank == 0

    global SEED
    SEED = args.seed
    global EPOCHS
    EPOCHS = args.epoch
    BATCH_SIZE = args.batch_size
    global GRADIENT_ACCUMULATION
    GRADIENT_ACCUMULATION = args.grad_acc
    LEARNING_RATE = args.learning_rate
    # SEQ_LEN = args.gene_num + 1
    VALIDATE_EVERY = args.valid_every
    CLASS = args.bin_num + 2
    MASK_PROB = args.mask_prob
    REPLACE_PROB = args.replace_prob
    RANDOM_TOKEN_PROB = 0.
    MASK_TOKEN_ID = CLASS - 1
    PAD_TOKEN_ID = CLASS - 1
    MASK_IGNORE_TOKEN_IDS = [0]
    POS_EMBED_USING = args.pos_embed

    model_name = args.model_name
    ckpt_dir = args.ckpt_dir

    local_rank = args.local_rank
    rank = int(os.environ["RANK"])
    is_master = rank == 0

    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    world_size = torch.distributed.get_world_size()
    seed_all(SEED + torch.distributed.get_rank())
    
    run_train(args)
    # args.world_size = args.gpus * args.nodes
    # os.environ['MASTER_ADDR'] = '192.168.0.59'
    # os.environ['MASTER_PORT'] = '8888'
    # mp.spawn(run_train, nprocs=args.gpus, args=(args,))


if __name__ == "__main__":
    main()