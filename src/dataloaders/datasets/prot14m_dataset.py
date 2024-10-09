from pathlib import Path
from pyfaidx import Fasta
import polars as pl
import pandas as pd
import torch
from random import randrange, random
import numpy as np
import random
import os
import time
import math
from functools import reduce

# helper functions

def exists(val):
    return val is not None

def coin_flip():
    return random() > 0.5

# augmentations

string_complement_map = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'a': 't', 'c': 'g', 'g': 'c', 't': 'a'}

def string_reverse_complement(seq):
    rev_comp = ''
    for base in seq[::-1]:
        if base in string_complement_map:
            rev_comp += string_complement_map[base]
        # if bp not complement map, use the same bp
        else:
            rev_comp += base
    return rev_comp



MASK_PROB = 0.15
REPLACE_PROB = 0.8
RANDOM_TOKEN_PROB = 0.1
PAD_TOKEN_ID = 4
MASK_TOKEN_ID = 3
MASK_IGNORE_TOKEN_IDS = [0, 1, 2, 3, 4, 5, 6]

def prob_mask_like(t, prob):
    return torch.zeros_like(t).float().uniform_(0, 1) < prob

# get the mask matrix which cannot be masked
def mask_with_tokens(t, token_ids):
    init_no_mask = torch.full_like(t, False, dtype=torch.bool)
    mask = reduce(lambda acc, el: acc | (t == el), token_ids, init_no_mask)
    return mask

def get_mask_subset_with_prob(mask, prob):
    batch, seq_len = mask.shape
    device = mask.device
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

def get_mask_subset_with_prob(mask, prob):
    # print(mask.shape)
    seq_len = mask.size()[0]
    print(seq_len)
    device = mask.device
    max_masked = math.ceil(prob * seq_len)      # num of mask of a single sequence in average
    num_tokens = mask.sum(dim=-1, keepdim=True)     # num of pure tokens of each sequence except special tokens
    mask_excess = torch.cat((torch.zeros(0), torch.arange(mask.size(-1)).repeat(mask.size(0)))).reshape(mask.size(0),mask.size(-1)).to(device)
    mask_excess = (mask_excess >= (num_tokens * prob).ceil())        # only 15% of pure tokens can be masked
    mask_excess = mask_excess[:, :max_masked]       # get difference between 15% of pure tokens and 15% of all tokens
    rand = torch.rand(seq_len, device=device).masked_fill(~mask, -1e9)     # rand (0-1) as prob, special token use -1e9
    _, sampled_indices = rand.topk(max_masked, dim=-1)      # get index of topk prob to mask
    sampled_indices = (sampled_indices + 1).masked_fill_(mask_excess, 0)        # delete difference of mask not pure
    new_mask = torch.zeros(seq_len + 1, device=device)     # get (batch, seq_len) shape zero matrix
    new_mask.scatter_(-1, sampled_indices, 1)       # set masks in zero matrix as 1
    return new_mask[:, 1:].bool()       # the final mask, True is mask


def data_mask(data,
    mask_prob = MASK_PROB,
    replace_prob = REPLACE_PROB,
    num_tokens = None,
    random_token_prob = RANDOM_TOKEN_PROB,
    mask_token_id = MASK_TOKEN_ID,
    pad_token_id = PAD_TOKEN_ID,
    mask_ignore_token_ids = [0]
):
    mask_ignore_token_ids = set([*mask_ignore_token_ids, pad_token_id])
    # do not mask [pad] tokens, or any other tokens in the tokens designated to be excluded ([cls], [sep])
    # also do not include these special tokens in the tokens chosen at random
    no_mask = mask_with_tokens(data, mask_ignore_token_ids)   # ignore_token as True, will not be masked later
    print(no_mask.shape)
    print(no_mask)
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


import torch
import random

def mask_tokens(inputs, tokenizer, mlm_probability=0.15):
    """
    准备用于BERT的MLM任务的输入和标签。
    Args:
        inputs (torch.Tensor): 输入序列的token IDs。
        tokenizer: 使用的tokenizer实例。
        mlm_probability (float): 选为mask的token的概率。
    Returns:
        torch.Tensor: 用于训练的seq_ids。
        torch.Tensor: 对应的labels。
    """
    labels = inputs.clone()
    data = inputs.clone()
    # 生成mask数组
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = tokenizer.get_special_tokens_mask(labels, already_has_special_tokens=True)
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    # print(f"special_tokens_mask : {special_tokens_mask}")
    # print(f"masked_indices: {masked_indices}")
    labels[~masked_indices] = -100  # 只有被遮罩的token才会被预测

    # 80% 的时间将masked token替换为 [MASK]

    # print(f"before inputs: {inputs}")
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
   
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% 的时间随机替换为其他token
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced

    # 获取词汇表中非特殊token的范围
    vocab_size = len(tokenizer)
    special_tokens_ids = tokenizer.all_special_ids
    special_tokens_ids.append(5)
    # print(f"special: {special_tokens_ids}")
    # 创建一个mask，排除所有特殊token
    non_special_tokens_mask = torch.ones(vocab_size, dtype=torch.bool)
    for special_id in special_tokens_ids:
        non_special_tokens_mask[special_id] = False

    # 在非特殊token中随机选择
    random_tokens_choices = torch.arange(vocab_size)[non_special_tokens_mask]
    random_tokens = random_tokens_choices[torch.randint(len(random_tokens_choices), labels.shape, dtype=torch.long)]

    # 应用随机替换
    inputs[indices_random] = random_tokens[indices_random]


    # indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    # random_tokens = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    # print(random_tokens)
    # inputs[indices_random] = random_tokens[indices_random]

    # print(f"after inputs: {inputs}")
    # 剩下的10%不变
    return inputs, labels



class Prot14MDatasetMLM(torch.utils.data.Dataset):
    def __init__(
        self,
        split: str,
        fasta_file,
        max_length,
        pad_max_length=None,
        tokenizer=None,
        tokenizer_name=None,
        add_eos=False,
        return_seq_indices=False,
        shift_augs=None,
        rc_aug=False,
        return_augs=False,
        replace_N_token=False,  # replace N token with pad token
        pad_interval = False,  # options for different padding
        split_ratio=(0.8, 0.1, 0.1),
    ):
        
        self.max_length = max_length
        self.pad_max_length = pad_max_length if pad_max_length is not None else max_length
        self.tokenizer_name = tokenizer_name
        self.tokenizer = tokenizer
        self.return_augs = return_augs
        self.add_eos = add_eos
        self.replace_N_token = replace_N_token  
        self.pad_interval = pad_interval         
        self.split = split

        self.return_seq_indices = return_seq_indices
        # self.max_length = max_length # -1 for adding sos or eos token
        self.shift_augs = shift_augs
        self.rc_aug = rc_aug
        self.pad_interval = pad_interval 
        
        # fasta_file = "/raid_elmo/home/lr/zym/data/rna_data/rnacentral_active.fasta"
        fasta_file = Path(fasta_file)
        self.fasta_file = fasta_file
        
        assert sum(split_ratio) == 1, "Split ratios must sum up to 1"
        assert fasta_file.exists(), 'path to fasta file must exist'

        
        start_time = time.time()
        fasta_path_str = str(self.fasta_file)  # 将Path对象转换为字符串
        # 检查.fai索引文件是否存在
        if not os.path.exists(fasta_path_str + '.fai'):
            # 如果不存在，使用pyfaidx读取fasta文件并创建索引
            print(".fai not exists")
            self.seqs = Fasta(fasta_path_str)
        else:
            # 如果索引文件存在，直接使用它
            print(".fai exists")
            self.seqs = Fasta(fasta_path_str, build_index=False)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"读取fasta文件运行时间:{elapsed_time}秒")

        # save index of the sequence
        self.keys = list(self.seqs.keys())

        num_sequences = len(self.keys)
        train_end = int(num_sequences * split_ratio[0])
        valid_end = train_end + int(num_sequences * split_ratio[1])

        if split == "train":
            self.split_keys = self.keys[:train_end]
        elif split == "valid":
            self.split_keys = self.keys[train_end:valid_end]
        elif split == "test":
            self.split_keys = self.keys[valid_end:]
        else:
            raise ValueError("Invalid split. Expected one of: 'train', 'valid', 'test'")


    def __len__(self):
        return len(self.split_keys)
    
    def _getseq(self, seq):
        rna_length = len(seq)
        start, end = 0, rna_length - 1
        left_padding = right_padding = 0

        if exists(self.shift_augs):
            min_shift, max_shift = self.shift_augs
            max_shift += 1

            min_shift = max(start + min_shift, 0) - start
            max_shift = min(end + max_shift, rna_length) - end

            rand_shift = randrange(min_shift, max_shift)
            start += rand_shift
            end += rand_shift

        if rna_length < self.max_length:
            extra_seq = self.max_length - rna_length

            extra_left_seq = extra_seq // 2
            extra_right_seq = extra_seq - extra_left_seq

            start -= extra_left_seq
            end += extra_right_seq

        if start < 0:
            left_padding = -start
            start = 0

        if end > rna_length:
            right_padding = end - rna_length
            end = rna_length
        
        if rna_length > self.max_length:
            end = start + self.max_length

        seq = str(seq[start:end])

        if self.rc_aug and coin_flip():
            seq = string_reverse_complement(seq)

        if self.pad_interval:
            seq = ('.' * left_padding) + seq + ('.' * right_padding)

        return seq
    

    def __getitem__(self, idx):
        # 随机选择一个序列键值
        key = self.split_keys[idx % len(self.split_keys)]
        # seq = str(self.seqs[key][:].seq).replace('T', 'U')
        # seq = str(self.seqs[key][:].seq)
        # print(f"key={key}")
        seq = str(self.seqs[key][:].seq)
        # print(f"seq = {seq}")
        # preprocess sequence 
        # seq = self._getseq(seq)
        # print(seq)
           
        if self.tokenizer_name == 'char':
            seq = self.tokenizer(seq,
                add_special_tokens=True if self.add_eos else False,  # this is what controls adding eos
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
            )
            seq = seq["input_ids"]  # get input_ids
    
        elif self.tokenizer_name == 'bpe':
            seq = self.tokenizer(seq, 
                # add_special_tokens=False,
                padding="max_length",
                max_length=self.pad_max_length,
                truncation=True,
            ) 
            # get input_ids
            if self.add_eos:
                seq = seq["input_ids"][1:]  # remove the bos, keep the eos token
            else:
                seq = seq["input_ids"][1:-1]  # remove both special tokens
        
        # convert to tensor
        seq = torch.LongTensor(seq)  # hack, remove the initial cls tokens for now


        # print(f"seq: {seq}")
        # print("data: ", data.shape)
        # data, labels = data_mask(data)
        data = seq.clone()  
        data, labels = mask_tokens(data, self.tokenizer)

        
        # print(f"data: {data}")
        # print(f"seq : {seq}")
        # print(f"labels : {labels}")
        # target = seq[1:].clone()  # offset by 1, includes eos

        return data, labels
    





class Prot14MDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split: str,
        fasta_file,
        max_length,
        pad_max_length=None,
        tokenizer=None,
        tokenizer_name=None,
        add_eos=False,
        return_seq_indices=False,
        shift_augs=None,
        rc_aug=False,
        return_augs=False,
        replace_N_token=False,  # replace N token with pad token
        pad_interval = False,  # options for different padding
        split_ratio=(0.8, 0.1, 0.1),
    ):
        
        self.max_length = max_length
        self.pad_max_length = pad_max_length if pad_max_length is not None else max_length
        self.tokenizer_name = tokenizer_name
        self.tokenizer = tokenizer
        self.return_augs = return_augs
        self.add_eos = add_eos
        self.replace_N_token = replace_N_token  
        self.pad_interval = pad_interval         
        self.split = split

        self.return_seq_indices = return_seq_indices
        # self.max_length = max_length # -1 for adding sos or eos token
        self.shift_augs = shift_augs
        self.rc_aug = rc_aug
        self.pad_interval = pad_interval 
        
        # fasta_file = "/raid_elmo/home/lr/zym/data/rna_data/rnacentral_active.fasta"
        fasta_file = Path(fasta_file)
        self.fasta_file = fasta_file
        
        assert sum(split_ratio) == 1, "Split ratios must sum up to 1"
        assert fasta_file.exists(), 'path to fasta file must exist'

        
        start_time = time.time()
        fasta_path_str = str(self.fasta_file)  # 将Path对象转换为字符串
        # 检查.fai索引文件是否存在
        if not os.path.exists(fasta_path_str + '.fai'):
            # 如果不存在，使用pyfaidx读取fasta文件并创建索引
            print(".fai not exists")
            self.seqs = Fasta(fasta_path_str)
        else:
            # 如果索引文件存在，直接使用它
            print(".fai exists")
            self.seqs = Fasta(fasta_path_str, build_index=False)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"读取fasta文件运行时间:{elapsed_time}秒")

        # save index of the sequence
        self.keys = list(self.seqs.keys())

        num_sequences = len(self.keys)
        train_end = int(num_sequences * split_ratio[0])
        valid_end = train_end + int(num_sequences * split_ratio[1])

        if split == "train":
            self.split_keys = self.keys[:train_end]
        elif split == "valid":
            self.split_keys = self.keys[train_end:valid_end]
        elif split == "test":
            self.split_keys = self.keys[valid_end:]
        else:
            raise ValueError("Invalid split. Expected one of: 'train', 'valid', 'test'")


    def __len__(self):
        return len(self.split_keys)
    
    def _getseq(self, seq):
        rna_length = len(seq)
        start, end = 0, rna_length - 1
        left_padding = right_padding = 0

        if exists(self.shift_augs):
            min_shift, max_shift = self.shift_augs
            max_shift += 1

            min_shift = max(start + min_shift, 0) - start
            max_shift = min(end + max_shift, rna_length) - end

            rand_shift = randrange(min_shift, max_shift)
            start += rand_shift
            end += rand_shift

        if rna_length < self.max_length:
            extra_seq = self.max_length - rna_length

            extra_left_seq = extra_seq // 2
            extra_right_seq = extra_seq - extra_left_seq

            start -= extra_left_seq
            end += extra_right_seq

        if start < 0:
            left_padding = -start
            start = 0

        if end > rna_length:
            right_padding = end - rna_length
            end = rna_length
        
        if rna_length > self.max_length:
            end = start + self.max_length

        seq = str(seq[start:end])

        if self.rc_aug and coin_flip():
            seq = string_reverse_complement(seq)

        if self.pad_interval:
            seq = ('.' * left_padding) + seq + ('.' * right_padding)

        return seq
    

    def __getitem__(self, idx):
        # 随机选择一个序列键值
        key = self.split_keys[idx % len(self.split_keys)]
        # seq = str(self.seqs[key][:].seq).replace('T', 'U')
        # seq = str(self.seqs[key][:].seq)
        # print(f"key={key}")
        seq = str(self.seqs[key][:].seq)
        # print(f"seq = {seq}")
        # preprocess sequence 
        # seq = self._getseq(seq)
        # print(seq)
        if self.tokenizer_name == 'char':
            seq = self.tokenizer(seq,
                add_special_tokens=True if self.add_eos else False,  # this is what controls adding eos
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
            )
            seq = seq["input_ids"]  # get input_ids

        elif self.tokenizer_name == 'bpe':
            seq = self.tokenizer(seq, 
                # add_special_tokens=False, 
                padding="max_length",
                max_length=self.pad_max_length,
                truncation=True,
            ) 
            # get input_ids
            if self.add_eos:
                seq = seq["input_ids"][1:]  # remove the bos, keep the eos token
            else:
                seq = seq["input_ids"][1:-1]  # remove both special tokens
        # convert to tensor
        seq = torch.LongTensor(seq)  # hack, remove the initial cls tokens for now

        if self.replace_N_token:
            # replace N token with a pad token, so we can ignore it in the loss
            seq = self.replace_value(seq, self.tokenizer._vocab_str_to_int['N'], self.tokenizer.pad_token_id)

        data = seq[:-1].clone()  # remove eos
        
        target = seq[1:].clone()  # offset by 1, includes eos

        return data, target
    