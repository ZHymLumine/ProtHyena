import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import os
from itertools import islice
from functools import partial
import os
import functools
# import json
# from pathlib import Path
# from pyfaidx import Fasta
# import polars as pl
# import pandas as pd
import torch
from random import randrange, random
import numpy as np
from pathlib import Path

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union
from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer


class CharacterTokenizer(PreTrainedTokenizer):
    def __init__(self, characters: Sequence[str], model_max_length: int, padding_side: str='left', **kwargs):
        """Character tokenizer for Hugging Face transformers.
        Args:
            characters (Sequence[str]): List of desired characters. Any character which
                is not included in this list will be replaced by a special token called
                [UNK] with id=6. Following are list of all of the special tokens with
                their corresponding ids:
                    "[CLS]": 0
                    "[SEP]": 1
                    "[BOS]": 2
                    "[MASK]": 3
                    "[PAD]": 4
                    "[RESERVED]": 5
                    "[UNK]": 6
                an id (starting at 7) will be assigned to each character.
            model_max_length (int): Model maximum sequence length.
        """
        self.characters = characters
        self.model_max_length = model_max_length
        bos_token = AddedToken("[BOS]", lstrip=False, rstrip=False)
        eos_token = AddedToken("[SEP]", lstrip=False, rstrip=False)
        sep_token = AddedToken("[SEP]", lstrip=False, rstrip=False)
        cls_token = AddedToken("[CLS]", lstrip=False, rstrip=False)
        pad_token = AddedToken("[PAD]", lstrip=False, rstrip=False)
        unk_token = AddedToken("[UNK]", lstrip=False, rstrip=False)

        mask_token = AddedToken("[MASK]", lstrip=True, rstrip=False)
        super().__init__(
            bos_token=bos_token,
            eos_token=sep_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            add_prefix_space=False,
            model_max_length=model_max_length,
            padding_side=padding_side,
            **kwargs,
        )

        self._vocab_str_to_int = {
            "[CLS]": 0,
            "[SEP]": 1,
            "[BOS]": 2,
            "[MASK]": 3,
            "[PAD]": 4,
            "[RESERVED]": 5,
            "[UNK]": 6,
            **{ch: i + 7 for i, ch in enumerate(characters)},
        }
        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def _tokenize(self, text: str) -> List[str]:
        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int["[UNK]"])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        sep = [self.sep_token_id]
        # cls = [self.cls_token_id]
        result = token_ids_0 + sep
        if token_ids_1 is not None:
            result += token_ids_1 + sep
        return result

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )

        result = ([0] * len(token_ids_0)) + [1]
        if token_ids_1 is not None:
            result += ([0] * len(token_ids_1)) + [1]
        return result

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        result = len(cls + token_ids_0 + sep) * [0]
        if token_ids_1 is not None:
            result += len(token_ids_1 + sep) * [1]
        return result

    def get_config(self) -> Dict:
        return {
            "char_ords": [ord(ch) for ch in self.characters],
            "model_max_length": self.model_max_length,
        }

    @classmethod
    def from_config(cls, config: Dict) -> "CharacterTokenizer":
        cfg = {}
        cfg["characters"] = [chr(i) for i in config["char_ords"]]
        cfg["model_max_length"] = config["model_max_length"]
        return cls(**cfg)

    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
        cfg_file = Path(save_directory) / "tokenizer_config.json"
        cfg = self.get_config()
        with open(cfg_file, "w") as f:
            json.dump(cfg, f, indent=4)

    @classmethod
    def from_pretrained(cls, save_directory: Union[str, os.PathLike], **kwargs):
        cfg_file = Path(save_directory) / "tokenizer_config.json"
        with open(cfg_file) as f:
            cfg = json.load(f)
        return cls.from_config(cfg)

class ProteinLocationDataset(Dataset):
    def __init__(
        self,
        split,
        max_length,
        dataset_name="location",
        d_output=10, # default binary classification
        dest_path=None,
        tokenizer=None,
        tokenizer_name=None,
        use_padding=True,
        add_eos=False,
        rc_aug=False,
        return_augs=False,
        return_mask=False,
    ):
        
        self.split = split
        self.max_length = max_length
        self.use_padding = use_padding
        self.tokenizer_name = tokenizer_name
        self.tokenizer = tokenizer
        self.return_augs = return_augs
        self.add_eos = add_eos
        self.d_output = d_output  # needed for decoder to grab
        self.rc_aug = rc_aug
        self.return_mask = return_mask

        # base_path = Path(dest_path)  / split
        fasta_file = os.path.join(dest_path, f"{split}.fa")
        self.sequences, self.labels = self._load_fasta(fasta_file)
        self.label_to_id = {'Nucleus': 0, 'Cytoplasm':1, 'Extracellular': 2, 'Mitochondrion':3, 
                            'Cell.membrane': 4, 'Endoplasmic.reticulum': 5, 'Plastid': 6, 
                            'Golgi.apparatus': 7, 'Lysosome/Vacuole': 8, 'Peroxisome': 9}
        
            
    def _load_fasta(self, file_path):
        sequences = []
        labels = []
        with open(file_path, 'r') as file:
            for line in file:
                if line.startswith('>'):
                    label = line.split()[1].split('-')[0]  # Split and get the label part
                    labels.append(label)
                else:
                    sequences.append(line.strip())  # Remove newline characters
        return sequences, labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]

        seq = self.tokenizer(sequence,
            add_special_tokens=True if self.add_eos else False,  # this is what controls adding eos
            padding="max_length" if self.use_padding else "do_not_pad",
            max_length=self.max_length,
            truncation=True,
        )
        seq_ids = seq["input_ids"]  # get input_ids

        seq_ids = torch.LongTensor(seq_ids)

        target = torch.LongTensor([self.label_to_id[label]])  # offset by 1, includes eos
       
        if self.return_mask:
            return seq_ids, target, {'mask': torch.BoolTensor(seq['attention_mask'])}
        else:
            return seq_ids, target



class HomologyDataset(Dataset):
    def __init__(
        self,
        split,
        max_length,
        dataset_name="homology",
        d_output=1195, # default binary classification
        dest_path=None,
        tokenizer=None,
        tokenizer_name=None,
        use_padding=True,
        add_eos=False,
        rc_aug=False,
        return_augs=False,
        return_mask=False,
    ):
        
        self.split = split
        self.max_length = max_length
        self.use_padding = use_padding
        self.tokenizer_name = tokenizer_name
        self.tokenizer = tokenizer
        self.return_augs = return_augs
        self.add_eos = add_eos
        self.d_output = d_output  # needed for decoder to grab
        self.rc_aug = rc_aug
        self.return_mask = return_mask

        base_path = Path(dest_path)  / split
        fasta_file = os.path.join(dest_path, f"{split}.fasta")
        self.sequences, self.labels = self._load_fasta(fasta_file)
        
        # csv_file = os.path.join(dest_path, f"{split}.csv")
        # self.data = pd.read_csv(csv_file)

            
    def _load_fasta(self, file_path):
        sequences = []
        labels = []
        with open(file_path, 'r') as file:
            for line in file:
                if line.startswith('>'):
                    label = line.split()[0][1:]  # Split and get the label part
                    labels.append(label)
                else:
                    sequences.append(line.strip())  # Remove newline characters
        
        return sequences, labels

    def __len__(self):
        # return len(self.data)
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        # print(idx)
        # print("len: ", len(self.labels))
        label = int(self.labels[idx])
        # sequence = self.data.iloc[idx, 0]
        # label = int(self.data.iloc[idx, 1])


        seq = self.tokenizer(sequence,
            add_special_tokens=True if self.add_eos else False,  # this is what controls adding eos
            padding="max_length" if self.use_padding else "do_not_pad",
            max_length=self.max_length,
            truncation=True,
        )
        seq_ids = seq["input_ids"]  # get input_ids

        seq_ids = torch.LongTensor(seq_ids)

        target = torch.LongTensor([label])  # offset by 1, includes eos
       
        if self.return_mask:
            return seq_ids, target, {'mask': torch.BoolTensor(seq['attention_mask'])}
        else:
            return seq_ids, target


class FoldClassDataset(Dataset):
    def __init__(
        self,
        split,
        max_length,
        dataset_name="fold_class",
        d_output=7, # default binary classification
        dest_path=None,
        tokenizer=None,
        tokenizer_name=None,
        use_padding=True,
        add_eos=False,
        rc_aug=False,
        return_augs=False,
        return_mask=False,
    ):
        
        self.split = split
        self.max_length = max_length
        self.use_padding = use_padding
        self.tokenizer_name = tokenizer_name
        self.tokenizer = tokenizer
        self.return_augs = return_augs
        self.add_eos = add_eos
        self.d_output = d_output  # needed for decoder to grab
        self.rc_aug = rc_aug
        self.return_mask = return_mask

        # base_path = Path(dest_path)  / split
        # fasta_file = os.path.join(dest_path, f"{split}.fasta")
        # self.sequences, self.labels = self._load_fasta(fasta_file)
        
        csv_file = os.path.join(dest_path, f"{split}.csv")
        self.data = pd.read_csv(csv_file)
        self.label_to_id = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e':4, 'f':5, 'g':6}
            
    def _load_fasta(self, file_path):
        sequences = []
        labels = []
        with open(file_path, 'r') as file:
            for line in file:
                if line.startswith('>'):
                    label = line.split()[0][1:]  # Split and get the label part
                    labels.append(label)
                else:
                    sequences.append(line.strip())  # Remove newline characters
        
        return sequences, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # sequence = self.sequences[idx]
        # print(idx)
        # print("len: ", len(self.labels))
        # label = int(self.labels[idx])
        sequence = self.data.iloc[idx, 0]
        label = int(self.label_to_id[self.data.iloc[idx, 1]])


        seq = self.tokenizer(sequence,
            add_special_tokens=True if self.add_eos else False,  # this is what controls adding eos
            padding="max_length" if self.use_padding else "do_not_pad",
            max_length=self.max_length,
            truncation=True,
        )
        seq_ids = seq["input_ids"]  # get input_ids

        seq_ids = torch.LongTensor(seq_ids)

        target = torch.LongTensor([label])  # offset by 1, includes eos
       
        if self.return_mask:
            return seq_ids, target, {'mask': torch.BoolTensor(seq['attention_mask'])}
        else:
            return seq_ids, target


class SignalPeptideDataset(Dataset):
    def __init__(
        self,
        split,
        max_length,
        dataset_name="signalP",
        d_output=2, # default binary classification
        dest_path=None,
        tokenizer=None,
        tokenizer_name=None,
        use_padding=True,
        add_eos=False,
        rc_aug=False,
        return_augs=False,
        return_mask=False,
    ):
        
        self.split = split
        self.max_length = max_length
        self.use_padding = use_padding
        self.tokenizer_name = tokenizer_name
        self.tokenizer = tokenizer
        self.return_augs = return_augs
        self.add_eos = add_eos
        self.d_output = d_output  # needed for decoder to grab
        self.rc_aug = rc_aug
        self.return_mask = return_mask

        # base_path = Path(dest_path)  / split
        csv_file = os.path.join(dest_path, f"{split}.csv")
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data.iloc[idx, 1]  
        label = int(self.data.iloc[idx, 0])

        seq = self.tokenizer(sequence,
            add_special_tokens=True if self.add_eos else False,  # this is what controls adding eos
            padding="max_length" if self.use_padding else "do_not_pad",
            max_length=self.max_length,
            truncation=True,
        )
        seq_ids = seq["input_ids"]  # get input_ids

        seq_ids = torch.LongTensor(seq_ids)

        target = torch.LongTensor([label])  # offset by 1, includes eos
       
        if self.return_mask:
            return seq_ids, target, {'mask': torch.BoolTensor(seq['attention_mask'])}
        else:
            return seq_ids, target
        

class CleavageDataset(Dataset):
    def __init__(
        self,
        split,
        max_length,
        dataset_name="cleavage",
        d_output=2, # default binary classification
        dest_path=None,
        tokenizer=None,
        tokenizer_name=None,
        use_padding=True,
        add_eos=False,
        rc_aug=False,
        return_augs=False,
        return_mask=False,
    ):
        
        self.split = split
        self.max_length = max_length
        self.use_padding = use_padding
        self.tokenizer_name = tokenizer_name
        self.tokenizer = tokenizer
        self.return_augs = return_augs
        self.add_eos = add_eos
        self.d_output = d_output  # needed for decoder to grab
        self.rc_aug = rc_aug
        self.return_mask = return_mask

        # base_path = Path(dest_path)  / split
        csv_file = os.path.join(dest_path, f"{split}.csv")
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data.iloc[idx, 0]  
        label = int(self.data.iloc[idx, 1])

        seq = self.tokenizer(sequence,
            add_special_tokens=True if self.add_eos else False,  # this is what controls adding eos
            padding="max_length" if self.use_padding else "do_not_pad",
            max_length=self.max_length,
            truncation=True,
        )
        seq_ids = seq["input_ids"]  # get input_ids

        seq_ids = torch.LongTensor(seq_ids)

        target = torch.LongTensor([label])  # offset by 1, includes eos
       
        if self.return_mask:
            return seq_ids, target, {'mask': torch.BoolTensor(seq['attention_mask'])}
        else:
            return seq_ids, target
        

class SolubilityDataset(Dataset):
    def __init__(
        self,
        split,
        max_length,
        dataset_name="solubility",
        d_output=2, # default binary classification
        dest_path=None,
        tokenizer=None,
        tokenizer_name=None,
        use_padding=True,
        add_eos=False,
        rc_aug=False,
        return_augs=False,
        return_mask=False,
    ):
        
        self.split = split
        self.max_length = max_length
        self.use_padding = use_padding
        self.tokenizer_name = tokenizer_name
        self.tokenizer = tokenizer
        self.return_augs = return_augs
        self.add_eos = add_eos
        self.d_output = d_output  # needed for decoder to grab
        self.rc_aug = rc_aug
        self.return_mask = return_mask

        # base_path = Path(dest_path)  / split
        fasta_file = os.path.join(dest_path, f"{split}.fa")
        self.sequences, self.labels = self._load_fasta(fasta_file)
       
            
    def _load_fasta(self, file_path):
        sequences = []
        labels = []
        with open(file_path, 'r') as file:
            for line in file:
                if line.startswith('>'):
                    label = line.split()[1]  # Split and get the label part
                    labels.append(label)
                else:
                    sequences.append(line.strip())  # Remove newline characters
        return sequences, labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = int(self.labels[idx])

        seq = self.tokenizer(sequence,
            add_special_tokens=True if self.add_eos else False,  # this is what controls adding eos
            padding="max_length" if self.use_padding else "do_not_pad",
            max_length=self.max_length,
            truncation=True,
        )
        seq_ids = seq["input_ids"]  # get input_ids

        seq_ids = torch.LongTensor(seq_ids)

        target = torch.LongTensor([label])  # offset by 1, includes eos
       
        if self.return_mask:
            return seq_ids, target, {'mask': torch.BoolTensor(seq['attention_mask'])}
        else:
            return seq_ids, target


class SecondaryStructureDataset(Dataset):
    """Custom Dataset for reading sequence data from a CSV file."""
    def __init__(
        self, 
        split,
        max_length,
        dataset_name="ss",
        d_output=3, # default binary classification
        dest_path=None,
        tokenizer=None,
        tokenizer_name=None,
        use_padding=True,
        add_eos=False,
        rc_aug=False,
        return_augs=False,
        return_mask=False,
    ):
        """
        Args:
            csv_file (string): Path to the csv file with seq and label.
        """
        # Load the data
        self.split = split
        self.max_length = max_length
        self.use_padding = use_padding
        self.tokenizer_name = tokenizer_name
        self.tokenizer = tokenizer
        self.return_augs = return_augs
        self.add_eos = add_eos
        self.d_output = d_output  # needed for decoder to grab
        self.rc_aug = rc_aug
        self.return_mask = return_mask
        csv_file = os.path.join(dest_path, f"{split}.csv")
        self.data = pd.read_csv(csv_file)
        self.label_to_id = {'H': 0, 'E':1, 'C': 2, 'G':3, 'I': 4, 'T': 5, 'S': 6, 'B': 7}

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.data)


    def __getitem__(self, idx):
        sequence = self.data.iloc[idx, 0]
        padding_length = self.max_length - len(sequence)
        q3_label = self.data.iloc[idx, 1]
        q8_label = self.data.iloc[idx, 2]
        q3_label = [self.label_to_id[char] for char in q3_label]
        q8_label = [self.label_to_id[char] for char in q8_label]

        assert len(q3_label) == len(q8_label)
        if len(q3_label) > self.max_length:
            q3_label = q3_label[:self.max_length]
            q8_label = q8_label[:self.max_length]
        # 如果需要填充，且填充长度大于0，则在前面添加填充值
        elif padding_length > 0:
            q3_label = [-100] * padding_length + q3_label
            q8_label = [-100] * padding_length + q8_label

        seq = self.tokenizer(sequence,
            add_special_tokens=True if self.add_eos else False,  # this is what controls adding eos
            padding="max_length" if self.use_padding else "do_not_pad",
            max_length=self.max_length,
            truncation=True,
        )

        seq_ids = seq["input_ids"]  # get input_ids
        # print(len(seq_ids), " ", len(label))
        assert len(seq_ids) == len(q3_label)

        seq_ids = torch.LongTensor(seq_ids)
        target = torch.LongTensor([q3_label])  # offset by 1, includes eos
        # print(len(seq_ids), " ", len(target))
        if self.return_mask:
            return seq_ids, target, {'mask': torch.BoolTensor(seq['attention_mask'])}
        else:
            return seq_ids, target
        

# class SecondaryStructureDataset(Dataset):
#     """Custom Dataset for reading sequence data from a CSV file."""
#     def __init__(
#         self, 
#         split,
#         max_length,
#         dataset_name="ss",
#         d_output=3, # default binary classification
#         dest_path=None,
#         tokenizer=None,
#         tokenizer_name=None,
#         use_padding=True,
#         add_eos=False,
#         rc_aug=False,
#         return_augs=False,
#         return_mask=False,
#     ):
#         """
#         Args:
#             csv_file (string): Path to the csv file with seq and label.
#         """
#         # Load the data
#         self.split = split
#         self.max_length = max_length
#         self.use_padding = use_padding
#         self.tokenizer_name = tokenizer_name
#         self.tokenizer = tokenizer
#         self.return_augs = return_augs
#         self.add_eos = add_eos
#         self.d_output = d_output  # needed for decoder to grab
#         self.rc_aug = rc_aug
#         self.return_mask = return_mask
#         csv_file = os.path.join(dest_path, f"{split}.csv")
#         self.data = pd.read_csv(csv_file)
#         self.label_to_id = {'H': 0, 'E':1, 'C': 2, 'G':3, 'I': 4, 'T': 5, 'S': 6, 'B': 7}

#     def __len__(self):
#         """Return the total number of samples in the dataset."""
#         return len(self.data)


#     def __getitem__(self, idx):
#         sequence = self.data.iloc[idx, 0]
#         padding_length = self.max_length - len(sequence)
#         q3_label = self.data.iloc[idx, 1]
#         q3_label = [int(char) for char in q3_label]
#         if len(q3_label) > self.max_length:
#             q3_label = q3_label[:self.max_length]
           
#         # 如果需要填充，且填充长度大于0，则在前面添加填充值
#         elif padding_length > 0:
#             q3_label = [-100] * padding_length + q3_label

#         seq = self.tokenizer(sequence,
#             add_special_tokens=True if self.add_eos else False,  # this is what controls adding eos
#             padding="max_length" if self.use_padding else "do_not_pad",
#             max_length=self.max_length,
#             truncation=True,
#         )

#         seq_ids = seq["input_ids"]  # get input_ids
#         # print(len(seq_ids), " ", len(label))
#         assert len(seq_ids) == len(q3_label)

#         seq_ids = torch.LongTensor(seq_ids)
#         target = torch.LongTensor([q3_label])  # offset by 1, includes eos
#         # print(len(seq_ids), " ", len(target))
#         if self.return_mask:
#             return seq_ids, target, {'mask': torch.BoolTensor(seq['attention_mask'])}
#         else:
#             return seq_ids, target


class DisorderDataset(Dataset):
    """Custom Dataset for reading sequence data from a CSV file."""
    def __init__(
        self, 
        split,
        max_length,
        dataset_name="disorder",
        d_output=2, # default binary classification
        dest_path=None,
        tokenizer=None,
        tokenizer_name=None,
        use_padding=True,
        add_eos=False,
        rc_aug=False,
        return_augs=False,
        return_mask=False,
    ):
        """
        Args:
            csv_file (string): Path to the csv file with seq and label.
        """
        # Load the data
        self.split = split
        self.max_length = max_length
        self.use_padding = use_padding
        self.tokenizer_name = tokenizer_name
        self.tokenizer = tokenizer
        self.return_augs = return_augs
        self.add_eos = add_eos
        self.d_output = d_output  # needed for decoder to grab
        self.rc_aug = rc_aug
        self.return_mask = return_mask
        csv_file = os.path.join(dest_path, f"{split}.csv")
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.data)


    def __getitem__(self, idx):
        sequence = self.data.iloc[idx, 0]
        padding_length = self.max_length - len(sequence)
        label = self.data.iloc[idx, 1]
        label = [int(char) for char in label]

        if len(label) > self.max_length:
            label = label[:self.max_length]
        # 如果需要填充，且填充长度大于0，则在前面添加填充值
        elif padding_length > 0:
            label = [-100] * padding_length + label

        seq = self.tokenizer(sequence,
            add_special_tokens=True if self.add_eos else False,  # this is what controls adding eos
            padding="max_length" if self.use_padding else "do_not_pad",
            max_length=self.max_length,
            truncation=True,
        )

        seq_ids = seq["input_ids"]  # get input_ids
        # print(len(seq_ids), " ", len(label))
        assert len(seq_ids) == len(label)

        seq_ids = torch.LongTensor(seq_ids)
        target = torch.LongTensor([label])  # offset by 1, includes eos
        # print(len(seq_ids), " ", len(target))
        if self.return_mask:
            return seq_ids, target, {'mask': torch.BoolTensor(seq['attention_mask'])}
        else:
            return seq_ids, target


class FluorescenceDataset(Dataset):
    """Custom Dataset for reading sequence data from a CSV file."""
    def __init__(
        self, 
        split,
        max_length,
        dataset_name="fluorescence",
        d_output=1, # default binary classification
        dest_path=None,
        tokenizer=None,
        tokenizer_name=None,
        use_padding=True,
        add_eos=False,
        rc_aug=False,
        return_augs=False,
        return_mask=False,
    ):
        """
        Args:
            csv_file (string): Path to the csv file with seq and label.
        """
        # Load the data
        self.split = split
        self.max_length = max_length
        self.use_padding = use_padding
        self.tokenizer_name = tokenizer_name
        self.tokenizer = tokenizer
        self.return_augs = return_augs
        self.add_eos = add_eos
        self.d_output = d_output  # needed for decoder to grab
        self.rc_aug = rc_aug
        self.return_mask = return_mask
        csv_file = os.path.join(dest_path, f"{split}.csv")
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.data)


    def __getitem__(self, idx):
        sequence = self.data.iloc[idx, 0]
        label = float(self.data.iloc[idx, 1])

        seq = self.tokenizer(sequence,
            add_special_tokens=True if self.add_eos else False,  # this is what controls adding eos
            padding="max_length" if self.use_padding else "do_not_pad",
            max_length=self.max_length,
            truncation=True,
        )
        seq_ids = seq["input_ids"]  # get input_ids
        seq_ids = torch.LongTensor(seq_ids)

        target = torch.tensor([label], dtype=torch.float)
        if self.return_mask:
            return seq_ids, target, {'mask': torch.BoolTensor(seq['attention_mask'])}
        else:
            return seq_ids, target


class StabilityDataset(Dataset):
    """Custom Dataset for reading sequence data from a CSV file."""
    def __init__(
        self, 
        split,
        max_length,
        dataset_name="stability",
        d_output=1, # default binary classification
        dest_path=None,
        tokenizer=None,
        tokenizer_name=None,
        use_padding=True,
        add_eos=False,
        rc_aug=False,
        return_augs=False,
        return_mask=False,
    ):
        """
        Args:
            csv_file (string): Path to the csv file with seq and label.
        """
        # Load the data
        self.split = split
        self.max_length = max_length
        self.use_padding = use_padding
        self.tokenizer_name = tokenizer_name
        self.tokenizer = tokenizer
        self.return_augs = return_augs
        self.add_eos = add_eos
        self.d_output = d_output  # needed for decoder to grab
        self.rc_aug = rc_aug
        self.return_mask = return_mask
        csv_file = os.path.join(dest_path, f"{split}.csv")
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.data)


    def __getitem__(self, idx):
        sequence = self.data.iloc[idx, 0]
        label = float(self.data.iloc[idx, 1])

        seq = self.tokenizer(sequence,
            add_special_tokens=True if self.add_eos else False,  # this is what controls adding eos
            padding="max_length" if self.use_padding else "do_not_pad",
            max_length=self.max_length,
            truncation=True,
        )
        seq_ids = seq["input_ids"]  # get input_ids
        seq_ids = torch.LongTensor(seq_ids)

        target = torch.tensor([label], dtype=torch.float)
        if self.return_mask:
            return seq_ids, target, {'mask': torch.BoolTensor(seq['attention_mask'])}
        else:
            return seq_ids, target


class TCRDataset(Dataset):
    def __init__(
        self,
        split,
        max_length,
        dataset_name="tcr",
        d_output=2, # default binary classification
        dest_path=None,
        tokenizer=None,
        tokenizer_name=None,
        use_padding=True,
        add_eos=False,
        rc_aug=False,
        return_augs=False,
        return_mask=False,
    ):
        
        self.split = split
        self.max_length = max_length
        self.use_padding = use_padding
        self.tokenizer_name = tokenizer_name
        self.tokenizer = tokenizer
        self.return_augs = return_augs
        self.add_eos = add_eos
        self.d_output = d_output  # needed for decoder to grab
        self.rc_aug = rc_aug
        self.return_mask = return_mask

        # base_path = Path(dest_path)  / split
        tsv_file = os.path.join(dest_path, f"{split}.tsv")
        self.data = pd.read_csv(tsv_file, sep='\t')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        epitope_seq = self.data.iloc[idx, 4]
        tcr_seq = self.data.iloc[idx, 5]
        
        sequence = epitope_seq + tcr_seq
        label = int(self.data.iloc[idx, 3])
        print(epitope_seq, ' ', tcr_seq)
        epitope_token = tokenizer.encode(epitope_seq)
        tcr_token = tokenizer.encode(tcr_seq)
        print(epitope_token, ' ', tcr_token)
        # seq_ids = tokenizer.build_inputs_with_special_tokens(epitope_token, tcr_token)

        seq_token = epitope_token + tcr_token
        padding_length = self.max_length - len(seq_token)
        if padding_length > 0:
            seq_token = [4] * padding_length + seq_token
        # seq = tokenizer.encode_plus(
        #     sequence,
        #     max_length=self.max_length,  # 指定最大长度
        #     padding='max_length',  # 指定padding到最大长度
        #     truncation=True,  # 如果超出最大长度则进行截断
        # )

        # seq_ids = seq["input_ids"]  # get input_ids

        seq_ids = torch.LongTensor(seq_token)

        target = torch.LongTensor([label])  # offset by 1, includes eos
    
        return seq_ids, target



if __name__ == "__main__":

    tokenizer = CharacterTokenizer(
        characters=['D', 'N', 'E', 'K', 'V', 'Y', 'A', 'Q', 'M', 'I', 'T', 
                    'L', 'R', 'F', 'G', 'C', 'S', 'P', 'H', 'W', 'X', 'U', 'B', 'O', 'Z'],
        model_max_length=1024,
        add_special_tokens=False,
        padding_side="left",
    )

    train_dataset = SecondaryStructureDataset(split='train',
        max_length=1024,
        dataset_name="homology",
        d_output=3, # default binary classification
        dest_path="/Users/zym/Downloads/Okumura_lab/data/protein_data/fine_tuning/secondary_structure",
        tokenizer=tokenizer,
        tokenizer_name="char",
        use_padding=True,
        add_eos=False,
        rc_aug=False,
        return_augs=False,
        return_mask=False,
        )
    dev_dataset = SecondaryStructureDataset(split='valid',
        max_length=1024,
        dataset_name="homology",
        d_output=3, # default binary classification
        dest_path="/Users/zym/Downloads/Okumura_lab/data/protein_data/fine_tuning/secondary_structure",
        tokenizer=tokenizer,
        tokenizer_name="char",
        use_padding=True,
        add_eos=False,
        rc_aug=False,
        return_augs=False,
        return_mask=False,
        )
    test_dataset = SecondaryStructureDataset(split='test',
        max_length=1024,
        dataset_name="homology",
        d_output=3, # default binary classification
        dest_path="/Users/zym/Downloads/Okumura_lab/data/protein_data/fine_tuning/secondary_structure",
        tokenizer=tokenizer,
        tokenizer_name="char",
        use_padding=True,
        add_eos=False,
        rc_aug=False,
        return_augs=False,
        return_mask=False,
        )
    
    
    print(train_dataset[0])
    print(len(train_dataset))
    print(len(dev_dataset))
    print(len(test_dataset))


    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


    # import pandas as pd

    # # Load the CSV file
    # file_path = '/Users/zym/Downloads/Okumura_lab/data/protein_data/fine_tuning/disorder/valid.csv'
    # data = pd.read_csv(file_path)

    # # Initialize lists to store lengths
    # seq_lengths = []
    # label_lengths = []

    # # Iterate through the DataFrame and calculate lengths
    # for index, row in data.iterrows():
    #     seq_length = len(row['seq'])
    #     label_length = len(row['label'])
    #     seq_lengths.append(seq_length)
    #     label_lengths.append(label_length)

    # Display the first few lengths for verification
    # seq_lengths[:5], label_lengths[:5]
    # print(seq_length)
    # print(label_lengths)
    # 遍历训练数据集
    # print("Training dataset:")
    # for i, batch in enumerate(train_loader):
    #     print(f"Batch {i}: {batch}")
    #     seq, label = batch
    #     cnt_seq, cnt_label = 0, 0
    #     for c in seq:
    #         if c == 4:
    #             cnt_seq += 1
    #     for c in label:
    #         if c == -100:
    #             cnt_label += 1
    #     assert cnt_seq == cnt_label
        

    # 遍历验证数据集
    # print("\nDevelopment dataset:")
    # for i, batch in enumerate(dev_loader):
    #     print(f"Batch {i}: ")
    #     seq, label = batch
        # print(len(seq))
        # if len(seq) > 1024:
        #     print(seq)

    # 遍历测试数据集
    # print("\nTest dataset:")
    # for i, batch in enumerate(test_loader):
    #     print(f"Batch {i}: {batch}")