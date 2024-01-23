# Adapted from https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_clm.py
# Adapted from https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
from pathlib import Path
from typing import Any, List, Union
from torch.utils.data.dataloader import DataLoader, Dataset
from transformers import AutoTokenizer
from datasets import Dataset

from src.dataloaders.base import SequenceDataset, default_data_path
from src.dataloaders.fault_tolerant_sampler import RandomFaultTolerantSampler
from src.dataloaders.fault_tolerant_sampler import FaultTolerantDistributedSampler
# genomics datasets
from src.dataloaders.datasets.hg38_char_tokenizer import CharacterTokenizer
from src.dataloaders.datasets.mlm_char_tokenizer import MLMCharacterTokenizer
from src.dataloaders.datasets.bpe_tokenizer import BPETokenizer
from src.dataloaders.datasets.protein_bench_dataset import ProteinLocationDataset, SolubilityDataset, HomologyDataset, DisorderDataset, FluorescenceDataset, StabilityDataset, SecondaryStructureDataset, FoldClassDataset, SignalPeptideDataset, CleavageDataset, TCRDataset
from src.dataloaders.datasets.prot14m_dataset import Prot14MDataset
from src.dataloaders.datasets.prot14m_dataset import Prot14MDatasetMLM

class Prot14M_MLM(SequenceDataset):
    """
    Base class, other dataloaders can inherit from this class.

    You must implement the following functions:
        - __init__
        - setup

    You can then use (already have access to) the following functions:
        - train_dataloader
        - val_dataloader
        - test_dataloader

    """
    ###### very important to set this! ######
    _name_ = "prot14m_mlm"  # this name is how the dataset config finds the right dataloader
    #########################################

    def __init__(self, fasta_file, tokenizer_name=None, dataset_config_name=None, max_length=1024, d_output=2, rc_aug=False,
                 max_length_val=None, max_length_test=None, val_ratio=0.0005, val_split_seed=2357, use_fixed_len_val=False,
                 add_eos=True, detokenize=False, val_only=False, batch_size=32, batch_size_eval=None, num_workers=1,
                 shuffle=False, pin_memory=False, drop_last=False, fault_tolerant=False, ddp=False,
                 fast_forward_epochs=None, fast_forward_batches=None, replace_N_token=False, pad_interval=False,
                 *args, **kwargs):
        self.dataset_config_name = dataset_config_name
        self.tokenizer_name = tokenizer_name
        self.d_output = d_output
        self.rc_aug = rc_aug  # reverse compliment augmentation
        self.max_length = max_length
        self.max_length_val = max_length_val if max_length_val is not None else max_length
        self.max_length_test = max_length_test if max_length_test is not None else max_length
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.add_eos = add_eos
        self.detokenize = detokenize
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        # self.bed_file = bed_file
        self.fasta_file = fasta_file
        self.use_fixed_len_val = use_fixed_len_val
        self.replace_N_token = replace_N_token
        self.pad_interval = pad_interval        

        # handle if file paths are None (default paths)
        # if self.bed_file is None:
        #     self.bed_file = default_data_path / self._name_ / 'human-sequences.bed'
        if self.fasta_file is None:
            self.fasta_file = default_data_path / self._name_ / 'Pfam-A.fasta'

        if fault_tolerant:
            assert self.shuffle
        self.fault_tolerant = fault_tolerant
        if ddp:
            assert fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        if self.fast_forward_epochs is not None or self.fast_forward_batches is not None:
            assert ddp and fault_tolerant

    def setup(self, stage=None):
        """Set up the tokenizer and init the datasets."""
        # TODO instantiate with registry

        if self.tokenizer_name == 'char':
            print("**Using Char-level tokenizer**")
            self.tokenizer = MLMCharacterTokenizer(
                characters=['D', 'N', 'E', 'K', 'V', 'Y', 'A', 'Q', 'M', 'I', 'T', 
                    'L', 'R', 'F', 'G', 'C', 'S', 'P', 'H', 'W', 'X', 'U', 'B', 'O', 'Z'],  # for protein seq
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=True,
                padding_side="right",
            )
        elif self.tokenizer_name == 'bpe':
            print("Using BPE tokenizer")
            self.tokenizer = BPETokenizer(
                vocab_file='/raid_elmo/home/lr/zym/data/protein_data/tokenization/byte-level-bpe/updated_vocab.json',
                merges_file='/raid_elmo/home/lr/zym/data/protein_data/tokenization/byte-level-bpe/merges.txt',
                model_max_length=self.max_length # 假设 max_length 是预定义的
            )
            # print("**using pretrained AIRI tokenizer**")
            # self.tokenizer = AutoTokenizer.from_pretrained('AIRI-Institute/gena-lm-bert-base')
            # print("**using local tokenizer**")
            # local_tokenizer_path = '/raid_elmo/home/lr/zym/data/protein_data/tokenization/byte-level-bpe'  # 将此路径替换为您本地tokenizer的实际路径
            # self.tokenizer = AutoTokenizer.from_pretrained(local_tokenizer_path, padding_side='left')

        self.vocab_size = len(self.tokenizer)

        self.init_datasets()  # creates the datasets.  You can also just create this inside the setup() here.

    def init_datasets(self):
        """Init the datasets (separate from the tokenizer)"""

        # delete old datasets to free memory
        if hasattr(self, 'dataset_train'):
            self.dataset_train.seqs.close()
            del self.dataset_train.seqs

        # delete old datasets to free memory
        if hasattr(self, 'dataset_test'):
            self.dataset_test.seqs.close()
            del self.dataset_test.seqs
    
        # Create all splits: torch datasets
        
        print("creat dataset")
        self.dataset_train, self.dataset_val, self.dataset_test = [
            Prot14MDatasetMLM(split=split,
                        fasta_file=self.fasta_file,
                        max_length=max_len,
                        tokenizer=self.tokenizer,  # pass the tokenize wrapper
                        tokenizer_name=self.tokenizer_name,
                        add_eos=self.add_eos,
                        return_seq_indices=False,
                        shift_augs=None,
                        rc_aug=self.rc_aug,
                        return_augs=False,
                        replace_N_token=self.replace_N_token,
                        pad_interval=self.pad_interval)
            for split, max_len in zip(['train', 'valid', 'test'], [self.max_length, self.max_length_val, self.max_length_test])
        ]

        print("train len: ", len(self.dataset_train), " val len:", len(self.dataset_val), " test len: ", len(self.dataset_test))
        # if self.use_fixed_len_val:
        #     # we're placing the fixed test set in the val dataloader, for visualization!!!
        #     # that means we should track mode with test loss, not val loss

        #     # new option to use fixed val set
        #     print("Using fixed length val set!")
        #     # start end of chr14 and chrX grabbed from Enformer
        #     chr_ranges = {'chr14': [19726402, 106677047],
        #                     'chrX': [2825622, 144342320],
        #                     }

        #     self.dataset_val = HG38FixedDataset(
        #         chr_ranges=chr_ranges,
        #         fasta_file=self.fasta_file,
        #         max_length=self.max_length,
        #         pad_max_length=self.max_length,
        #         tokenizer=self.tokenizer,
        #         add_eos=True,
        #     )

        return

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """ The train dataloader """
        if self.shuffle and self.fault_tolerant:
            shuffle = False
            # TD [2022-12-26]: We need the distributed_sampler_kwargs in case of model parallel:
            # In that case the number of replicas and the data parallel rank are more complicated.
            distributed_sampler_kwargs = self.trainer.distributed_sampler_kwargs
            sampler = (FaultTolerantDistributedSampler(self.dataset_train,
                                                       **self.trainer.distributed_sampler_kwargs)
                       if self.ddp else RandomFaultTolerantSampler(self.dataset_train))
            # TD [2022-08-06]: Only the DDP sampler supports fast-forwarding for now
            # We assume that it's being resumed with the same number of GPUs
            if self.ddp and self.fast_forward_epochs is not None and self.fast_forward_batches is not None:
                sampler.load_state_dict({
                    'epoch': self.fast_forward_epochs,
                    'counter': self.fast_forward_batches * self.batch_size
                })
        else:
            shuffle = self.shuffle
            sampler = None
        return self._data_loader(self.dataset_train, batch_size=self.batch_size,
                                 shuffle=shuffle, sampler=sampler)

    def val_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The val dataloader """
        return self._data_loader(self.dataset_val, batch_size=self.batch_size_eval)

    def test_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The test dataloader """
        return self._data_loader(self.dataset_test, batch_size=self.batch_size_eval)

    def _data_loader(self, dataset: Dataset, batch_size: int, shuffle: bool = False,
                     sampler=None) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=self.num_workers,  # Data is already in memory, we don't need many workers
            shuffle=shuffle,
            sampler=sampler,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

    def load_state_dict(self, checkpoint):
        if self.fault_tolerant:
            self.fast_forward_epochs = checkpoint['loops']['fit_loop']['epoch_progress']['current']['completed']
            # TD [2022-08-07] ['epoch_loop.batch_progress']['total']['completed'] is 1 iteration
            # behind, so we're using the optimizer's progress. This is set correctly in seq.py.
            self.fast_forward_batches = checkpoint['loops']['fit_loop']['epoch_loop.batch_progress']['current']['completed']
        # At this point the train loader hasn't been constructed yet
  

class Prot14M(SequenceDataset):
    """
    Base class, other dataloaders can inherit from this class.

    You must implement the following functions:
        - __init__
        - setup

    You can then use (already have access to) the following functions:
        - train_dataloader
        - val_dataloader
        - test_dataloader

    """
    ###### very important to set this! ######
    _name_ = "prot14m"  # this name is how the dataset config finds the right dataloader
    #########################################

    def __init__(self, fasta_file, tokenizer_name=None, dataset_config_name=None, max_length=1024, d_output=2, rc_aug=False,
                 max_length_val=None, max_length_test=None, val_ratio=0.0005, val_split_seed=2357, use_fixed_len_val=False,
                 add_eos=True, detokenize=False, val_only=False, batch_size=32, batch_size_eval=None, num_workers=1,
                 shuffle=False, pin_memory=False, drop_last=False, fault_tolerant=False, ddp=False,
                 fast_forward_epochs=None, fast_forward_batches=None, replace_N_token=False, pad_interval=False,
                 *args, **kwargs):
        self.dataset_config_name = dataset_config_name
        self.tokenizer_name = tokenizer_name
        self.d_output = d_output
        self.rc_aug = rc_aug  # reverse compliment augmentation
        self.max_length = max_length
        self.max_length_val = max_length_val if max_length_val is not None else max_length
        self.max_length_test = max_length_test if max_length_test is not None else max_length
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.add_eos = add_eos
        self.detokenize = detokenize
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        # self.bed_file = bed_file
        self.fasta_file = fasta_file
        self.use_fixed_len_val = use_fixed_len_val
        self.replace_N_token = replace_N_token
        self.pad_interval = pad_interval        

        # handle if file paths are None (default paths)
        # if self.bed_file is None:
        #     self.bed_file = default_data_path / self._name_ / 'human-sequences.bed'
        if self.fasta_file is None:
            self.fasta_file = default_data_path / self._name_ / 'Pfam-A.fasta'

        if fault_tolerant:
            assert self.shuffle
        self.fault_tolerant = fault_tolerant
        if ddp:
            assert fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        if self.fast_forward_epochs is not None or self.fast_forward_batches is not None:
            assert ddp and fault_tolerant

    def setup(self, stage=None):
        """Set up the tokenizer and init the datasets."""
        # TODO instantiate with registry

        if self.tokenizer_name == 'char':
            print("**Using Char-level tokenizer**")
            self.tokenizer = CharacterTokenizer(
                characters=['D', 'N', 'E', 'K', 'V', 'Y', 'A', 'Q', 'M', 'I', 'T', 
                    'L', 'R', 'F', 'G', 'C', 'S', 'P', 'H', 'W', 'X', 'U', 'B', 'O', 'Z'],  # for protein seq
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
            )
        elif self.tokenizer_name == 'bpe':
            print("Using BPE tokenizer")
            self.tokenizer = BPETokenizer(
                vocab_file='/raid_elmo/home/lr/zym/data/protein_data/tokenization/byte-level-bpe/updated_vocab.json',
                merges_file='/raid_elmo/home/lr/zym/data/protein_data/tokenization/byte-level-bpe/merges.txt',
                model_max_length=self.max_length # 假设 max_length 是预定义的
            )
            # print("**using pretrained AIRI tokenizer**")
            # self.tokenizer = AutoTokenizer.from_pretrained('AIRI-Institute/gena-lm-bert-base')
            # print("**using local tokenizer**")
            # local_tokenizer_path = '/raid_elmo/home/lr/zym/data/protein_data/tokenization/byte-level-bpe'  # 将此路径替换为您本地tokenizer的实际路径
            # self.tokenizer = AutoTokenizer.from_pretrained(local_tokenizer_path, padding_side='left')

        self.vocab_size = len(self.tokenizer)

        self.init_datasets()  # creates the datasets.  You can also just create this inside the setup() here.

    def init_datasets(self):
        """Init the datasets (separate from the tokenizer)"""

        # delete old datasets to free memory
        if hasattr(self, 'dataset_train'):
            self.dataset_train.seqs.close()
            del self.dataset_train.seqs

        # delete old datasets to free memory
        if hasattr(self, 'dataset_test'):
            self.dataset_test.seqs.close()
            del self.dataset_test.seqs
    
        # Create all splits: torch datasets
        
        print("creat dataset")
        self.dataset_train, self.dataset_val, self.dataset_test = [
            Prot14MDataset(split=split,
                        fasta_file=self.fasta_file,
                        max_length=max_len,
                        tokenizer=self.tokenizer,  # pass the tokenize wrapper
                        tokenizer_name=self.tokenizer_name,
                        add_eos=self.add_eos,
                        return_seq_indices=False,
                        shift_augs=None,
                        rc_aug=self.rc_aug,
                        return_augs=False,
                        replace_N_token=self.replace_N_token,
                        pad_interval=self.pad_interval)
            for split, max_len in zip(['train', 'valid', 'test'], [self.max_length, self.max_length_val, self.max_length_test])
        ]

        print("train len: ", len(self.dataset_train), " val len:", len(self.dataset_val), " test len: ", len(self.dataset_test))
        # if self.use_fixed_len_val:
        #     # we're placing the fixed test set in the val dataloader, for visualization!!!
        #     # that means we should track mode with test loss, not val loss

        #     # new option to use fixed val set
        #     print("Using fixed length val set!")
        #     # start end of chr14 and chrX grabbed from Enformer
        #     chr_ranges = {'chr14': [19726402, 106677047],
        #                     'chrX': [2825622, 144342320],
        #                     }

        #     self.dataset_val = HG38FixedDataset(
        #         chr_ranges=chr_ranges,
        #         fasta_file=self.fasta_file,
        #         max_length=self.max_length,
        #         pad_max_length=self.max_length,
        #         tokenizer=self.tokenizer,
        #         add_eos=True,
        #     )

        return

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """ The train dataloader """
        if self.shuffle and self.fault_tolerant:
            shuffle = False
            # TD [2022-12-26]: We need the distributed_sampler_kwargs in case of model parallel:
            # In that case the number of replicas and the data parallel rank are more complicated.
            distributed_sampler_kwargs = self.trainer.distributed_sampler_kwargs
            sampler = (FaultTolerantDistributedSampler(self.dataset_train,
                                                       **self.trainer.distributed_sampler_kwargs)
                       if self.ddp else RandomFaultTolerantSampler(self.dataset_train))
            # TD [2022-08-06]: Only the DDP sampler supports fast-forwarding for now
            # We assume that it's being resumed with the same number of GPUs
            if self.ddp and self.fast_forward_epochs is not None and self.fast_forward_batches is not None:
                sampler.load_state_dict({
                    'epoch': self.fast_forward_epochs,
                    'counter': self.fast_forward_batches * self.batch_size
                })
        else:
            shuffle = self.shuffle
            sampler = None
        return self._data_loader(self.dataset_train, batch_size=self.batch_size,
                                 shuffle=shuffle, sampler=sampler)

    def val_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The val dataloader """
        return self._data_loader(self.dataset_val, batch_size=self.batch_size_eval)

    def test_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The test dataloader """
        return self._data_loader(self.dataset_test, batch_size=self.batch_size_eval)

    def _data_loader(self, dataset: Dataset, batch_size: int, shuffle: bool = False,
                     sampler=None) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=self.num_workers,  # Data is already in memory, we don't need many workers
            shuffle=shuffle,
            sampler=sampler,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

    def load_state_dict(self, checkpoint):
        if self.fault_tolerant:
            self.fast_forward_epochs = checkpoint['loops']['fit_loop']['epoch_progress']['current']['completed']
            # TD [2022-08-07] ['epoch_loop.batch_progress']['total']['completed'] is 1 iteration
            # behind, so we're using the optimizer's progress. This is set correctly in seq.py.
            self.fast_forward_batches = checkpoint['loops']['fit_loop']['epoch_loop.batch_progress']['current']['completed']
        # At this point the train loader hasn't been constructed yet
  
  

class ProteinLocation(Prot14M):
    _name_ = "location"
    l_output = 10  # need to set this for decoder to work correctly

    def __init__(self, dest_path=None, tokenizer_name=None, dataset_config_name=None, d_output=24, max_length=1024, rc_aug=False,
                 max_length_val=None, max_length_test=None, cache_dir=None, val_ratio=0.0005, val_split_seed=2357,
                 add_eos=True, detokenize=False, val_only=False, batch_size=32, batch_size_eval=None, num_workers=1,
                 shuffle=False, pin_memory=False, drop_last=False, fault_tolerant=False, ddp=False,
                 fast_forward_epochs=None, fast_forward_batches=None,
                total_size=None, remove_tail_ends=False, cutoff_train=0.1, cutoff_test=0.2,
                 *args, **kwargs):
        self.dataset_config_name = dataset_config_name
        self.tokenizer_name = tokenizer_name
        self.rc_aug = rc_aug  # reverse compliment augmentation
        self.dest_path = dest_path
        self.d_output = d_output
        self.cache_dir = None if cache_dir is None else Path(cache_dir).expanduser()
        self.max_length = max_length
        self.max_length_val = max_length_val if max_length_val is not None else max_length
        self.max_length_test = max_length_test if max_length_test is not None else max_length
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.add_eos = add_eos
        self.detokenize = detokenize
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.total_size = total_size
        self.remove_tail_ends = remove_tail_ends
        self.cutoff_train = cutoff_train
        self.cutoff_test = cutoff_test

        
        if fault_tolerant:
            assert self.shuffle
        self.fault_tolerant = fault_tolerant
        if ddp:
            assert fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        if self.fast_forward_epochs is not None or self.fast_forward_batches is not None:
            assert ddp and fault_tolerant

    def setup(self, stage=None):
        if self.tokenizer_name == 'char':
            print("**Using Char-level tokenizer**")
            self.tokenizer = CharacterTokenizer(
                characters=['D', 'N', 'E', 'K', 'V', 'Y', 'A', 'Q', 'M', 'I', 'T', 
                    'L', 'R', 'F', 'G', 'C', 'S', 'P', 'H', 'W', 'X', 'U', 'B', 'O', 'Z'],
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
            )
        elif self.tokenizer_name == 'bpe':
            print("Using BPE tokenizer")
            self.tokenizer = BPETokenizer(
                vocab_file='/raid_elmo/home/lr/zym/data/protein_data/tokenization/byte-level-bpe/updated_vocab.json',
                merges_file='/raid_elmo/home/lr/zym/data/protein_data/tokenization/byte-level-bpe/merges.txt',
                model_max_length=self.max_length # 假设 max_length 是预定义的
            )
            # print("**using pretrained AIRI tokenizer**")
            # self.tokenizer = AutoTokenizer.from_pretrained('AIRI-Institute/gena-lm-bert-base')
        else:
            raise ValueError(f"Invalid tokenizer name: {self.tokenizer_name}")

        self.vocab_size = len(self.tokenizer)
        
        # Create datasets
        self.init_datasets()

    def init_datasets(self):

        # delete old datasets
        # NOTE: For some reason only works to close files for train
        if hasattr(self, 'dataset_train'):
            # for spec in list(self.dataset_train.fastas.keys()):
            #     for chromosome in list(self.dataset_train.fastas[spec].keys()):
            #         self.dataset_train.fastas[spec][chromosome].close()
            #         del self.dataset_train.fastas[spec][chromosome]
            pass
        if hasattr(self, 'dataset_val'):
            pass

        if hasattr(self, 'dataset_test'):
            pass

        # Create all splits: torch datasets
        self.dataset_train, self.dataset_val, self.dataset_test = [
            ProteinLocationDataset(split=split,
                            max_length=max_len,
                            dest_path=self.dest_path,
                            d_output=self.d_output,
                            tokenizer=self.tokenizer,  # pass the tokenize wrapper
                            tokenizer_name=self.tokenizer_name,
                            add_eos=self.add_eos,
                            rc_aug=self.rc_aug,
                            )
            for split, max_len in zip(['train', 'test', 'test'], [self.max_length, self.max_length_val, self.max_length_test])
        ]
        return


class Solubility(Prot14M):
    _name_ = "solubility"
    l_output = 0  # need to set this for decoder to work correctly

    def __init__(self, dest_path=None, tokenizer_name=None, dataset_config_name=None, d_output=2, max_length=1024, rc_aug=False,
                 max_length_val=None, max_length_test=None, cache_dir=None, val_ratio=0.0005, val_split_seed=2357,
                 add_eos=True, detokenize=False, val_only=False, batch_size=32, batch_size_eval=None, num_workers=1,
                 shuffle=False, pin_memory=False, drop_last=False, fault_tolerant=False, ddp=False,
                 fast_forward_epochs=None, fast_forward_batches=None,
                total_size=None, remove_tail_ends=False, cutoff_train=0.1, cutoff_test=0.2,
                 *args, **kwargs):
        self.dataset_config_name = dataset_config_name
        self.tokenizer_name = tokenizer_name
        self.rc_aug = rc_aug  # reverse compliment augmentation
        self.dest_path = dest_path
        self.d_output = d_output
        self.cache_dir = None if cache_dir is None else Path(cache_dir).expanduser()
        self.max_length = max_length
        self.max_length_val = max_length_val if max_length_val is not None else max_length
        self.max_length_test = max_length_test if max_length_test is not None else max_length
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.add_eos = add_eos
        self.detokenize = detokenize
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.total_size = total_size
        self.remove_tail_ends = remove_tail_ends
        self.cutoff_train = cutoff_train
        self.cutoff_test = cutoff_test

        
        if fault_tolerant:
            assert self.shuffle
        self.fault_tolerant = fault_tolerant
        if ddp:
            assert fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        if self.fast_forward_epochs is not None or self.fast_forward_batches is not None:
            assert ddp and fault_tolerant

    def setup(self, stage=None):
        if self.tokenizer_name == 'char':
            print("**Using Char-level tokenizer**")
            self.tokenizer = CharacterTokenizer(
                characters=['D', 'N', 'E', 'K', 'V', 'Y', 'A', 'Q', 'M', 'I', 'T', 
                    'L', 'R', 'F', 'G', 'C', 'S', 'P', 'H', 'W', 'X', 'U', 'B', 'O', 'Z'],
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
            )
        elif self.tokenizer_name == 'bpe':
            print("Using BPE tokenizer")
            self.tokenizer = BPETokenizer(
                vocab_file='/raid_elmo/home/lr/zym/data/protein_data/tokenization/byte-level-bpe/updated_vocab.json',
                merges_file='/raid_elmo/home/lr/zym/data/protein_data/tokenization/byte-level-bpe/merges.txt',
                model_max_length=self.max_length # 假设 max_length 是预定义的
            )
            # print("**using pretrained AIRI tokenizer**")
            # self.tokenizer = AutoTokenizer.from_pretrained('AIRI-Institute/gena-lm-bert-base')
        else:
            raise ValueError(f"Invalid tokenizer name: {self.tokenizer_name}")

        self.vocab_size = len(self.tokenizer)
        
        # Create datasets
        self.init_datasets()

    def init_datasets(self):

        # delete old datasets
        # NOTE: For some reason only works to close files for train
        if hasattr(self, 'dataset_train'):
            # for spec in list(self.dataset_train.fastas.keys()):
            #     for chromosome in list(self.dataset_train.fastas[spec].keys()):
            #         self.dataset_train.fastas[spec][chromosome].close()
            #         del self.dataset_train.fastas[spec][chromosome]
            pass
        if hasattr(self, 'dataset_val'):
            pass

        if hasattr(self, 'dataset_test'):
            pass

        # Create all splits: torch datasets
        self.dataset_train, self.dataset_val, self.dataset_test = [
            SolubilityDataset(split=split,
                            max_length=max_len,
                            dest_path=self.dest_path,
                            d_output=self.d_output,
                            tokenizer=self.tokenizer,  # pass the tokenize wrapper
                            tokenizer_name=self.tokenizer_name,
                            add_eos=self.add_eos,
                            rc_aug=self.rc_aug,
                            )
            for split, max_len in zip(['train', 'valid', 'test'], [self.max_length, self.max_length_val, self.max_length_test])
        ]
        return


class SignalPeptide(Prot14M):
    _name_ = "signalP"
    l_output = 0  # need to set this for decoder to work correctly

    def __init__(self, dest_path=None, tokenizer_name=None, dataset_config_name=None, d_output=2, max_length=1024, rc_aug=False,
                 max_length_val=None, max_length_test=None, cache_dir=None, val_ratio=0.0005, val_split_seed=2357,
                 add_eos=True, detokenize=False, val_only=False, batch_size=32, batch_size_eval=None, num_workers=1,
                 shuffle=False, pin_memory=False, drop_last=False, fault_tolerant=False, ddp=False,
                 fast_forward_epochs=None, fast_forward_batches=None,
                total_size=None, remove_tail_ends=False, cutoff_train=0.1, cutoff_test=0.2,
                 *args, **kwargs):
        self.dataset_config_name = dataset_config_name
        self.tokenizer_name = tokenizer_name
        self.rc_aug = rc_aug  # reverse compliment augmentation
        self.dest_path = dest_path
        self.d_output = d_output
        self.cache_dir = None if cache_dir is None else Path(cache_dir).expanduser()
        self.max_length = max_length
        self.max_length_val = max_length_val if max_length_val is not None else max_length
        self.max_length_test = max_length_test if max_length_test is not None else max_length
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.add_eos = add_eos
        self.detokenize = detokenize
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.total_size = total_size
        self.remove_tail_ends = remove_tail_ends
        self.cutoff_train = cutoff_train
        self.cutoff_test = cutoff_test

        
        if fault_tolerant:
            assert self.shuffle
        self.fault_tolerant = fault_tolerant
        if ddp:
            assert fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        if self.fast_forward_epochs is not None or self.fast_forward_batches is not None:
            assert ddp and fault_tolerant

    def setup(self, stage=None):
        if self.tokenizer_name == 'char':
            print("**Using Char-level tokenizer**")
            self.tokenizer = CharacterTokenizer(
                characters=['D', 'N', 'E', 'K', 'V', 'Y', 'A', 'Q', 'M', 'I', 'T', 
                    'L', 'R', 'F', 'G', 'C', 'S', 'P', 'H', 'W', 'X', 'U', 'B', 'O', 'Z'],
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
            )
        elif self.tokenizer_name == 'bpe':
            print("Using BPE tokenizer")
            self.tokenizer = BPETokenizer(
                vocab_file='/raid_elmo/home/lr/zym/data/protein_data/tokenization/byte-level-bpe/updated_vocab.json',
                merges_file='/raid_elmo/home/lr/zym/data/protein_data/tokenization/byte-level-bpe/merges.txt',
                model_max_length=self.max_length # 假设 max_length 是预定义的
            )
            # print("**using pretrained AIRI tokenizer**")
            # self.tokenizer = AutoTokenizer.from_pretrained('AIRI-Institute/gena-lm-bert-base')
        else:
            raise ValueError(f"Invalid tokenizer name: {self.tokenizer_name}")

        self.vocab_size = len(self.tokenizer)
        
        # Create datasets
        self.init_datasets()

    def init_datasets(self):

        # delete old datasets
        # NOTE: For some reason only works to close files for train
        if hasattr(self, 'dataset_train'):
            # for spec in list(self.dataset_train.fastas.keys()):
            #     for chromosome in list(self.dataset_train.fastas[spec].keys()):
            #         self.dataset_train.fastas[spec][chromosome].close()
            #         del self.dataset_train.fastas[spec][chromosome]
            pass
        if hasattr(self, 'dataset_val'):
            pass

        if hasattr(self, 'dataset_test'):
            pass

        # Create all splits: torch datasets
        self.dataset_train, self.dataset_val, self.dataset_test = [
            SignalPeptideDataset(split=split,
                            max_length=max_len,
                            dest_path=self.dest_path,
                            d_output=self.d_output,
                            tokenizer=self.tokenizer,  # pass the tokenize wrapper
                            tokenizer_name=self.tokenizer_name,
                            add_eos=self.add_eos,
                            rc_aug=self.rc_aug,
                            )
            for split, max_len in zip(['train', 'test', 'test'], [self.max_length, self.max_length_val, self.max_length_test])
        ]
        return


class Cleavage(Prot14M):
    _name_ = "cleavage"
    l_output = 0  # need to set this for decoder to work correctly

    def __init__(self, dest_path=None, tokenizer_name=None, dataset_config_name=None, d_output=2, max_length=1024, rc_aug=False,
                 max_length_val=None, max_length_test=None, cache_dir=None, val_ratio=0.0005, val_split_seed=2357,
                 add_eos=True, detokenize=False, val_only=False, batch_size=32, batch_size_eval=None, num_workers=1,
                 shuffle=False, pin_memory=False, drop_last=False, fault_tolerant=False, ddp=False,
                 fast_forward_epochs=None, fast_forward_batches=None,
                total_size=None, remove_tail_ends=False, cutoff_train=0.1, cutoff_test=0.2,
                 *args, **kwargs):
        self.dataset_config_name = dataset_config_name
        self.tokenizer_name = tokenizer_name
        self.rc_aug = rc_aug  # reverse compliment augmentation
        self.dest_path = dest_path
        self.d_output = d_output
        self.cache_dir = None if cache_dir is None else Path(cache_dir).expanduser()
        self.max_length = max_length
        self.max_length_val = max_length_val if max_length_val is not None else max_length
        self.max_length_test = max_length_test if max_length_test is not None else max_length
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.add_eos = add_eos
        self.detokenize = detokenize
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.total_size = total_size
        self.remove_tail_ends = remove_tail_ends
        self.cutoff_train = cutoff_train
        self.cutoff_test = cutoff_test

        
        if fault_tolerant:
            assert self.shuffle
        self.fault_tolerant = fault_tolerant
        if ddp:
            assert fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        if self.fast_forward_epochs is not None or self.fast_forward_batches is not None:
            assert ddp and fault_tolerant

    def setup(self, stage=None):
        if self.tokenizer_name == 'char':
            print("**Using Char-level tokenizer**")
            self.tokenizer = CharacterTokenizer(
                characters=['D', 'N', 'E', 'K', 'V', 'Y', 'A', 'Q', 'M', 'I', 'T', 
                    'L', 'R', 'F', 'G', 'C', 'S', 'P', 'H', 'W', 'X', 'U', 'B', 'O', 'Z'],
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
            )
        elif self.tokenizer_name == 'bpe':
            print("Using BPE tokenizer")
            self.tokenizer = BPETokenizer(
                vocab_file='/raid_elmo/home/lr/zym/data/protein_data/tokenization/byte-level-bpe/updated_vocab.json',
                merges_file='/raid_elmo/home/lr/zym/data/protein_data/tokenization/byte-level-bpe/merges.txt',
                model_max_length=self.max_length # 假设 max_length 是预定义的
            )
            # print("**using pretrained AIRI tokenizer**")
            # self.tokenizer = AutoTokenizer.from_pretrained('AIRI-Institute/gena-lm-bert-base')
        else:
            raise ValueError(f"Invalid tokenizer name: {self.tokenizer_name}")

        self.vocab_size = len(self.tokenizer)
        
        # Create datasets
        self.init_datasets()

    def init_datasets(self):

        # delete old datasets
        # NOTE: For some reason only works to close files for train
        if hasattr(self, 'dataset_train'):
            # for spec in list(self.dataset_train.fastas.keys()):
            #     for chromosome in list(self.dataset_train.fastas[spec].keys()):
            #         self.dataset_train.fastas[spec][chromosome].close()
            #         del self.dataset_train.fastas[spec][chromosome]
            pass
        if hasattr(self, 'dataset_val'):
            pass

        if hasattr(self, 'dataset_test'):
            pass

        # Create all splits: torch datasets
        self.dataset_train, self.dataset_val, self.dataset_test = [
            CleavageDataset(split=split,
                            max_length=max_len,
                            dest_path=self.dest_path,
                            d_output=self.d_output,
                            tokenizer=self.tokenizer,  # pass the tokenize wrapper
                            tokenizer_name=self.tokenizer_name,
                            add_eos=self.add_eos,
                            rc_aug=self.rc_aug,
                            )
            for split, max_len in zip(['train', 'valid', 'test'], [self.max_length, self.max_length_val, self.max_length_test])
        ]
        return


class Homology(Prot14M):
    _name_ = "homology"
    l_output = 0  # need to set this for decoder to work correctly

    def __init__(self, dest_path=None, tokenizer_name=None, dataset_config_name=None, d_output=1195, max_length=1024, rc_aug=False,
                 max_length_val=None, max_length_test=None, cache_dir=None, val_ratio=0.0005, val_split_seed=2357,
                 add_eos=True, detokenize=False, val_only=False, batch_size=32, batch_size_eval=None, num_workers=1,
                 shuffle=False, pin_memory=False, drop_last=False, fault_tolerant=False, ddp=False,
                 fast_forward_epochs=None, fast_forward_batches=None,
                total_size=None, remove_tail_ends=False, cutoff_train=0.1, cutoff_test=0.2,
                 *args, **kwargs):
        self.dataset_config_name = dataset_config_name
        self.tokenizer_name = tokenizer_name
        self.rc_aug = rc_aug  # reverse compliment augmentation
        self.dest_path = dest_path
        self.d_output = d_output
        self.cache_dir = None if cache_dir is None else Path(cache_dir).expanduser()
        self.max_length = max_length
        self.max_length_val = max_length_val if max_length_val is not None else max_length
        self.max_length_test = max_length_test if max_length_test is not None else max_length
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.add_eos = add_eos
        self.detokenize = detokenize
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.total_size = total_size
        self.remove_tail_ends = remove_tail_ends
        self.cutoff_train = cutoff_train
        self.cutoff_test = cutoff_test

        
        if fault_tolerant:
            assert self.shuffle
        self.fault_tolerant = fault_tolerant
        if ddp:
            assert fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        if self.fast_forward_epochs is not None or self.fast_forward_batches is not None:
            assert ddp and fault_tolerant

    def setup(self, stage=None):
        if self.tokenizer_name == 'char':
            print("**Using Char-level tokenizer**")
            self.tokenizer = CharacterTokenizer(
                characters=['D', 'N', 'E', 'K', 'V', 'Y', 'A', 'Q', 'M', 'I', 'T', 
                    'L', 'R', 'F', 'G', 'C', 'S', 'P', 'H', 'W', 'X', 'U', 'B', 'O', 'Z'],
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
            )
        elif self.tokenizer_name == 'bpe':
            print("Using BPE tokenizer")
            self.tokenizer = BPETokenizer(
                vocab_file='/raid_elmo/home/lr/zym/data/protein_data/tokenization/byte-level-bpe/updated_vocab.json',
                merges_file='/raid_elmo/home/lr/zym/data/protein_data/tokenization/byte-level-bpe/merges.txt',
                model_max_length=self.max_length # 假设 max_length 是预定义的
            )
            # print("**using pretrained AIRI tokenizer**")
            # self.tokenizer = AutoTokenizer.from_pretrained('AIRI-Institute/gena-lm-bert-base')
        else:
            raise ValueError(f"Invalid tokenizer name: {self.tokenizer_name}")

        self.vocab_size = len(self.tokenizer)
        
        # Create datasets
        self.init_datasets()

    def init_datasets(self):

        # delete old datasets
        # NOTE: For some reason only works to close files for train
        if hasattr(self, 'dataset_train'):
            pass
        if hasattr(self, 'dataset_val'):
            pass

        if hasattr(self, 'dataset_test'):
            pass

        # Create all splits: torch datasets
        self.dataset_train, self.dataset_val, self.dataset_test = [
            HomologyDataset(split=split,
                            max_length=max_len,
                            dest_path=self.dest_path,
                            d_output=self.d_output,
                            tokenizer=self.tokenizer,  # pass the tokenize wrapper
                            tokenizer_name=self.tokenizer_name,
                            add_eos=self.add_eos,
                            rc_aug=self.rc_aug,
                            )
            for split, max_len in zip(['train', 'valid', 'test'], [self.max_length, self.max_length_val, self.max_length_test])
        ]
        return
    

class FoldClass(Prot14M):
    _name_ = "fold_class"
    l_output = 0  # need to set this for decoder to work correctly

    def __init__(self, dest_path=None, tokenizer_name=None, dataset_config_name=None, d_output=7, max_length=1024, rc_aug=False,
                 max_length_val=None, max_length_test=None, cache_dir=None, val_ratio=0.0005, val_split_seed=2357,
                 add_eos=True, detokenize=False, val_only=False, batch_size=32, batch_size_eval=None, num_workers=1,
                 shuffle=False, pin_memory=False, drop_last=False, fault_tolerant=False, ddp=False,
                 fast_forward_epochs=None, fast_forward_batches=None,
                total_size=None, remove_tail_ends=False, cutoff_train=0.1, cutoff_test=0.2,
                 *args, **kwargs):
        self.dataset_config_name = dataset_config_name
        self.tokenizer_name = tokenizer_name
        self.rc_aug = rc_aug  # reverse compliment augmentation
        self.dest_path = dest_path
        self.d_output = d_output
        self.cache_dir = None if cache_dir is None else Path(cache_dir).expanduser()
        self.max_length = max_length
        self.max_length_val = max_length_val if max_length_val is not None else max_length
        self.max_length_test = max_length_test if max_length_test is not None else max_length
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.add_eos = add_eos
        self.detokenize = detokenize
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.total_size = total_size
        self.remove_tail_ends = remove_tail_ends
        self.cutoff_train = cutoff_train
        self.cutoff_test = cutoff_test

        
        if fault_tolerant:
            assert self.shuffle
        self.fault_tolerant = fault_tolerant
        if ddp:
            assert fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        if self.fast_forward_epochs is not None or self.fast_forward_batches is not None:
            assert ddp and fault_tolerant

    def setup(self, stage=None):
        if self.tokenizer_name == 'char':
            print("**Using Char-level tokenizer**")
            self.tokenizer = CharacterTokenizer(
                characters=['D', 'N', 'E', 'K', 'V', 'Y', 'A', 'Q', 'M', 'I', 'T', 
                    'L', 'R', 'F', 'G', 'C', 'S', 'P', 'H', 'W', 'X', 'U', 'B', 'O', 'Z'],
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
            )
        elif self.tokenizer_name == 'bpe':
            print("Using BPE tokenizer")
            self.tokenizer = BPETokenizer(
                vocab_file='/raid_elmo/home/lr/zym/data/protein_data/tokenization/byte-level-bpe/updated_vocab.json',
                merges_file='/raid_elmo/home/lr/zym/data/protein_data/tokenization/byte-level-bpe/merges.txt',
                model_max_length=self.max_length # 假设 max_length 是预定义的
            )
            # print("**using pretrained AIRI tokenizer**")
            # self.tokenizer = AutoTokenizer.from_pretrained('AIRI-Institute/gena-lm-bert-base')
        else:
            raise ValueError(f"Invalid tokenizer name: {self.tokenizer_name}")

        self.vocab_size = len(self.tokenizer)
        
        # Create datasets
        self.init_datasets()

    def init_datasets(self):

        # delete old datasets
        # NOTE: For some reason only works to close files for train
        if hasattr(self, 'dataset_train'):
            # for spec in list(self.dataset_train.fastas.keys()):
            #     for chromosome in list(self.dataset_train.fastas[spec].keys()):
            #         self.dataset_train.fastas[spec][chromosome].close()
            #         del self.dataset_train.fastas[spec][chromosome]
            pass
        if hasattr(self, 'dataset_val'):
            pass

        if hasattr(self, 'dataset_test'):
            pass

        # Create all splits: torch datasets
        self.dataset_train, self.dataset_val, self.dataset_test = [
            FoldClassDataset(split=split,
                            max_length=max_len,
                            dest_path=self.dest_path,
                            d_output=self.d_output,
                            tokenizer=self.tokenizer,  # pass the tokenize wrapper
                            tokenizer_name=self.tokenizer_name,
                            add_eos=self.add_eos,
                            rc_aug=self.rc_aug,
                            )
            for split, max_len in zip(['train', 'test', 'test'], [self.max_length, self.max_length_val, self.max_length_test])
        ]
        return
    

class SecondaryStructure(Prot14M):
    _name_ = "ss"
    l_output = 0  # need to set this for decoder to work correctly

    def __init__(self, dest_path=None, tokenizer_name=None, dataset_config_name=None, d_output=3, max_length=1024, rc_aug=False,
                 max_length_val=None, max_length_test=None, cache_dir=None, val_ratio=0.0005, val_split_seed=2357,
                 add_eos=True, detokenize=False, val_only=False, batch_size=32, batch_size_eval=None, num_workers=1,
                 shuffle=False, pin_memory=False, drop_last=False, fault_tolerant=False, ddp=False,
                 fast_forward_epochs=None, fast_forward_batches=None,
                total_size=None, remove_tail_ends=False, cutoff_train=0.1, cutoff_test=0.2,
                 *args, **kwargs):
        self.dataset_config_name = dataset_config_name
        self.tokenizer_name = tokenizer_name
        self.rc_aug = rc_aug  # reverse compliment augmentation
        self.dest_path = dest_path
        self.d_output = d_output
        self.cache_dir = None if cache_dir is None else Path(cache_dir).expanduser()
        self.max_length = max_length
        self.max_length_val = max_length_val if max_length_val is not None else max_length
        self.max_length_test = max_length_test if max_length_test is not None else max_length
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.add_eos = add_eos
        self.detokenize = detokenize
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.total_size = total_size
        self.remove_tail_ends = remove_tail_ends
        self.cutoff_train = cutoff_train
        self.cutoff_test = cutoff_test

        
        if fault_tolerant:
            assert self.shuffle
        self.fault_tolerant = fault_tolerant
        if ddp:
            assert fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        if self.fast_forward_epochs is not None or self.fast_forward_batches is not None:
            assert ddp and fault_tolerant

    def setup(self, stage=None):
        if self.tokenizer_name == 'char':
            print("**Using Char-level tokenizer**")
            self.tokenizer = CharacterTokenizer(
                characters=['D', 'N', 'E', 'K', 'V', 'Y', 'A', 'Q', 'M', 'I', 'T', 
                    'L', 'R', 'F', 'G', 'C', 'S', 'P', 'H', 'W', 'X', 'U', 'B', 'O', 'Z'],
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
            )
        elif self.tokenizer_name == 'bpe':
            print("Using BPE tokenizer")
            self.tokenizer = BPETokenizer(
                vocab_file='/raid_elmo/home/lr/zym/data/protein_data/tokenization/byte-level-bpe/updated_vocab.json',
                merges_file='/raid_elmo/home/lr/zym/data/protein_data/tokenization/byte-level-bpe/merges.txt',
                model_max_length=self.max_length # 假设 max_length 是预定义的
            )
            # print("**using pretrained AIRI tokenizer**")
            # self.tokenizer = AutoTokenizer.from_pretrained('AIRI-Institute/gena-lm-bert-base')
        else:
            raise ValueError(f"Invalid tokenizer name: {self.tokenizer_name}")

        self.vocab_size = len(self.tokenizer)
        
        # Create datasets
        self.init_datasets()

    def init_datasets(self):

        # delete old datasets
        # NOTE: For some reason only works to close files for train
        if hasattr(self, 'dataset_train'):
            # for spec in list(self.dataset_train.fastas.keys()):
            #     for chromosome in list(self.dataset_train.fastas[spec].keys()):
            #         self.dataset_train.fastas[spec][chromosome].close()
            #         del self.dataset_train.fastas[spec][chromosome]
            pass
        if hasattr(self, 'dataset_val'):
            pass

        if hasattr(self, 'dataset_test'):
            pass

        # Create all splits: torch datasets
        self.dataset_train, self.dataset_val, self.dataset_test = [
            SecondaryStructureDataset(split=split,
                            max_length=max_len,
                            dest_path=self.dest_path,
                            d_output=self.d_output,
                            tokenizer=self.tokenizer,  # pass the tokenize wrapper
                            tokenizer_name=self.tokenizer_name,
                            add_eos=self.add_eos,
                            rc_aug=self.rc_aug,
                            )
            for split, max_len in zip(['training_hhblits', 'CB513', 'CB513'], [self.max_length, self.max_length_val, self.max_length_test])
        ]
        return


class Disorder(Prot14M):
    _name_ = "disorder"
    l_output = 0  # need to set this for decoder to work correctly

    def __init__(self, dest_path=None, tokenizer_name=None, dataset_config_name=None, d_output=2, max_length=1024, rc_aug=False,
                 max_length_val=None, max_length_test=None, cache_dir=None, val_ratio=0.0005, val_split_seed=2357,
                 add_eos=True, detokenize=False, val_only=False, batch_size=32, batch_size_eval=None, num_workers=1,
                 shuffle=False, pin_memory=False, drop_last=False, fault_tolerant=False, ddp=False,
                 fast_forward_epochs=None, fast_forward_batches=None,
                total_size=None, remove_tail_ends=False, cutoff_train=0.1, cutoff_test=0.2,
                 *args, **kwargs):
        self.dataset_config_name = dataset_config_name
        self.tokenizer_name = tokenizer_name
        self.rc_aug = rc_aug  # reverse compliment augmentation
        self.dest_path = dest_path
        self.d_output = d_output
        self.cache_dir = None if cache_dir is None else Path(cache_dir).expanduser()
        self.max_length = max_length
        self.max_length_val = max_length_val if max_length_val is not None else max_length
        self.max_length_test = max_length_test if max_length_test is not None else max_length
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.add_eos = add_eos
        self.detokenize = detokenize
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.total_size = total_size
        self.remove_tail_ends = remove_tail_ends
        self.cutoff_train = cutoff_train
        self.cutoff_test = cutoff_test

        
        if fault_tolerant:
            assert self.shuffle
        self.fault_tolerant = fault_tolerant
        if ddp:
            assert fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        if self.fast_forward_epochs is not None or self.fast_forward_batches is not None:
            assert ddp and fault_tolerant

    def setup(self, stage=None):
        if self.tokenizer_name == 'char':
            print("**Using Char-level tokenizer**")
            self.tokenizer = CharacterTokenizer(
                characters=['D', 'N', 'E', 'K', 'V', 'Y', 'A', 'Q', 'M', 'I', 'T', 
                    'L', 'R', 'F', 'G', 'C', 'S', 'P', 'H', 'W', 'X', 'U', 'B', 'O', 'Z'],
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
            )
        elif self.tokenizer_name == 'bpe':
            print("Using BPE tokenizer")
            self.tokenizer = BPETokenizer(
                vocab_file='/raid_elmo/home/lr/zym/data/protein_data/tokenization/byte-level-bpe/updated_vocab.json',
                merges_file='/raid_elmo/home/lr/zym/data/protein_data/tokenization/byte-level-bpe/merges.txt',
                model_max_length=self.max_length # 假设 max_length 是预定义的
            )
            # print("**using pretrained AIRI tokenizer**")
            # self.tokenizer = AutoTokenizer.from_pretrained('AIRI-Institute/gena-lm-bert-base')
        else:
            raise ValueError(f"Invalid tokenizer name: {self.tokenizer_name}")

        self.vocab_size = len(self.tokenizer)
        
        # Create datasets
        self.init_datasets()

    def init_datasets(self):

        # delete old datasets
        # NOTE: For some reason only works to close files for train
        if hasattr(self, 'dataset_train'):
            # for spec in list(self.dataset_train.fastas.keys()):
            #     for chromosome in list(self.dataset_train.fastas[spec].keys()):
            #         self.dataset_train.fastas[spec][chromosome].close()
            #         del self.dataset_train.fastas[spec][chromosome]
            pass
        if hasattr(self, 'dataset_val'):
            pass

        if hasattr(self, 'dataset_test'):
            pass

        # Create all splits: torch datasets
        self.dataset_train, self.dataset_val, self.dataset_test = [
            DisorderDataset(split=split,
                            max_length=max_len,
                            dest_path=self.dest_path,
                            d_output=self.d_output,
                            tokenizer=self.tokenizer,  # pass the tokenize wrapper
                            tokenizer_name=self.tokenizer_name,
                            add_eos=self.add_eos,
                            rc_aug=self.rc_aug,
                            )
            for split, max_len in zip(['train', 'valid', 'test'], [self.max_length, self.max_length_val, self.max_length_test])
        ]
        return
    

class Fluorescence(Prot14M):
    _name_ = "fluorescence"
    l_output = 0  # need to set this for decoder to work correctly

    def __init__(self, dest_path=None, tokenizer_name=None, dataset_config_name=None, d_output=1, max_length=1024, rc_aug=False,
                 max_length_val=None, max_length_test=None, cache_dir=None, val_ratio=0.0005, val_split_seed=2357,
                 add_eos=True, detokenize=False, val_only=False, batch_size=32, batch_size_eval=None, num_workers=1,
                 shuffle=False, pin_memory=False, drop_last=False, fault_tolerant=False, ddp=False,
                 fast_forward_epochs=None, fast_forward_batches=None,
                total_size=None, remove_tail_ends=False, cutoff_train=0.1, cutoff_test=0.2,
                 *args, **kwargs):
        self.dataset_config_name = dataset_config_name
        self.tokenizer_name = tokenizer_name
        self.rc_aug = rc_aug  # reverse compliment augmentation
        self.dest_path = dest_path
        self.d_output = d_output
        self.cache_dir = None if cache_dir is None else Path(cache_dir).expanduser()
        self.max_length = max_length
        self.max_length_val = max_length_val if max_length_val is not None else max_length
        self.max_length_test = max_length_test if max_length_test is not None else max_length
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.add_eos = add_eos
        self.detokenize = detokenize
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.total_size = total_size
        self.remove_tail_ends = remove_tail_ends
        self.cutoff_train = cutoff_train
        self.cutoff_test = cutoff_test

        
        if fault_tolerant:
            assert self.shuffle
        self.fault_tolerant = fault_tolerant
        if ddp:
            assert fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        if self.fast_forward_epochs is not None or self.fast_forward_batches is not None:
            assert ddp and fault_tolerant

    def setup(self, stage=None):
        if self.tokenizer_name == 'char':
            print("**Using Char-level tokenizer**")
            self.tokenizer = CharacterTokenizer(
                characters=['D', 'N', 'E', 'K', 'V', 'Y', 'A', 'Q', 'M', 'I', 'T', 
                    'L', 'R', 'F', 'G', 'C', 'S', 'P', 'H', 'W', 'X', 'U', 'B', 'O', 'Z'],
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
            )
        elif self.tokenizer_name == 'bpe':
            print("Using BPE tokenizer")
            self.tokenizer = BPETokenizer(
                vocab_file='/raid_elmo/home/lr/zym/data/protein_data/tokenization/byte-level-bpe/updated_vocab.json',
                merges_file='/raid_elmo/home/lr/zym/data/protein_data/tokenization/byte-level-bpe/merges.txt',
                model_max_length=self.max_length # 假设 max_length 是预定义的
            )
            # print("**using pretrained AIRI tokenizer**")
            # self.tokenizer = AutoTokenizer.from_pretrained('AIRI-Institute/gena-lm-bert-base')
        else:
            raise ValueError(f"Invalid tokenizer name: {self.tokenizer_name}")

        self.vocab_size = len(self.tokenizer)
        
        # Create datasets
        self.init_datasets()

    def init_datasets(self):

        # delete old datasets
        # NOTE: For some reason only works to close files for train
        if hasattr(self, 'dataset_train'):
            # for spec in list(self.dataset_train.fastas.keys()):
            #     for chromosome in list(self.dataset_train.fastas[spec].keys()):
            #         self.dataset_train.fastas[spec][chromosome].close()
            #         del self.dataset_train.fastas[spec][chromosome]
            pass
        if hasattr(self, 'dataset_val'):
            pass

        if hasattr(self, 'dataset_test'):
            pass

        # Create all splits: torch datasets
        self.dataset_train, self.dataset_val, self.dataset_test = [
            FluorescenceDataset(split=split,
                            max_length=max_len,
                            dest_path=self.dest_path,
                            d_output=self.d_output,
                            tokenizer=self.tokenizer,  # pass the tokenize wrapper
                            tokenizer_name=self.tokenizer_name,
                            add_eos=self.add_eos,
                            rc_aug=self.rc_aug,
                            )
            for split, max_len in zip(['train', 'valid', 'test'], [self.max_length, self.max_length_val, self.max_length_test])
        ]
        return


class Stability(Prot14M):
    _name_ = "stability"
    l_output = 0  # need to set this for decoder to work correctly

    def __init__(self, dest_path=None, tokenizer_name=None, dataset_config_name=None, d_output=1, max_length=1024, rc_aug=False,
                 max_length_val=None, max_length_test=None, cache_dir=None, val_ratio=0.0005, val_split_seed=2357,
                 add_eos=True, detokenize=False, val_only=False, batch_size=32, batch_size_eval=None, num_workers=1,
                 shuffle=False, pin_memory=False, drop_last=False, fault_tolerant=False, ddp=False,
                 fast_forward_epochs=None, fast_forward_batches=None,
                total_size=None, remove_tail_ends=False, cutoff_train=0.1, cutoff_test=0.2,
                 *args, **kwargs):
        self.dataset_config_name = dataset_config_name
        self.tokenizer_name = tokenizer_name
        self.rc_aug = rc_aug  # reverse compliment augmentation
        self.dest_path = dest_path
        self.d_output = d_output
        self.cache_dir = None if cache_dir is None else Path(cache_dir).expanduser()
        self.max_length = max_length
        self.max_length_val = max_length_val if max_length_val is not None else max_length
        self.max_length_test = max_length_test if max_length_test is not None else max_length
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.add_eos = add_eos
        self.detokenize = detokenize
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.total_size = total_size
        self.remove_tail_ends = remove_tail_ends
        self.cutoff_train = cutoff_train
        self.cutoff_test = cutoff_test

        
        if fault_tolerant:
            assert self.shuffle
        self.fault_tolerant = fault_tolerant
        if ddp:
            assert fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        if self.fast_forward_epochs is not None or self.fast_forward_batches is not None:
            assert ddp and fault_tolerant

    def setup(self, stage=None):
        if self.tokenizer_name == 'char':
            print("**Using Char-level tokenizer**")
            self.tokenizer = CharacterTokenizer(
                characters=['D', 'N', 'E', 'K', 'V', 'Y', 'A', 'Q', 'M', 'I', 'T', 
                    'L', 'R', 'F', 'G', 'C', 'S', 'P', 'H', 'W', 'X', 'U', 'B', 'O', 'Z'],
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
            )
        elif self.tokenizer_name == 'bpe':
            print("Using BPE tokenizer")
            self.tokenizer = BPETokenizer(
                vocab_file='/raid_elmo/home/lr/zym/data/protein_data/tokenization/byte-level-bpe/updated_vocab.json',
                merges_file='/raid_elmo/home/lr/zym/data/protein_data/tokenization/byte-level-bpe/merges.txt',
                model_max_length=self.max_length # 假设 max_length 是预定义的
            )
            # print("**using pretrained AIRI tokenizer**")
            # self.tokenizer = AutoTokenizer.from_pretrained('AIRI-Institute/gena-lm-bert-base')
        else:
            raise ValueError(f"Invalid tokenizer name: {self.tokenizer_name}")

        self.vocab_size = len(self.tokenizer)
        
        # Create datasets
        self.init_datasets()

    def init_datasets(self):

        # delete old datasets
        # NOTE: For some reason only works to close files for train
        if hasattr(self, 'dataset_train'):
            # for spec in list(self.dataset_train.fastas.keys()):
            #     for chromosome in list(self.dataset_train.fastas[spec].keys()):
            #         self.dataset_train.fastas[spec][chromosome].close()
            #         del self.dataset_train.fastas[spec][chromosome]
            pass
        if hasattr(self, 'dataset_val'):
            pass

        if hasattr(self, 'dataset_test'):
            pass

        # Create all splits: torch datasets
        self.dataset_train, self.dataset_val, self.dataset_test = [
            StabilityDataset(split=split,
                            max_length=max_len,
                            dest_path=self.dest_path,
                            d_output=self.d_output,
                            tokenizer=self.tokenizer,  # pass the tokenize wrapper
                            tokenizer_name=self.tokenizer_name,
                            add_eos=self.add_eos,
                            rc_aug=self.rc_aug,
                            )
            for split, max_len in zip(['train', 'valid', 'test'], [self.max_length, self.max_length_val, self.max_length_test])
        ]
        return


class TCR(Prot14M):
    _name_ = "tcr"
    l_output = 0  # need to set this for decoder to work correctly

    def __init__(self, dest_path=None, tokenizer_name=None, dataset_config_name=None, d_output=2, max_length=1024, rc_aug=False,
                 max_length_val=None, max_length_test=None, cache_dir=None, val_ratio=0.0005, val_split_seed=2357,
                 add_eos=True, detokenize=False, val_only=False, batch_size=32, batch_size_eval=None, num_workers=1,
                 shuffle=False, pin_memory=False, drop_last=False, fault_tolerant=False, ddp=False,
                 fast_forward_epochs=None, fast_forward_batches=None,
                total_size=None, remove_tail_ends=False, cutoff_train=0.1, cutoff_test=0.2,
                 *args, **kwargs):
        self.dataset_config_name = dataset_config_name
        self.tokenizer_name = tokenizer_name
        self.rc_aug = rc_aug  # reverse compliment augmentation
        self.dest_path = dest_path
        self.d_output = d_output
        self.cache_dir = None if cache_dir is None else Path(cache_dir).expanduser()
        self.max_length = max_length
        self.max_length_val = max_length_val if max_length_val is not None else max_length
        self.max_length_test = max_length_test if max_length_test is not None else max_length
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.add_eos = add_eos
        self.detokenize = detokenize
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.total_size = total_size
        self.remove_tail_ends = remove_tail_ends
        self.cutoff_train = cutoff_train
        self.cutoff_test = cutoff_test

        
        if fault_tolerant:
            assert self.shuffle
        self.fault_tolerant = fault_tolerant
        if ddp:
            assert fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        if self.fast_forward_epochs is not None or self.fast_forward_batches is not None:
            assert ddp and fault_tolerant

    def setup(self, stage=None):
        if self.tokenizer_name == 'char':
            print("**Using Char-level tokenizer**")
            self.tokenizer = CharacterTokenizer(
                characters=['D', 'N', 'E', 'K', 'V', 'Y', 'A', 'Q', 'M', 'I', 'T', 
                    'L', 'R', 'F', 'G', 'C', 'S', 'P', 'H', 'W', 'X', 'U', 'B', 'O', 'Z'],
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
            )
        elif self.tokenizer_name == 'bpe':
            print("Using BPE tokenizer")
            self.tokenizer = BPETokenizer(
                vocab_file='/raid_elmo/home/lr/zym/data/protein_data/tokenization/byte-level-bpe/updated_vocab.json',
                merges_file='/raid_elmo/home/lr/zym/data/protein_data/tokenization/byte-level-bpe/merges.txt',
                model_max_length=self.max_length # 假设 max_length 是预定义的
            )
            # print("**using pretrained AIRI tokenizer**")
            # self.tokenizer = AutoTokenizer.from_pretrained('AIRI-Institute/gena-lm-bert-base')
        else:
            raise ValueError(f"Invalid tokenizer name: {self.tokenizer_name}")

        self.vocab_size = len(self.tokenizer)
        
        # Create datasets
        self.init_datasets()

    def init_datasets(self):

        # delete old datasets
        # NOTE: For some reason only works to close files for train
        if hasattr(self, 'dataset_train'):
            # for spec in list(self.dataset_train.fastas.keys()):
            #     for chromosome in list(self.dataset_train.fastas[spec].keys()):
            #         self.dataset_train.fastas[spec][chromosome].close()
            #         del self.dataset_train.fastas[spec][chromosome]
            pass
        if hasattr(self, 'dataset_val'):
            pass

        if hasattr(self, 'dataset_test'):
            pass

        # Create all splits: torch datasets
        self.dataset_train, self.dataset_val, self.dataset_test = [
            TCRDataset(split=split,
                            max_length=max_len,
                            dest_path=self.dest_path,
                            d_output=self.d_output,
                            tokenizer=self.tokenizer,  # pass the tokenize wrapper
                            tokenizer_name=self.tokenizer_name,
                            add_eos=self.add_eos,
                            rc_aug=self.rc_aug,
                            )
            for split, max_len in zip(['train', 'dev', 'test'], [self.max_length, self.max_length_val, self.max_length_test])
        ]
        return

# if __name__ == '__main__':
#     """Quick test using dataloader. Can't call from here though."""

#     loader = HG38(
#         bed_file='/home/exnx/enformer-pytorch/data/basenji/human-sequences.bed',
#         fasta_file='/home/exnx/enformer-pytorch/data/basenji/hg38.ml.fa',
#         tokenizer_name='char_level', max_length=2000
#     )

    # breakpoint()

    # it = iter(ds)
    # elem = next(it)
    # print(len(elem))
    # breakpoint()
