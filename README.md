# ProtHyena

## Important links:

- [biorxiv](https://www.biorxiv.org/content/10.1101/2024.01.18.576206v1)

- [Dataset](https://drive.google.com/drive/folders/1oNBFGQw3F9aoOYOQSB71PS1sfhhgwyzm?usp=sharing)
- [Model checkpoint](https://drive.google.com/file/d/1s89PV6sNCxSEq5Qztwqs-5XUxrngbc-z/view?usp=sharing)
- [Colab notebook](https://colab.research.google.com/drive/1kjSJbhslSX8k0XGS-JB9hAfK29z0FVZK#scrollTo=JWUc70H7Iiwd) for easy downstream inference.

## Intro

Welcome to the ProtHyena repo!

Credit: much of the code is forked and extended from [HyenaDNA](https://github.com/HazyResearch/hyena-dna) and [Safari](https://github.com/HazyResearch/safari).

## Dependencies

<a name="dependencies"></a>

For this repo, let's start with the dependancies that are needed.

- clone repo, cd into it

```
git clone https://github.com/ZHymLumine/ProtHyena.git
```

if you fail to run the command, you may need install [git lfs](https://git-lfs.com/) for cloning large files. Or you can just downdoad the zip file.

- create a conda environment, with Python 3.8

```
conda create -n prot-hyena python=3.8
```

- The repo is developed with Pytorch 2.4, using cuda 12.4

```
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

- install requirements:

```
pip install -r requirements.txt
```

- install Flash Attention, these [notes](https://github.com/HazyResearch/safari#getting-started) will be helpful.

```
cd ProtHyena
cd flash-attention
pip install -e . --no-build-isolation
```

## Pretrain

- to pretrain a prothyena model, in `ProtHyena` folder, run

```
CUDA_VISIBLE_DEVICES=0 python -m train wandb=null experiment=prot14m/prot14m_hyena trainer.devices=1
```

## Fine-tuning

Note: we have provided the pretrained checkpoint and dataset in the `checkpoint` and `data` folders in this repo for your convenience.

1. Download the [checkpoint](https://drive.google.com/file/d/1s89PV6sNCxSEq5Qztwqs-5XUxrngbc-z/view?usp=sharing) and put it into `checkpoint` folder. Change the `pretrained_model_path` in the file `experiment/prot14m/{task}.yaml` to the correct path on your computer.
2. download dataset (or use the dataset in `data` folder. Change the `dest_path` in the file `dataset/{task}.yaml` to the correct path on your computer.

3. For specific tasks, run the command below:

   - fluorescence

   ```
   CUDA_VISIBLE_DEVICES=0 python -m train wandb=null experiment=prot14m/fluorescence trainer.devices=1
   ```

   - stability

   ```
   CUDA_VISIBLE_DEVICES=0 python -m train wandb=null experiment=prot14m/stability trainer.devices=1
   ```

   - cleavage

   ```
   CUDA_VISIBLE_DEVICES=0 python -m train wandb=null experiment=prot14m/cleavage trainer.devices=1
   ```

   - disorder

   ```
   CUDA_VISIBLE_DEVICES=0 python -m train wandb=null experiment=prot14m/disorder trainer.devices=1
   ```

   - signal peptide

   ```
   CUDA_VISIBLE_DEVICES=0 python -m train wandb=null experiment=prot14m/signalP trainer.devices=1
   ```

   - solubility

   ```
   CUDA_VISIBLE_DEVICES=0 python -m train wandb=null experiment=prot14m/solubility trainer.devices=1
   ```

   you can change the batch size through command line.
   e.g

   ```
   CUDA_VISIBLE_DEVICES=0 python -m train experiment=prot14m/stability trainer.devices=1 dataset.batch_size=128 dataset.batch_size_eval=128
   ```

   or you can set these parameters in `configs/experiment/prot14m/{task}.yaml` for specific task.

## Finetune on a new downsteam task

To fine-tune on a new task, you need to create new configuration files in the `pipeline`, `experiment`, and `dataset` folders. You can follow the examples we provide in these folders.

For example, if you want to fine-tune a task called `fold_class` (you can name it anything, here we use `{task_name}` as a placeholder), you need to create the following files:

- `experiment/prot14m/{task_name}.yaml`
- `pipeline/{task_name}.yaml`
- `dataset/{task_name}.yaml`

### In `experiment/prot14m/{task_name}.yaml`:

1. Change `/pipeline:` in the `defaults` section to `{task_name}`.
2. Update `pretrained_model_path` to the correct path on your computer where the pretrained model is located.
3. Optionally, update the `metrics` by checking the available ones in `src/tasks/metrics.py`, or create a new one.

### In `pipeline/{task_name}.yaml`:

1. Change `/dataset:` in the `defaults` section to `{task_name}`.
2. If your task is at the protein sequence level (where a whole sequence gets a label), use:
   ```
   decoder:
     _name_: nd
     mode: pool
   ```
3. If your task is at the residue level (where each amino acid has a label), use:
   ```
   decoder:
     _name_: token
   ```

### In `dataset/{task_name}.yaml`:

1. Set `_name_` and `dataset_name` to `{task_name}`.
2. Set `dest_path` to the correct path where your data is stored.
3. Set `train_len` to the number of training examples.
4. Create `train.csv`, `valid.csv`, and `test.csv` files in the `dest_path` directory. These files should have two columns: `seq` (for the sequence) and `label` (for the label).

In `src/dataloaders/dataset/protein_bench_dataset.py`, create new Dataset class

example

```
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
```

In `src/dataloaders/proteomics.py`, create new dataloader class and import the Dataset class from `src.dataloaders.dataset.protein_bench_dataset`

```
from src.dataloaders.datasets.protein_bench_dataset import SignalPeptideDataset

class SignalPeptide(Prot14M):
    _name_ = "signalP"
    l_output = 0

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
        self.d_output = d_output  # Set this correct

		...

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

```

Make sure that the `_name_` matches your specific `{task_name}`. Set `d_output` to the number of classes for multi-class datasets, and use `d_output = 1` for regression tasks.

## Downstream Inference

If you'd like to use our fine-tuned model for downstream analysis (inference), follow our [Colab notebook](https://colab.research.google.com/drive/1kjSJbhslSX8k0XGS-JB9hAfK29z0FVZK#scrollTo=JWUc70H7Iiwd). The notebook is fully integrated with Hugging Face and provides everything you need to:

- **Load the model** and fine-tuned weights.
- **Run inference** on new data.
- **Extract embeddings** from protein sequences.

This notebook serves as a self-contained environment to streamline your workflow for prediction and further analysis.

## Citation

Feel free to cite us if you find our work useful :)

```
@article {Zhang2024.01.18.576206,
	author = {Yiming Zhang and Manabu Okumura},
	title = {ProtHyena: A fast and efficient foundation protein language model at single amino acid Resolution},
	elocation-id = {2024.01.18.576206},
	year = {2024},
	doi = {10.1101/2024.01.18.576206},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {The emergence of self-supervised deep language models has revolutionized natural language processing tasks and has recently extended its applications to biological sequence analysis. Traditional models, primarily based on the Transformer and BERT architectures, demonstrate substantial effectiveness in various applications. However, these models are inherently constrained by the attention mechanism{\textquoteright}s quadratic computational complexity O(L2), limiting their efficiency and the length of context they can process. Addressing these limitations, we introduce ProtHyena, a novel approach that leverages the Hyena operator. This innovative methodology circumvents the constraints imposed by attention mechanisms, thereby reducing the time complexity to a subquadratic, enabling the modeling of extra-long protein sequences at the single amino acid level without the need to compress data. ProtHyena is able to achieve, and in many cases exceed, state-of-the-art results in various downstream tasks with only 10\% of the parameters typically required by attention-based models. The architecture of ProtHyena presents a highly efficient solution for training protein predictors, offering a promising avenue for fast and efficient analysis of biological sequences.Competing Interest StatementThe authors have declared no competing interest.},
	URL = {https://www.biorxiv.org/content/early/2024/01/22/2024.01.18.576206},
	eprint = {https://www.biorxiv.org/content/early/2024/01/22/2024.01.18.576206.full.pdf},
	journal = {bioRxiv}
}
```
