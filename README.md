# ProtHyena

## Important links:

- [biorxiv](https://www.biorxiv.org/content/10.1101/2024.01.18.576206v1)

- [Dataset](https://drive.google.com/drive/folders/1oNBFGQw3F9aoOYOQSB71PS1sfhhgwyzm?usp=sharing)
- [Model checkpoint](https://drive.google.com/file/d/1s89PV6sNCxSEq5Qztwqs-5XUxrngbc-z/view?usp=sharing)

## Intro

Welcome to the ProtHyena repo!

Credit: much of the code is forked and extended from [HyenaDNA](https://github.com/HazyResearch/hyena-dna) and [Safari](https://github.com/HazyResearch/safari).

## Dependencies

<a name="dependencies"></a>

For this repo, let's start with the dependancies that are needed.

- clone repo, cd into it

```
git clone --recurse-submodules https://github.com/ZHymLumine/ProtHyena.git && cd ProtHyena
```

- create a conda environment, with Python 3.8+

```
conda create -n prot-hyena python=3.8
```

- The repo is developed with Pytorch 1.13, using cuda 11.7

```
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
```

- install requirements:

```
pip install -r requirements.txt
```

- install Flash Attention, these [notes](https://github.com/HazyResearch/safari#getting-started) will be helpful.

```
cd prot-hyena
git submodule update --init
cd flash-attention
git submodule update --init
pip install -e . --no-build-isolation
```

- optional fused layers for speed (takes a bit of time)

```
# from inside flash-attn/
cd csrc/layer_norm && pip install . --no-build-isolation
```

## Pretrain

<a name="pretrain"></a>

- to pretrain a prothyena model

```
CUDA_VISIBLE_DEVICES=0 python -m train wandb=null experiment=prot14m/prot14m_hyena trainer.devices=1
```

## Fine-tuning

Download the [checkpoint](https://drive.google.com/file/d/1s89PV6sNCxSEq5Qztwqs-5XUxrngbc-z/view?usp=sharing)
<a name="fine-tuning"></a>

- Disorder

```
CUDA_VISIBLE_DEVICES=0 python -m train wandb=null experiment=prot14m/disorder trainer.devices=1 train.prepretrained_model_path=/path/to/the/checkpoint
```

- Remote Homology

```
CUDA_VISIBLE_DEVICES=0 python -m train wandb=null experiment=prot14m/homology trainer.devices=1 train.prepretrained_model_path=/path/to/the/checkpoint
```

and etc.

- To fine-tune on a new downstream task
  add new dataset for evaluation, create new configs in `pipeline`, `experienment` and `dataset` folders. You can follow ours examples in those folders.

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
