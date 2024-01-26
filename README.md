# ProtHyena


## Important links:  
- [biorxiv](https://www.biorxiv.org/content/10.1101/2024.01.18.576206v1)  

## Intro

Welcome to the ProtHyena repo! 


  
Credit: much of the code is forked and extended from [S4](https://github.com/HazyResearch/state-spaces) and [Safari](https://github.com/HazyResearch/safari).


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
