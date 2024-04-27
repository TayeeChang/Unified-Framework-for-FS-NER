# Unified Label-Aware Contrastive Learning Framework: 
Source code for the paper:   
A Unified Label-Aware Contrastive Learning Framework for Few-Shot Named
Entity Recognition

## Requirements
- python 3.8
- tensorflow==2.6
- keras_transformers

The package `keras_transformers` is based on the open-sourced transformer framework [keras_transformers](https://github.com/TayeeChang/keras_transformers). 
You can install by:  
```shell
pip install git+https://github.com/TayeeChang/keras_transformers.git
```

## Datasets
For the FEW-NERD datasets, you can refer to the [data source](https://github.com/thunlp/Few-NERD).
Due to the large size of the dataset, for each episode, we here afford one example to show the data format we use.
For more episodes, you can download them from the source above.

Some data sets are not publicly available, please obtain permission before use.

## Checkpoint
We use [BERT](https://github.com/google-research/bert) sourced by google as our pre-training checkpoint. 

## Running scripts
`train_in_source_domain.py` and `fine-tuning.py` are training and fine-tuning scripts.




