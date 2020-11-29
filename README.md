# BatchAML_Decorrelation

Code for paper 'Batch Decorrelation for Active Metric Learning', IJCAI-PRICAI 2020 https://arxiv.org/abs/2005.10008

## Abstract ##
We present an active learning strategy for training parametric models of distance metric, given triplet-based similarity comparisons: object A is more similar to object B than to object C. The standard active learning approaches degrade when annotations are requested for batches of triplets due to correlation among them. In this work, we propose a novel method to decorrelate batches of triplets, that jointly balances informativeness and diversity while decoupling the choice of a heuristic for each criterion.


## Datasets ##

For each dataset, the training, validation, and test triplets are present in the data folder. The file triplet.py contains all ground-truth triplets for the particular dataset


## Requirements
The model is implemented in PyTorch. Please install other Python libraries using requirements.txt

`$pip install -r requirements.txt`


## Train Model

Specify the data directory in the utils file for corresponding dataset. Train the model using scripts

`./run_food.sh`  or  `./run_haptic.sh`
