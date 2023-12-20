# CORDON
- This repository includes the codes and scripts for CORDON. Arxiv Link: [https://arxiv.org/abs/1911.00359](https://arxiv.org/abs/2312.01025).

## Quick Start

- With this repository, you can
  - Train the constrained-mscn model and reproduce the main results in the paper
  - Load the pretrained models for easy reproduction

## Content

```
├── CORDON === Libraries of CORDON
| ├── eligibility_checker === Library for constraint eligibility checking
| ├── loss_generator === Library for loss term generation
| └── query_augmentor ===  Library for training query augmentation
├── data === Information about datasets
├── mscn === Include the mscn model architecture and helper functions for training
├── pretrained_models === Pretrained models of MSCN and constrained-MSCN
|	├── dsb === For DSB dataset
|            ├── mscn.pth === Valina MSCN model
|            └── constrained-mscn.pth === Constrained-MSCN trained with CORDON
|	└── imdb === For IMDb dataset
|            ├── mscn.pth === Valina MSCN model
|            └── constrained-mscn.pth === Constrained-MSCN trained with CORDON
├── runtime_workloads === Query worklaods used for end-to-end experiments
|	├── dsb === For DSB dataset
|	└── imdb === For IMDb dataset
└── samples === Samples (1000 per table) used for bitmap lookup
|	├── dsb === For DSB dataset
|            ├── dsb-50.sql
|            ├── dsb-consistency.sql === DKS workload chosen by consistency constraint
|            └── dsb-pkfk.sql === DKS workload chosen by pk-fk constraint
|	└── IMDb === For IMDb dataset
|            ├── imdb-consistency.sql === DKS workload chosen by consistency constraint
|            ├── imdb-pkfk.sql === DKS workload chosen by pk-fk constraint
|            └── job-light.sql
├── sql_scripts === Dataset loading and index building for Postgres
├── wokloads === Training and test workloads for both datasets
├── train_dsb.py === Train Constrained-MSCN on DSB
└── train_imdb.py === Train Constrained-MSCN on IMDb
```

## Training
Due to the file size limit, please first download the zip file of all bitmap files from [here](https://drive.google.com/file/d/1-QMn1o7DdC6HJyVN3BzSyjHuo166ZCcP/view?usp=sharing) and unzip it into the [workload](workloads) folder.

Then, for IMDb dataset, run
```shell
$ python train_imdb.py --queries 30000 --epochs 50
```

For DSB dataset, run
```shell
$ python train_dsb.py --queries 20000 --epochs 50
```

- The programs will output the cardinality estimation performance (main results) in the terminal.
- You can easily save the trained model and check the constraint violation ratios. We plan to provide relative scripts soon later.
- We also provide trained models in [this folder](pretrained_models) for easy reproduction.


## End-to-End Runtime Performance

1. To reproduce the runtime experiments, please first download and install the modified version of PostgreSQL from [CEB project](https://github.com/learnedsystems/CEB) or [another project](https://github.com/Nathaniel-Han/End-to-End-CardEst-Benchmark/tree/master).
2. Please download the IMDb dataset from [here](http://homepages.cwi.nl/~boncz/job/imdb.tgz), and download the populated DSB dataset from [here](https://drive.google.com/file/d/1-KDtwtzt2wD_m0oA9M-rYcFaCWYeapFQ/view).
3. Then, please load the data into PostgreSQL and build indexes using scripts in [this folder](sql_scripts).
4. Now, you can run PostgreSQL over the workloads provided in [this folder](runtime_workloads).

