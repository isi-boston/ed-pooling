# ed-pooling

This repository contains the code to reproduce the results presented in 
`The Impact of Subword Pooling Strategy for Cross-lingual Event Detection`.

## Table of Contents
1. [Setup](#Setup)
2. [Data](#data)
   * [ACE](#ACE)
   * [BETTER](#BETTER)
   * [MINION](#MINION)
3. [Training](#training)

## Setup
```bash
$ conda create -f environment.yml
$ conda activate ed-pooling
$ export PYTHONPATH=.
```

## Data
### ACE
The ACE training, validation, and test data is available in English, Arabic and
Chinese. 
Since Chinese is a non-white-space delimited language, we have excluded it from the present effort. 
We use the same train/dev/test splits as used in [Huang et al. 2022](https://aclanthology.org/2022.acl-long.317/) 
and [Xu et al. 2021](https://arxiv.org/abs/2103.02205).
The English and Arabic data, formatted to be compatible with the code in this repository,
is available under
```
data
├── ar-ace
│   ├── ar_dev.jhu.better-split-80.json
│   ├── ar_test.jhu.better-split-80.json
│   └── ar_train.jhu.better-split-80.json
└── en-ace
    ├── en_dev.jhu.better.json
    ├── en_test.jhu.better.json
    └── en_train.jhu.better.json
```

### BETTER
To access the BETTER data (Abstract/Phase-1/Phase-2), please visit the official [IARPA BETTER website](https://ir.nist.gov/better).

### MINION
This dataset was introduced in [Pouran Ben Veyseh et al. (2022)](https://aclanthology.org/2022.naacl-main.166/).
We use the same train/dev/test splits as in the official release. 
The data, formatted to be compatible with the code in this repository,
is available under
```
data
└── {en,es,hi,ko,pl,pt,tr}-minion
    ├── dev.json
    ├── test.json
    └── train.json
```

## Training

We give an example on how to train an `xlm-roberta-large` model on the Ace english data:
```bash
$ for strategy in first_token last_token average; do \
  for seed in 42; do \
    bash scripts/train_triggers.sh \
      en-ace \
      ${strategy} \
      xlm-roberta-large \
      expts/en-ace \
      ${seed}; \
  done; \
done

$ for strategy in attention; do \
  for seed in 42; do \
    bash scripts/train_triggers.sh \
      en-ace \
      ${strategy} \
      xlm-roberta-large \
      expts/en-ace \
      ${seed} \
      1.; \  # temperature
  done; \
done
```
