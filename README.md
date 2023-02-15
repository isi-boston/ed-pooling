# ed-pooling

This repository contains the code to reproduce the results presented in 
`The Impact of Subword Pooling Strategy for Cross-lingual Event Detection`.

## Table of Contents
1. [Setup](#Setup)
2. [Data](#data)
   * [ACE](#ACE)
   * [BETTER](#BETTER)
   * [MINION](#MINION)
   * [IDF](#IDF)
3. [Training](#training)
4. [Prediction](#prediction)
5. [Score](#score)


## Setup
```bash
$ conda create -f environment.yml
$ conda activate ed-pooling
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
The data, formatted to be compatible with the code in this repository, is available under
```
data
└── {en,es,hi,ko,pl,pt,tr}-minion
    ├── dev.json
    ├── test.json
    └── train.json
```

### IDF
We gather a corpus in the language on interest and subsequently use the `scripts/token_scores.py` 
script to generate the IDF scores for each token. 
The files containing the IDF scores which were used in this work are provided under `data/idfs`.


## Training
We give an example on how to train an `xlm-roberta-large` model on the ACE english data.


For `strategy=first_token/last_token/average`:
```bash
$ TASK=en-ace
$ STRATEGY=first_token  # can be either first_token/last_token/average
$ MLM=xlm-roberta-large
$ OUTPUT_DIR=expts/en-ace
$ SEED=42
$ bash scripts/train_triggers.sh \
    ${TASK} \
    ${STRATEGY} \
    ${MLM} \
    ${OUTPUT_DIR} \
    ${SEED}
```

For `strategy=attention`:
```bash
$ STRATEGY=attention
$ ATTENTION_TEMPERATURE=1.
$ bash scripts/train_triggers.sh \
    ${TASK} \
    ${STRATEGY} \
    ${MLM} \
    ${OUTPUT_DIR} \
    ${SEED} \
    ${ATTENTION_TEMPERATURE}
```

For `strategy=idf`:
```bash
$ STRATEGY=idf
$ IDF_TEMPERATURE=1.
$ IDF_DEFAULT_SCORE=10
$ IDF_SCORES_FILE=data/idfs/en.tsv
$ bash scripts/train_triggers.sh \
    ${TASK} \
    ${STRATEGY} \
    ${MLM} \
    ${OUTPUT_DIR} \
    ${SEED} \
    ${IDF_TEMPERATURE} \
    ${IDF_DEFAULT_SCORE} \
    ${IDF_SCORES_FILE}
```

## Prediction
Using the models trained on English ACE, we can now make predictions on English/Arabic ACE test sets.
In the following, we provide commands to make predictions on Arabic ACE. 


For `strategy=first_token/last_token/average`:
```bash
$ TASK=en-ace
$ TRAINING_LANG=en
$ TASK_TYPE=AceTrigger
$ STRATEGY=first_token  # can be either first_token/last_token/average
$ MLM=xlm-roberta-large
$ MLM_TYPE=xlmr
$ OUTPUT_DIR=expts/en-ace
$ SEED=42
$ MAX_SEQ_LENGTH=128
$ TEST_INPUT_FILE=data/ar-ace/ar_test.jhu.better-split-80.json
$ OUTPUT_FILE=ar_test.jhu.better-split-80.preds.json
$ python run_token_classification.py \
    --task_type AceTrigger \
    --model_name_or_path ${MLM} \
    --model_type ${MLM_TYPE} \
    --test_file ${TEST_INPUT_FILE} \
    --do_predict \
    --preds_out_file ${OUTPUT_FILE} \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --output_dir ${OUTPUT_DIR}/${TASK}_${TRAINING_LANG}_${STRATEGY}_${MLM}_${SEED} \
    --seed ${SEED} \
    --pooling_strategy ${STRATEGY}
```

For `strategy=attention`:
```bash
$ STRATEGY=attention
$ ATTENTION_TEMPERATURE=1.
$ python run_token_classification.py \
    --task_type AceTrigger \
    --model_name_or_path ${MLM} \
    --model_type ${MLM_TYPE} \
    --test_file ${TEST_INPUT_FILE} \
    --do_predict \
    --preds_out_file ${OUTPUT_FILE} \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --output_dir ${OUTPUT_DIR}/${TASK}_${TRAINING_LANG}_${STRATEGY}_${MLM}_${SEED} \
    --seed ${SEED} \
    --pooling_strategy ${STRATEGY} \
    --token_scores_temperature ${ATTENTION_TEMPERATURE}
```

For `strategy=idf`:
```bash
$ STRATEGY=idf
$ IDF_TEMPERATURE=1.
$ IDF_DEFAULT_SCORE=10
$ IDF_SCORES_FILE=data/idfs/ar.tsv  # ar.tsv since we are predicting on Arabic data
$ python run_token_classification.py \
    --task_type AceTrigger \
    --model_name_or_path ${MLM} \
    --model_type ${MLM_TYPE} \
    --test_file ${TEST_INPUT_FILE} \
    --do_predict \
    --preds_out_file ${OUTPUT_FILE} \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --output_dir ${OUTPUT_DIR}/${TASK}_${TRAINING_LANG}_${STRATEGY}_${MLM}_${SEED} \
    --seed ${SEED} \
    --pooling_strategy ${STRATEGY} \
    --token_scores_temperature ${IDF_TEMPERATURE} \
    --default_token_score ${IDF_DEFAULT_SCORE} \
    --token_scores_file ${IDF_SCORES_FILE}
```

## Score
Once the predictions are made, we can evaluate the predictions against the gold test set.
```bash
$ TASK=en-ace
$ TRAINING_LANG=en
$ STRATEGY=first_token  # can be either first_token/last_token/average/attention/idf
$ MLM=xlm-roberta-large
$ OUTPUT_DIR=expts/en-ace
$ SEED=42
$ TEST_INPUT_FILE=data/ar-ace/ar_test.jhu.better-split-80.json
$ OUTPUT_FILE=ar_test.jhu.better-split-80.preds.json
$ python scripts/score_triggers.py \
    --gold ${TEST_INPUT_FILE} \
    --system ${OUTPUT_DIR}/${TASK}_${TRAINING_LANG}_${STRATEGY}_${MLM}_${SEED}/${OUTPUT_FILE} \
    --task ${TASK}
```

## How to Cite
```bibtex
```