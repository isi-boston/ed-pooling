#! /bin/bash

set -euo pipefail

TASK=$1
STRATEGY=$2
MODEL_NAME=$3
EXPT_DIR=$4
SEED=$5

THIS_DIR=$(dirname $0)
TRAIN_LANG="en"
LEARNING_RATE=5e-5
BATCH_SIZE=16
MAX_SEQ_LENGTH=128
DATA_DIR=${THIS_DIR}/../data/${TASK}

if [ $TASK == "abstract" ]; then
  TASK_TYPE="BetterAbstractTrigger"
  TRAIN_FILE=$DATA_DIR/abstract.original-english.train.augment_gold.en.json
  VALID_FILE=$DATA_DIR/abstract.original-english.devtest.augment_gold.en.json
  TEST_FILE=$DATA_DIR/abstract.original-english.analysis.augment_gold.en.json
  if [ $MODEL_NAME == "xlm-roberta-large" ]; then
    MODEL_TYPE='xlmr'
    TRAIN_EPOCHS=5
  fi

elif [ $TASK == "phase1" ]; then
  TASK_TYPE="BetterBasicTrigger"
  TRAIN_FILE=$DATA_DIR/basic-phase1.original-english.train.augment_gold.en.json
  VALID_FILE=$DATA_DIR/basic-phase1.original-english.devtest.augment_gold.en.json
  TEST_FILE=$DATA_DIR/basic-phase1.original-english.analysis.augment_gold.en.json
  if [ $MODEL_NAME == "xlm-roberta-large" ]; then
      MODEL_TYPE='xlmr'
      TRAIN_EPOCHS=50
  fi

elif [ $TASK == "phase2" ]; then
  TASK_TYPE="BetterBasicTrigger"
  TRAIN_FILE=$DATA_DIR/basic-phase2.original-english.train.augment_gold.en.json
  VALID_FILE=$DATA_DIR/basic-phase2.original-english.devtest.augment_gold.en.json
  TEST_FILE=$DATA_DIR/basic-phase2.original-english.analysis.augment_gold.en.json
  if [ $MODEL_NAME == "xlm-roberta-large" ]; then
      MODEL_TYPE='xlmr'
      TRAIN_EPOCHS=50
  fi

elif [ $TASK == "en-ace" ]; then
  TASK_TYPE="AceTrigger"
  TRAIN_FILE=$DATA_DIR/en_train.jhu.better.json
  VALID_FILE=$DATA_DIR/en_dev.jhu.better.json
  TEST_FILE=$DATA_DIR/en_test.jhu.better.json
  if [ $MODEL_NAME == "xlm-roberta-large" ]; then
      MODEL_TYPE='xlmr'
      TRAIN_EPOCHS=20
      LEARNING_RATE=5e-6
  fi
elif [ $TASK == "ar-ace" ]; then
  TASK_TYPE="AceTrigger"
  TRAIN_FILE=$DATA_DIR/ar_train.jhu.better-split-80.json
  VALID_FILE=$DATA_DIR/ar_dev.jhu.better-split-80.json
  TEST_FILE=$DATA_DIR/ar_test.jhu.better-split-80.json
  if [ $MODEL_NAME == "xlm-roberta-large" ]; then
      MODEL_TYPE='xlmr'
      TRAIN_EPOCHS=10
  fi
elif [ $TASK == "en-minion" ] || [ $TASK == "ko-minion" ]; then
  TASK_TYPE="TriggerClassificationMinionTask"
  TRAIN_FILE=$DATA_DIR/train.json
  VALID_FILE=$DATA_DIR/dev.json
  TEST_FILE=$DATA_DIR/test.json
  if [ $MODEL_NAME == "xlm-roberta-large" ]; then
      MODEL_TYPE='xlmr'
      TRAIN_EPOCHS=20
      LEARNING_RATE=5e-6
      MAX_SEQ_LENGTH=256
  elif [ $MODEL_NAME == "xlm-roberta-base" ]; then
      MODEL_TYPE='xlmr'
      TRAIN_EPOCHS=50
      LEARNING_RATE=5e-5
      MAX_SEQ_LENGTH=256
  fi

fi

if [ "$STRATEGY" == "first_token" ] || [ "$STRATEGY" == "last_token" ] || [ "$STRATEGY" == "average" ]; then

  if [ "$#" -ne 5 ]; then
    echo "Usage: $0 task strategy mode-name expt-dir seed"
    exit 1
  fi

  python run_token_classification.py \
    --task_type ${TASK_TYPE} \
    --model_name_or_path ${MODEL_NAME} \
    --model_type ${MODEL_TYPE} \
    --train_file ${TRAIN_FILE} \
    --valid_file ${VALID_FILE} \
    --test_file ${TEST_FILE} \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --valid_batch_size ${BATCH_SIZE} \
    --train_batch_size ${BATCH_SIZE} \
    --num_train_epochs ${TRAIN_EPOCHS} \
    --do_predict \
    --do_train \
    --learning_rate ${LEARNING_RATE} \
    --output_dir ${EXPT_DIR}/${TASK}_${TRAIN_LANG}_${STRATEGY}_${MODEL_NAME}_${SEED} \
    --seed ${SEED} \
    --pooling_strategy ${STRATEGY}
fi

if [ "$STRATEGY" == "idf" ]; then

  if [ "$#" -ne 8 ]; then
    echo "Usage: $0 task strategy mode-name expt-dir seed temperature default-score score-file"
    exit 1
  fi

  TEMPERATURE=$6
  DEFAULT_SCORE=$7
  SCORES_FILE=$8
  python run_token_classification.py \
      --task_type ${TASK_TYPE} \
      --model_name_or_path ${MODEL_NAME} \
      --model_type ${MODEL_TYPE} \
      --train_file ${TRAIN_FILE} \
      --valid_file ${VALID_FILE} \
      --test_file ${TEST_FILE} \
      --max_seq_length ${MAX_SEQ_LENGTH} \
      --valid_batch_size ${BATCH_SIZE} \
      --train_batch_size ${BATCH_SIZE} \
      --num_train_epochs ${TRAIN_EPOCHS} \
      --do_predict \
      --do_train \
      --learning_rate ${LEARNING_RATE} \
      --output_dir ${EXPT_DIR}/${TASK}_${TRAIN_LANG}_${STRATEGY}_${MODEL_NAME}_${SEED} \
      --seed ${SEED} \
      --pooling_strategy ${STRATEGY} \
      --token_scores_temperature ${TEMPERATURE} \
      --default_token_score ${DEFAULT_SCORE} \
      --token_scores_file ${SCORES_FILE}
fi

if [ "$STRATEGY" == "attention" ]; then
  if [ "$#" -ne 6 ]; then
    echo "Usage: $0 task strategy mode-name expt-dir seed temperature"
    exit 1
  fi

  TEMPERATURE=$6
  python run_token_classification.py \
        --task_type ${TASK_TYPE} \
        --model_name_or_path ${MODEL_NAME} \
        --model_type ${MODEL_TYPE} \
        --train_file ${TRAIN_FILE} \
        --valid_file ${VALID_FILE} \
        --test_file ${TEST_FILE} \
        --max_seq_length ${MAX_SEQ_LENGTH} \
        --valid_batch_size ${BATCH_SIZE} \
        --train_batch_size ${BATCH_SIZE} \
        --num_train_epochs ${TRAIN_EPOCHS} \
        --do_predict \
        --do_train \
        --learning_rate ${LEARNING_RATE} \
        --output_dir ${EXPT_DIR}/${TASK}_${TRAIN_LANG}_${STRATEGY}_${MODEL_NAME}_${SEED} \
        --seed ${SEED} \
        --pooling_strategy ${STRATEGY} \
        --token_scores_temperature ${TEMPERATURE}
fi
