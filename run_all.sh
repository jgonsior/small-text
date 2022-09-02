#!/bin/sh
NUM_ITER=3
BATCH_SIZE=20
EXP_NAME=test

for DATASET in trec6 ag_news subj rotten imdb; do
    python test.py --num_iterations $NUM_ITER --batch_size $BATCH_SIZE --exp_name $EXP_NAME --dataset $DATASET
done