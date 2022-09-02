#!/bin/sh
NUM_ITER=50
BATCH_SIZE=20
EXP_NAME=baseline
RANDOM_SEED=42

for QUERY_STRATEGY in LC Rand Ent MM; do
    for RANDOM_SEED in 42 43 44 45 46; do
        for DATASET in trec6 ag_news subj rotten imdb; do
            python test.py --num_iterations $NUM_ITER --batch_size $BATCH_SIZE --exp_name $EXP_NAME --dataset $DATASET --random_seed $RANDOM_SEED --query_strategy $QUERY_STRATEGY
        done
    done
done