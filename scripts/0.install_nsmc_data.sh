#!/bin/bash

TRAIN_DATA="ratings_train.txt"
TEST_DATA="ratings_test.txt"
DATA_DIR="data"

# make data directory
if [ ! -d "$DATA_DIR" ]; then
  echo "there is no directory, making $DATA_DIR directory";
  mkdir data
fi
  echo "'data' directory exist"

# nsmc data 다운로드
if [ ! -e ./"$DATA_DIR"/"$TRAIN_DATA" ] || [ ! -e ./"$DATA_DIR"/"$TEST_DATA" ]; then
  wget https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt -P ./data
  wget https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt -P ./data
else
  echo dataset already exist!
fi
