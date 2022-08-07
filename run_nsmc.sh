#!/bin/bash

python3 ./src/trainer.py \
  --data_dir ./data \
  --train_file ratings_train.txt \
  --test_file ratings_test.txt \
  --tokenizer_type mecab \
  --pretrained_path ./models \
  --pretrained_file cc.ko.300.vec \
  --max_seq_len 128 \
  --vocab_size 3000 \
  --optimizer_type adam \
  --learning_rate 0.001 \
  --seed 42 \
  --train_batch_size 32 \
  --eval_batch_size 64 \
  --num_train_epochs 50 \
  --output_dir ./outs \
  --drop_out 0.9 \
  --filter_size 3,4,5 \
  --patience 5 \
  --use_pretrained
