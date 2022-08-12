#!/bin/bash

python3 ./src/trainer.py \
  --model_type cnn \
  --data_dir ./data \
  --train_file ratings_train.txt \
  --test_file ratings_test.txt \
  --tokenizer_type mecab \
  --pretrained_path ./models \
  --pretrained_file cc.ko.300.vec \
  --pad_token_id 0 \
  --max_seq_len 128 \
  --vocab_size 3000 \
  --optimizer_type adam \
  --learning_rate 0.001 \
  --seed 42 \
  --train_batch_size 32 \
  --eval_batch_size 64 \
  --num_train_epochs 50 \
  --val_check_interval 0.25 \
  --dropout 0.5 \
  --patience 5 \
  --output_dir ./outs_cnn \
  --filter_window_size 3,4,5 \
  --filter_features_dim 100,100,100 \
  --use_pretrained
