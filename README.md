# CNN, RNN Text-Classification NSMC

## Description

- **Convolutional Neural Networks(CNN) & Recurrent Neural Network(RNN)** 을 이용한 NSMC 텍스트 분류기
- `Pytorch`만 사용해서 모델 구현

## Dataset

### 1. NSMC(Naver Sentiment movie corpus) Data

- `ratings_train.txt` : 150k
- `ratings_test.txt` : 50k
- [Data description (NSMC github)](https://github.com/e9t/nsmc)

## Setup

### 1. python

- Highly recommend to use **Anaconda**

```bash
# OSX no GPU
conda install pytorch==1.12.0 -c pytorch

# Linux no GPU
conda install pytorch==1.12.0 cpuonly -c pytorch

# Linux with GPU
conda install pytorch==1.12.0 cudatoolkit=11.3 -c pytorch
```

Reference: [Pytorch Install](https://pytorch.org/get-started/locally/)

### 2. Downlaod Data & Model

```bash
# Install package
pip install -r requirements.txt

# Install nsmc data
sh ./scripts/0.install_nsmc_data.sh

# Install Mecab
sh ./scripts/1.install_mecab.sh

# Install Fasttext pretrained model
sh ./scripts/2.install_fasttext_ko.sh
```

## How to Run

```bash
# you can run CNN & RNN
sh run_cnn.sh
# or
sh run_rnn.sh

# arguments
python3 ./src/trainer.py -h

  Training CNN, RNN model on a text classification NSMC task

  optional arguments:

    --data_dir DATA_DIR   
                          The path of the dataset to use.
    --train_file TRAIN_FILE
                          A .txt file containing the training data.
    --test_file TEST_FILE
                          A .txt file containing the test data.
    --tokenizer_type {basic,mecab,hannanum,kkma,komoran,okt}
                          Choice Tokenizer [Mecab, Hannanum, Kkma, Komoran, Okt], default is space split tokens
    --use_pretrained      
                          If passed, will use fasttext pretrained word vector.
    --freeze_embedding    
                          If passed, will not update embedding.
    --pretrained_path PRETRAINED_PATH
                          The path of fasttext pretrained model.
    --pretrained_file PRETRAINED_FILE
                          A .vec file pretrained file
    --pad_token_id PAD_TOKEN_ID
                          if not use pretrained, should define pad_token_id
    --max_seq_len MAX_SEQ_LEN
                          The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded
    --max_seq_option {mean,median}
                          If max_seq_len is None, use mean of texts tokens length. Option
    --vocab_size VOCAB_SIZE
                          The vocab size, contain ([PAD], [UNK]) tokens
    --optimizer_type {adadelta,adagrad,sgd,adam,adamw}
                          Optimizer to use
    --learning_rate LEARNING_RATE
                          Initial learning rate to use.
    --seed SEED           
                          A seed for reproducible training.
    --train_batch_size TRAIN_BATCH_SIZE
                          Batch size for the training dataloader.
    --eval_batch_size EVAL_BATCH_SIZE
                          Batch size for the evaluation dataloader.
    --num_train_epochs NUM_TRAIN_EPOCHS
                          Total number of training epochs to perform.
    --val_check_interval VAL_CHECK_INTERVAL
                          Check validation set X times during a training epoch
    --dropout DROPOUT     
                          The number of model dropout
    --filter_window_size FILTER_WINDOW_SIZE
                          The list of filter_size(kernel_size) CNN model
    --filter_features_dim FILTER_FEATURES_DIM
                          The dimension of each filter_window(kernel) CNN model
    --patience PATIENCE   
                          The number of validation epochs with no improvement after which training will be stopped.
    --output_dir OUTPUT_DIR
                          Where to store the final model.
    --model_type {cnn,rnn,lstm,gru}
                          Choice Model for training [CNN, RNN, LSTM, GRU]
    --bidirectional       
                          If passed, will LSTM & GRU parameter use bidirectional.
    --hidden_size HIDDEN_SIZE
                          Hidden size for RNN model (simple_RNN, LSTM, BiLSTM)
    --num_layers NUM_LAYERS
                          Depth of layers RNN model (simple_RNN, LSTM, BiLSTM)
```

## Evaluation Result

### 1. Parameters

- `tokenizer_type` = Mecab
- `train_batch_size` = 32
- `max_seq_len` = 128
- `vocab_size` = 30,000
- `optimizer_type` = Adam
- `learning_rate` = 0.001
- `dropout` = 0.5
- `filter_size` = [3,4,5]
- `Bidirectional` = True (LSTM, GRU)

### 2. Best Score & Parameters

|             | Static<br />(Acc/epochs) | non-Static<br />(Acc/epochs) | Rand<br />(Acc/epochs) |
| :---------- | :----------------------: | :--------------------------: | :--------------------: |
| CNN         |      0.857 / (9.25)      |      **0.859 / (3.5)**       |     0.839 / (2.25)     |
| Vanilla RNN |       0.822 / (6)        |      **0.834 / (2.25)**      |     0.823 / (5.5)      |
| BiLSTM      |      0.866 / (4.5)       |       **0.869 / (4)**        |      0.860 / (8)       |
| GRU         |      0.864 / (4.75)      |       **0.868 / (4)**        |     0.858 / (6.75)     |

## Reference

- [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)
- [https://www.youtube.com/watch?v=r3Liq9B6cTo](https://www.youtube.com/watch?v=r3Liq9B6cTo)
