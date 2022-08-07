#!/bin/bash

MODEL_FILE="cc.ko.300.vec"
MODEL_DIR="models"

# make data directory
if [ ! -d $MODEL_DIR ]; then
  echo "there is no models directory, making $MODEL_DIR directory, so make 'models' directory"
  mkdir models
else
  echo "'models' directory exist"
fi

# Determine OS
os=$(uname)
if [[ ! $os == "Linux" ]] && [[ ! $os == "Darwin" ]]; then
  echo "This script does not support this OS."
  exit 0
fi

if [ ! -e ./$MODEL_DIR/$MODEL_FILE ]; then
  if [ "$os" == "Linux" ]; then
    apt-get install wget gzip
  elif [ "$os" == "Darwin" ]; then
    brew install wget gzip
  fi
  wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/$MODEL_FILE.gz -P ./models
  gzip -d ./models/$MODEL_FILE.gz
else
  echo pretrained_model already exist!
fi
