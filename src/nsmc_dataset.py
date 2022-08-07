import math
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def load_nsmc_data(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        # id, document, label
        col_name = f.readline().strip().split("\t")
        nsmc_data = {k: [] for k in col_name}
        for line in f:
            line = line.strip().split("\t")
            nsmc_data[col_name[0]].append(line[0])
            nsmc_data[col_name[1]].append(line[1])
            nsmc_data[col_name[2]].append(line[2])

    return nsmc_data


def make_vocab(texts: list, vocab_size: int, tokenizer):

    vocab_progress_bar = tqdm(
        range(len(texts)),
        leave=True,
        dynamic_ncols=True,
        desc="Tokenizing",
        smoothing=0,
    )
    nsmc_texts = []
    nsmc_seq_len = []
    for text in texts:
        words, seq_len = tokenizer.tokenize(text)
        nsmc_texts.extend(words)
        nsmc_seq_len.append(seq_len)
        vocab_progress_bar.update(1)

    nsmc_seq_len = np.array(nsmc_seq_len, dtype=np.int16)
    nsmc_seq_len_mean = np.mean(nsmc_seq_len)
    nsmc_seq_len_median = np.median(nsmc_seq_len)

    seq_len = {"mean": math.ceil(nsmc_seq_len_mean), "median": math.ceil(nsmc_seq_len_median)}

    vocab = {}
    vocab["[PAD]"] = 0
    vocab["[UNK]"] = 1
    nsmc_counter = Counter(nsmc_texts).most_common()
    for i, (word, _) in enumerate(nsmc_counter[: vocab_size - 2]):
        vocab[word] = i + 2

    return vocab, seq_len


def encoding_dataset(dataset: dict, vocab: dict, max_seq_len: int = 16, device=None, tokenizer=None):
    encoded_dataset = {}
    encoded_texts = []
    encoded_labels = []
    for sentence, label in zip(dataset["document"], dataset["label"]):
        encoded_sent = []
        tokens, _ = tokenizer.tokenize(sentence)
        for word in tokens:
            if word in vocab:
                encoded_sent.append(vocab[word])
            else:
                encoded_sent.append(vocab["[UNK]"])

        # check length & add ["pad"] token or trim
        if len(encoded_sent) >= max_seq_len:
            encoded_sent = encoded_sent[:max_seq_len]
        else:
            pad_length = max_seq_len - len(encoded_sent)
            encoded_sent = encoded_sent + ([vocab["[PAD]"]] * pad_length)

        # check sentence is max_length
        assert len(encoded_sent) == max_seq_len

        encoded_texts.append(encoded_sent)
        encoded_labels.append(int(label))

    encoded_dataset["document"] = torch.IntTensor(encoded_texts).to(device)
    encoded_dataset["label"] = torch.LongTensor(encoded_labels).to(
        device
    )  # 학습때 nn.CrossEntropyLoss 계산시 LongTensor 타입(int64) 필요

    return encoded_dataset


class NSMCDataset(Dataset):
    def __init__(self, encoded_dataset):
        super(NSMCDataset, self).__init__()
        self.texts = encoded_dataset["document"]
        self.labels = encoded_dataset["label"]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return self.texts[index], self.labels[index]
