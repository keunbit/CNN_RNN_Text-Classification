import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def load_pretrained_vector(vocab, model_path):
    with open(model_path, "r", encoding="utf-8", errors="ignore") as f:
        size_, vector_dim = map(int, f.readline().split())
        load_vector_progress_bar = tqdm(
            range(size_),
            leave=True,
            dynamic_ncols=True,
            desc="Load Pretrained Vecotrs",
            smoothing=0,
        )

        # Initialize random embeddings
        embeddings = np.random.uniform(-0.25, 0.25, (len(vocab), vector_dim))
        embeddings[vocab["[PAD]"]] = np.zeros(vector_dim)

        # Load pretrained vectors
        count = 0
        for line in f:
            line_split = line.rstrip().split()
            word = line_split[0]
            if word in vocab:
                count += 1
                embeddings[vocab[word]] = np.array(line_split[1:])
            load_vector_progress_bar.update(1)

    return torch.FloatTensor(embeddings), count


class NlpCNN(nn.Module):
    def __init__(
        self,
        pretrained_embedding=None,
        freeze_embedding=False,
        vocab_size=None,
        filter_window_size=[3, 4, 5],
        num_filter_features=[100, 100, 100],
        embed_dim=300,
        dropout=0.5,
        num_labels=2,
        pad_token_id=0,
    ):
        super(NlpCNN, self).__init__()

        if pretrained_embedding is not None:
            self.vocab_size, self.embed_dim = pretrained_embedding.size()
            self.embedding = nn.Embedding.from_pretrained(embeddings=pretrained_embedding, freeze=freeze_embedding)
        else:
            self.embed_dim = embed_dim
            self.embedding = nn.Embedding(
                num_embeddings=vocab_size, embedding_dim=self.embed_dim, padding_idx=pad_token_id, max_norm=0.5
            )

        self.conv1d_0 = nn.Conv1d(
            in_channels=self.embed_dim, out_channels=num_filter_features[0], kernel_size=filter_window_size[0]
        )
        self.conv1d_1 = nn.Conv1d(
            in_channels=self.embed_dim, out_channels=num_filter_features[1], kernel_size=filter_window_size[1]
        )
        self.conv1d_2 = nn.Conv1d(
            in_channels=self.embed_dim, out_channels=num_filter_features[2], kernel_size=filter_window_size[2]
        )
        self.fc = nn.Linear(in_features=np.sum(num_filter_features), out_features=num_labels)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input_ids):
        # Output shape: (b, max_seq_len, embed_dim) -> (b, embed_dim, max_seq_len)
        # transpose(1,2) or permute(0,2,1)
        x_embed = self.embedding(input_ids).transpose(1, 2)

        # Output shape: (b, num_filter_features, affected by window_size)
        out_0 = self.dropout(F.relu(self.conv1d_0(x_embed)))
        out_1 = self.dropout(F.relu(self.conv1d_1(x_embed)))
        out_2 = self.dropout(F.relu(self.conv1d_2(x_embed)))

        # Output shape: (b, num_filter_features, 1)
        out_0 = F.max_pool1d(out_0, out_0.shape[2])
        out_1 = F.max_pool1d(out_1, out_1.shape[2])
        out_2 = F.max_pool1d(out_2, out_2.shape[2])

        # Concatenate pooled out_0, out_1, out_2
        # Output shape: (b, 3*100)
        out = torch.cat((out_0, out_1, out_2), dim=1)
        out = out.view(out.size(0), -1)
        logits = self.fc(self.dropout(out))
        # Output shape: (b, 2)
        return logits
