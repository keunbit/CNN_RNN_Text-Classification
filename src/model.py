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
    def __init__(self, args):
        super(NlpCNN, self).__init__()
        self.pretrained_embedding = args.pretrained_embedding
        self.freeze_embedding = args.freeze_embedding
        self.vocab_size = args.vocab_size
        self.filter_window_size = args.filter_window_size
        self.filter_features_dim = args.filter_features_dim
        self.embed_dim = args.embed_dim
        self.num_labels = args.num_labels
        self.pad_token_id = args.pad_token_id
        self.dropout = nn.Dropout(p=args.dropout)

        if self.pretrained_embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(
                embeddings=self.pretrained_embedding, freeze=self.freeze_embedding
            )
        else:
            self.embedding = nn.Embedding(
                num_embeddings=self.vocab_size,
                embedding_dim=self.embed_dim,
                padding_idx=self.pad_token_id,
                max_norm=0.5,
            )

        self.conv1d_0 = nn.Conv1d(
            in_channels=self.embed_dim, out_channels=self.filter_features_dim[0], kernel_size=self.filter_window_size[0]
        )
        self.conv1d_1 = nn.Conv1d(
            in_channels=self.embed_dim, out_channels=self.filter_features_dim[1], kernel_size=self.filter_window_size[1]
        )
        self.conv1d_2 = nn.Conv1d(
            in_channels=self.embed_dim, out_channels=self.filter_features_dim[2], kernel_size=self.filter_window_size[2]
        )
        self.fc = nn.Linear(in_features=np.sum(self.filter_features_dim), out_features=self.num_labels)

    def forward(self, input_ids, labels=None, **kwargs):
        # Output shape: (b, max_seq_len, embed_dim) -> (b, embed_dim, max_seq_len)
        # transpose(1,2) or permute(0,2,1)
        x_embed = self.embedding(input_ids).transpose(1, 2)

        # Output shape: (b, filter_features_dim, affected by window_size)
        out_0 = self.dropout(F.relu(self.conv1d_0(x_embed)))
        out_1 = self.dropout(F.relu(self.conv1d_1(x_embed)))
        out_2 = self.dropout(F.relu(self.conv1d_2(x_embed)))

        # Output shape: (b, self.filter_features_dim, 1)
        out_0 = F.max_pool1d(out_0, out_0.shape[2])
        out_1 = F.max_pool1d(out_1, out_1.shape[2])
        out_2 = F.max_pool1d(out_2, out_2.shape[2])

        # Concatenate pooled out_0, out_1, out_2
        # Output shape: (b, 3*100)
        out = torch.cat((out_0, out_1, out_2), dim=1)
        out = out.view(out.size(0), -1)
        logits = self.fc(self.dropout(out))

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return_dict = {
            "loss": loss,
            "logits": logits,
            "labels": labels,
        }

        return return_dict


class NlpRNN(nn.Module):
    def __init__(self, args):
        super(NlpRNN, self).__init__()
        self.pretrained_embedding = args.pretrained_embedding
        self.freeze_embedding = args.freeze_embedding
        self.vocab_size = args.vocab_size
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.embed_dim = args.embed_dim
        self.dropout = args.dropout
        self.num_labels = args.num_labels
        self.pad_token_id = args.pad_token_id
        self.model_type = args.model_type
        self.pretrained_embedding = args.pretrained_embedding
        self.bidirectional = args.bidirectional
        self.dropout = nn.Dropout(p=self.dropout)

        if self.pretrained_embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(
                embeddings=self.pretrained_embedding, freeze=self.freeze_embedding
            )
        else:
            self.embedding = nn.Embedding(
                num_embeddings=self.vocab_size,
                embedding_dim=self.embed_dim,
                padding_idx=self.pad_token_id,
                max_norm=0.5,
            )

        if self.model_type == "rnn":
            self.rnn = nn.RNN(
                input_size=self.embed_dim, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True
            )

        else:
            if self.model_type == "lstm":
                self.rnn = nn.LSTM(
                    input_size=self.embed_dim,
                    hidden_size=self.hidden_size,
                    num_layers=self.num_layers,
                    batch_first=True,
                    bidirectional=self.bidirectional,
                )

            elif self.model_type == "gru":
                self.rnn = nn.GRU(
                    input_size=self.embed_dim,
                    hidden_size=self.hidden_size,
                    num_layers=self.num_layers,
                    batch_first=True,
                    bidirectional=self.bidirectional,
                )
            else:
                raise ValueError("Only support models are RNN, GRU, LSTM. use 'rnn','lstm','gru' ")

        if self.bidirectional:
            self.last_fc_dim = self.hidden_size * 2
        else:
            self.last_fc_dim = self.hidden_size
        self.fc = nn.Linear(self.last_fc_dim, self.num_labels)

    def forward(self, input_ids, labels=None, seq_len=None):

        # Output Shape: (b, max_seq_len, embed_dim) -> (b, embed_dim, max_seq_len)
        x_embed = self.embedding(input_ids)

        packed_input = nn.utils.rnn.pack_padded_sequence(
            input=x_embed, lengths=seq_len.cpu().numpy(), batch_first=True, enforce_sorted=False
        )
        # sorted_output
        packed_output, h_n = self.rnn(packed_input)
        # only use last step's hidden layer
        # output, _ = nn.utils.rnn.pad_packed_sequence(sequence=packed_output, batch_first=True)

        # LSTM
        if self.rnn._get_name() == "LSTM":
            # only get hidden, not cell
            h_n = h_n[0]
            if self.bidirectional:
                last_hidden_fc = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
            else:
                last_hidden_fc = h_n.squeeze()
        # GRU
        elif self.rnn._get_name() == "GRU":
            if self.bidirectional:
                # concat bidirectional hidden
                # (2, 32, 128) -> (32, 256)
                last_hidden_fc = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
            else:
                # (1, 32, 128) -> (32, 128)
                last_hidden_fc = h_n.squeeze()
        # RNN
        else:
            last_hidden_fc = h_n.squeeze()

        logits = self.fc(self.dropout(last_hidden_fc))

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return_dict = {
            "loss": loss,
            "logits": logits,
            "labels": labels,
        }

        return return_dict
