import argparse
import logging
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import NlpCNN, load_pretrained_vector
from nsmc_dataset import NSMCDataset, encoding_dataset, load_nsmc_data, make_vocab
from tokenizer.tokenizer_processor import (
    BasicTokenizer,
    HannanumTokenizer,
    KkmaTokenizer,
    KomoranTokenizer,
    MecabTokenizer,
    OktTokenizer,
)
from train_utils import get_ckpt_version, save_model, save_score

logger = logging.getLogger("looger")
stream_handler = logging.StreamHandler()
logger.addHandler(stream_handler)
logger.setLevel(logging.DEBUG)

TOKENIZER_CLASS_MAPPER = {
    "basic": BasicTokenizer,
    "mecab": MecabTokenizer,
    "hannanum": HannanumTokenizer,
    "kkma": KkmaTokenizer,
    "komoran": KomoranTokenizer,
    "okt": OktTokenizer,
}

OPTIMIZER_MAPPER = {
    "adadelta": optim.Adadelta,
    "adagrad": optim.Adagrad,
    "sgd": optim.SGD,
    "adam": optim.Adam,
    "adamw": optim.AdamW,
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log_args(args: argparse.Namespace):
    args_dict = vars(args)
    max_len = max([len(k) for k in args_dict.keys()])
    fmt_string = "\t%" + str(max_len) + "s : %s"
    logger.info("Arguments:")
    for key, value in args_dict.items():
        logger.info(fmt_string, key, value)


def parse_args():
    parser = argparse.ArgumentParser(description="Training a CNN model on a text classification NSMC task")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        required=True,
        help="The path of the dataset to use.",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        required=True,
        help="A .txt file containing the training data.",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
        required=True,
        help="A .txt file containing the test data.",
    )
    parser.add_argument(
        "--tokenizer_type",
        type=str,
        default="basic",
        choices=["basic", "mecab", "hannanum", "kkma", "komoran", "okt"],
        help="Choice Tokenizer [Mecab, Hannanum, Kkma, Komoran, Okt], default is space split tokens",
    )
    parser.add_argument(
        "--use_pretrained",
        action="store_true",
        help="If passed, will use fasttext pretrained word vector.",
    )
    parser.add_argument(
        "--freeze_embedding",
        action="store_true",
        help="If passed, will not update embedding.",
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        help="The path of fasttext pretrained model.",
    )
    parser.add_argument(
        "--pretrained_file",
        type=str,
        default="cc.ko.300.vec",
        help="A .vec file pretrained file",
    )
    parser.add_argument("--pad_token_id", type=int, default=0, help="if not use pretrained, should define pad_token_id")
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=None,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded"
        ),
    )
    parser.add_argument(
        "--max_seq_option",
        type=str,
        default="mean",
        choices=["mean", "median"],
        help="If max_seq_len is None, use mean of texts tokens length. Option",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=30000,
        help=("The vocab size, contain ([PAD], [UNK]) tokens"),
    )
    parser.add_argument(
        "--optimizer_type",
        type=str,
        default="adadelta",
        choices=["adadelta", "adagrad", "sgd", "adam", "adamw"],
        help="Optimizer to use",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="Initial learning rate to use.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=32,
        help="Batch size for the training dataloader.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=64,
        help="Batch size for the evaluation dataloader.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=5,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--val_check_interval", default=0.25, type=float, help="Check validation set X times during a training epoch"
    )
    parser.add_argument(
        "--drop_out",
        default=0.5,
        type=float,
        help="The number of model dropout",
    )
    parser.add_argument(
        "--filter_size",
        default="3,4,5",
        type=str,
        help="The list of filter_size(kernel_size) CNN model",
    )
    parser.add_argument(
        "--patience",
        default=5,
        type=int,
        help="The number of validation epochs with no improvement after which training will be stopped.",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")

    args = parser.parse_args()

    # Sanity checks
    if args.use_pretrained:
        if args.pretrained_path is None or args.pretrained_file is None:
            raise ValueError("Need both pretrained path and pretrained file.")
        else:
            extension = args.pretrained_file.split(".")[-1]
            assert extension in ["vec"], "`pretrain_file` should be a vec file."

    if args.data_dir is None or args.train_file is None or args.test_file is None:
        raise ValueError("Need data path & training/test file.")

    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["txt"], "`train_file` should be a txt file."
        if args.test_file is not None:
            extension = args.test_file.split(".")[-1]
            assert extension in ["txt"], "`test_file` should be a txt file."

    # convert type [str -> int]
    args.filter_size = [int(size_) for size_ in args.filter_size.split(",")]
    args.tokenizer_type = args.tokenizer_type.lower()
    args.optimizer_type = args.optimizer_type.lower()

    assert args.val_check_interval >= 0

    return args


def main():
    args = parse_args()
    # log argments
    log_args(args)

    if args.seed is not None:
        set_seed(args.seed)

    if not os.path.exists(f"{args.output_dir}"):
        os.makedirs(f"{args.output_dir}")
    # check ckpt_version
    ckpt_version = get_ckpt_version(args)

    train_data = os.path.join(args.data_dir, args.train_file)
    test_data = os.path.join(args.data_dir, args.test_file)
    args.tokenizer = TOKENIZER_CLASS_MAPPER[args.tokenizer_type]()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = load_nsmc_data(train_data)
    test_data = load_nsmc_data(test_data)

    logger.info("***** Tokenizing *****")
    logger.info(f"  Tokenizer Type = {args.tokenizer.type_}")
    vocab, seq_len = make_vocab(train_data["document"], vocab_size=args.vocab_size, tokenizer=args.tokenizer)
    logger.info(f"  Datasets tokens length(mean) = {seq_len['mean']}")
    logger.info(f"  Datasets tokens length(median) = {seq_len['median']}")

    if args.max_seq_len is None:
        args.max_seq_len = seq_len[args.max_seq_option]
        logger.info(f"  If max_seq_len = None; We set max_seq_len = {seq_len['mean']}")
    logger.info(f"  max_seq_len = {args.max_seq_len}")

    if args.use_pretrained:
        pretraiend_model_path = os.path.join(args.pretrained_path, args.pretrained_file)
        pretrained_embedding, count = load_pretrained_vector(vocab=vocab, model_path=pretraiend_model_path)
        cnn_model = NlpCNN(
            pretrained_embedding=pretrained_embedding,
            freeze_embedding=args.freeze_embedding,
            vocab_size=args.vocab_size,
        )
        logger.info(f"  There are {count} / {len(vocab)} pretrained vectors found.")
    else:
        cnn_model = NlpCNN(vocab_size=args.vocab_size, pad_token_id=args.pad_token_id)

    cnn_model.to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = OPTIMIZER_MAPPER[args.optimizer_type](cnn_model.parameters(), lr=args.learning_rate)

    # prepare dataset & dataloader
    encoded_train = encoding_dataset(
        dataset=train_data, vocab=vocab, max_seq_len=args.max_seq_len, device=args.device, tokenizer=args.tokenizer
    )
    encoded_test = encoding_dataset(
        dataset=test_data, vocab=vocab, max_seq_len=args.max_seq_len, device=args.device, tokenizer=args.tokenizer
    )
    train_dataset = NSMCDataset(encoded_dataset=encoded_train)
    test_dataset = NSMCDataset(encoded_dataset=encoded_test)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False)

    # Total Step
    train_steps_per_epoch = len(train_dataloader)
    total_training_steps = args.num_train_epochs * train_steps_per_epoch
    eval_step = math.ceil(train_steps_per_epoch * args.val_check_interval)

    # Only show the progress bar once on each machine.
    train_progress_bar = tqdm(
        range(total_training_steps),
        leave=True,
        dynamic_ncols=True,
        desc="Training",
        smoothing=0,
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num train examples = {len(train_dataset)}")
    logger.info(f"  Num test examples = {len(test_dataset) if test_dataset else len(test_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous train batch size = {args.train_batch_size}")
    logger.info(f"  Instantaneous eval batch size = {args.eval_batch_size}")
    logger.info(f"  Total optimization steps = {total_training_steps}")

    completed_steps = 0
    best_score = 0.0
    cur_patience = 0

    for _ in range(args.num_train_epochs):
        for train_step, batch in enumerate(train_dataloader):
            cnn_model.train()
            b_intput_ids, b_labels = batch
            outs = cnn_model(b_intput_ids)
            loss = criterion(outs, b_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            completed_steps += 1

            train_progress_bar.update(1)
            # NOTE Eval on every `val_check_interval`
            if (train_step + 1) % eval_step == 0 or train_step == train_steps_per_epoch - 1:
                valid_progress_bar = tqdm(
                    range(len(test_dataloader)),
                    dynamic_ncols=True,
                    leave=False,
                    desc="Test",
                )
                # evaluate
                batch_accumulate_score = 0
                cnn_model.eval()
                for batch in test_dataloader:
                    with torch.no_grad():
                        b_intput_ids, b_labels = batch
                        outs = cnn_model(b_intput_ids)

                        loss = criterion(outs, b_labels)
                        preds = torch.argmax(outs, 1).flatten()

                        accuracy = (preds == b_labels).cpu().numpy().mean()
                        batch_accumulate_score += accuracy
                        valid_progress_bar.update(1)

                # eval score result
                eval_step_score = round(batch_accumulate_score / len(test_dataloader), 4)
                logger.info(f"*** Test Result ({completed_steps} step) ***")
                logger.info(f"  test/accuracy = {eval_step_score*100}%")

                # write score
                cur_epochs = round(completed_steps / train_steps_per_epoch, 3)
                save_score(
                    args, ckpt_version=ckpt_version, step=train_step, score=eval_step_score, cur_epochs=cur_epochs
                )
                # check cur_score is best & Early Stopping
                if best_score < eval_step_score:
                    # cur socre is best
                    best_score = eval_step_score
                    cur_patience = 0
                    save_model(args, ckpt_version=ckpt_version, ckpt_model=cnn_model)
                else:
                    cur_patience += 1
                    logger.info(
                        f"test/accuracy was not in top 1 (best score: {best_score}% / cur patience: {cur_patience})"
                    )

                    if cur_patience >= args.patience:
                        # Stop training
                        logger.info(f"Reached all patience {cur_patience}. So stop training!")
                        break

        if cur_patience >= args.patience:
            break


if __name__ == "__main__":
    main()
