import os

from torch.utils.data import DataLoader

from src.nsmc_dataset import NSMCDataset, encoding_dataset, load_nsmc_data, make_vocab

from .testing_utils import get_tests_dir


def test_dataset():

    SAMPLE_FILE = os.path.join(get_tests_dir("sample_data"), "test_data.txt")

    nsmc_data = load_nsmc_data(SAMPLE_FILE)
    vocab = make_vocab(nsmc_data["document"], vocab_size=50000)
    encoded_train = encoding_dataset(dataset=nsmc_data, vocab=vocab, max_seq_len=16)
    dataset = NSMCDataset(encoded_dataset=encoded_train)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    print(f"batch_size is {len(dataloader)}")
