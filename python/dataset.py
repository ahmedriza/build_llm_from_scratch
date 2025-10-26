import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

# The GPTDatasetV1 class is based on the PyTorch Dataset class and defines
# how individual rows are fetched from the dataset, where each row consists
# of a number of token IDs (based on a max_length) assigned to an 
# `input_chunk` tensor.
# The `target_chunk` tensor contains the corresponding targets.
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokeniser, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokeniser.encode(txt)

        print(f"Total number of tokens in the book: {len(token_ids)}")

        # Use a sliding window to chunk the book into overlapping 
        # sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

        print(f"Total number of input chunks created: {len(self.input_ids)}")
        print(f"Total number of target chunks created: {len(self.target_ids)}")

    def __len__(self):
        """return the total number of rows in the dataset"""
        l = len(self.input_ids)
        print(f"Dataset length: {l}")
        return l

    def __getitem__(self, idx):
        """Return a single row from the dataset"""
        print(f"Fetching item at index: {idx}")
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, batch_size = 4, max_length = 256,
                         stride = 128, shuffle = True, drop_last = True,
                         num_workers = 0):
    """
    drop_last = True: drops the last batch if it is shorter than the 
                      specified batch_size to prevent loss spikes during 
                      training.
    num_workers: specifies the number of CPU processes to use for preprocessing.
    """
    tokeniser = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokeniser, max_length, stride)
    dataloader = DataLoader(
            dataset, 
            batch_size = batch_size,
            shuffle = shuffle, 
            drop_last = drop_last,
            num_workers = num_workers
            )
    return dataloader

if __name__ == "__main__":
    is_mps_available = torch.backends.mps.is_available()
    print("MPS available: ", is_mps_available)

    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_txt = f.read()

    dataloader = create_dataloader_v1(
            raw_txt, 
            batch_size = 1, 
            max_length = 4, 
            stride = 1, 
            shuffle = False, 
            )

    # convert dataloader to a Python iterator to fetch the next entry via 
    # Python's built-in `next()` function
    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    print("first batch: ", first_batch)

    second_batch = next(data_iter)
    print("second batch: ", second_batch)

