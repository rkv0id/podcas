from torch.utils.data import Dataset


class TextDataset(Dataset):
    """A simple Dataset class to wrap text data for DataLoader."""
    def __init__(self, texts: list[str]):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int):
        return self.texts[idx]
