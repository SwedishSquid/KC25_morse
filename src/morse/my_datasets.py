import torch
from pathlib import Path


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, features: list, labels: list, transform = lambda x: x):
        self.features = features
        self.labels = labels
        assert len(features) == len(labels)
        self.transform = transform
        pass

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        return self.transform(self.features[index]), self.labels[index]



def load_tensors(dir_path, filenames):
    for name in filenames:
        path = Path(dir_path, name)
        yield torch.load(path, weights_only=True)



def filenames_to_torch(filenames):
    for name in filenames:
        yield Path(name).with_suffix('.pt')

