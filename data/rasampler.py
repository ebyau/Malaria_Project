import torch
import math

class RASampler(torch.utils.data.Sampler):
    """Sampler that restricts data loading to a subset of the dataset,
    with repeated augmentation.
    """
    def __init__(self, dataset, shuffle=True):
        self.dataset = dataset
        self.shuffle = shuffle
        self.epoch = 0
        self.num_samples = len(self.dataset) * 3
        self.total_size = self.num_samples
        self.num_selected_samples = len(self.dataset)

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices = [ele for ele in indices for i in range(3)]
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[:self.num_selected_samples]
        assert len(indices) == self.num_selected_samples

        return iter(indices)

    def __len__(self):
        return self.num_selected_samples

    def set_epoch(self, epoch):
        self.epoch = epoch