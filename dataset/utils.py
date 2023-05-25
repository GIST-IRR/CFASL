import torch.utils.data as data

class DisenDataLoader(data.Dataset):
    def __init__(self,
                 path,
                 shuffle_dataset=True,
                 random_seed=42,
                 split_ratio = 0.2):
        self.path = path
        self.shuffle_dataset = shuffle_dataset
        self.random_seed = random_seed
        self.split_ratio = split_ratio

        self.data, self.latents_values, self.latents_classes = None, None, None
        self.train_idxs, self.test_idxs = None, None
        self.factor_num = None

    def __getitem__(self, item):
        raise NotImplementedError("Build getitem function")

    def dataset_sample_batch(self, num_samples: int, mode: str, replace: bool):
        raise NotImplementedError("Build dataset_sample_batch function")

    def __len__(self):
        return len(self.data)