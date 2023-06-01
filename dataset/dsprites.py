import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data.dataloader import default_collate

from src.seed import manual_seed
from dataset.utils import DisenDataLoader
from tqdm import tqdm


class dstripeDataLoader(DisenDataLoader):
    def __init__(self, path, shuffle_dataset=True, random_seed=42, split_ratio=0.0):
        super(dstripeDataLoader, self).__init__(
            path, shuffle_dataset, random_seed, split_ratio
        )

        # def __call__(self, *args, **kwargs):
        np_load_old = np.load
        np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
        np.load = np_load_old
        dataset_zip = np.load(self.path, allow_pickle=True, encoding="bytes")

        self.data = np.expand_dims(dataset_zip["imgs"], axis=1)
        self.latents_values = dataset_zip["latents_values"]
        self.latents_classes = dataset_zip["latents_classes"][:, 1:]
        self.factor_num = self.latents_values[:, 1:].shape[-1]
        self.factor_dict = {0: 3, 1: 6, 2: 40, 3: 32, 4: 32}
        assert self.factor_num == 5

    def random_sampling_for_disen_global_variance(self, batch_size, replace=False):
        manual_seed(self.random_seed)
        g = np.random.Generator(np.random.PCG64(seed=np.random.randint(0, 2**32)))
        indices = g.choice(len(self.data), batch_size, replace=replace)
        return torch.Tensor(self.data[indices])

    def sampling_factors_and_img(self, batch_size, num_train):
        dataset_size = len(self.data)
        idxs = list(range(dataset_size))
        factors, imgs = [], []
        manual_seed(self.random_seed)
        for i in tqdm(range(num_train)):
            np.random.shuffle(idxs)
            factor_idxs = idxs[:batch_size]
            factors.append(torch.Tensor(self.latents_classes[factor_idxs]))
            imgs.append(torch.Tensor(self.data[factor_idxs]))

        return torch.stack(imgs, dim=0), torch.stack(factors, dim=0)

    def __getitem__(self, idx):
        data = torch.Tensor(self.data[idx])
        classes = torch.Tensor(self.latents_classes[idx])
        return data, classes

    def __len__(self):
        return len(self.data)

    def img_from_idx(self, idx):
        return self.data[idx]

    def factor_from_idx(self, idx):
        return self.latents_classes[idx]

    def idx_from_factor(self, factor):
        base = np.concatenate(
            self.latents_values[1:][::-1].cumprod()[::-1][1:],
            np.array(
                [
                    1,
                ]
            ),
        )
        return np.dot(factor, base).astype(int)

    def dataset_sample_batch(self, num_samples, mode, replace=False):
        g = np.random.Generator(np.random.PCG64(seed=np.random.randint(0, 2**32)))
        indices = g.choice(len(self), num_samples, replace=replace)
        return self.dataset_batch_from_indices(indices, mode=mode)

    def dataset_batch_from_indices(self, indices, mode):
        return default_collate([self.dataset_get(idx, mode=mode) for idx in indices])

    def dataset_get(self, idx, mode: str):
        try:
            idx = int(idx)
        except:
            raise TypeError(f"Indices must be integer-like ({type(idx)}): {idx}")
