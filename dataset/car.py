import os
import numpy as np
import PIL
from PIL import Image
import scipy.io as sio
from six.moves import range
from sklearn.utils import extmath
import torch
from dataset.utils import DisenDataLoader
from src.seed import manual_seed
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
class StateSpaceAtomIndex(object):
    """Index mapping from features to positions of state space atoms."""

    def __init__(self, factor_sizes, features):
        """Creates the StateSpaceAtomIndex.
    Args:
      factor_sizes: List of integers with the number of distinct values for each
        of the factors.
      features: Numpy matrix where each row contains a different factor
        configuration. The matrix needs to cover the whole state space.
    """
        self.factor_sizes = factor_sizes
        num_total_atoms = np.prod(self.factor_sizes)
        self.factor_bases = num_total_atoms / np.cumprod(self.factor_sizes)
        feature_state_space_index = self._features_to_state_space_index(features)
        if np.unique(feature_state_space_index).size != num_total_atoms:
            raise ValueError("Features matrix does not cover the whole state space.")
        lookup_table = np.zeros(num_total_atoms, dtype=np.int64)
        lookup_table[feature_state_space_index] = np.arange(num_total_atoms)
        self.state_space_to_save_space_index = lookup_table

    def features_to_index(self, features):
        """Returns the indices in the input space for given factor configurations.
    Args:
      features: Numpy matrix where each row contains a different factor
        configuration for which the indices in the input space should be
        returned.
    """
        state_space_index = self._features_to_state_space_index(features)
        return self.state_space_to_save_space_index[state_space_index]

    def _features_to_state_space_index(self, features):
        """Returns the indices in the atom space for given factor configurations.
    Args:
      features: Numpy matrix where each row contains a different factor
        configuration for which the indices in the atom space should be
        returned.
    """
        if (np.any(features > np.expand_dims(self.factor_sizes, 0)) or
                np.any(features < 0)):
            raise ValueError("Feature indices have to be within [0, factor_size-1]!")
        return np.array(np.dot(features, self.factor_bases), dtype=np.int64)


class _3DcarDataLoader(DisenDataLoader):
    def __init__(self,
                 path,
                 shuffle_dataset=True,
                 random_seed=42,
                 split_ratio=0.0):
        super(_3DcarDataLoader, self).__init__(path,
                                               shuffle_dataset=True,
                                               random_seed=42,
                                               split_ratio=0.0)
    #def __call__(self, *args, **kwargs):
        self.factor_sizes = [4, 24, 183]
        features = extmath.cartesian([np.array(list(range(i))) for i in self.factor_sizes])
        self.latent_factor_indices = [0, 1, 2]
        self.factor_num = features.shape[1]
        self.index = StateSpaceAtomIndex(self.factor_sizes, features)
        self.data_shape = [64, 64, 3]
        self.data, self.latents_classes = self._load_data() # numpy array (17568, 64, 64, 3)
        self.data = self.data.transpose(0,3,1,2) # numpy array (17568, 3, 64, 64)
        self.factor_dict = {0:4, 1:24, 2:183}

    def __getitem__(self, idx):
        data = torch.Tensor(self.data[idx])
        classes = torch.Tensor(self.latents_classes[idx])
        return data, classes

    def random_sampling_for_disen_global_variance(self, batch_size, replace=False):
        manual_seed(self.random_seed)
        g = np.random.Generator(np.random.PCG64(seed=np.random.randint(0, 2 ** 32)))
        indices = g.choice(self.data.shape[0], batch_size, replace=replace)
        return torch.Tensor(self.data[indices,:,:,:])#.permute(0,3,1,2)

    def sampling_factors_and_img(self, batch_size, num_train):
        dataset_size = len(self.data)
        idxs = list(range(dataset_size))
        factors, imgs = [], []
        manual_seed(self.random_seed)
        for i in tqdm(range(num_train)):
            np.random.shuffle(idxs)
            factor_idxs = idxs[:batch_size]
            factors.append(torch.Tensor(self.latents_classes[factor_idxs, :])) #(B, num factors)
            imgs.append(torch.Tensor(self.data[factor_idxs, :, :, :])) # (B, C, H, W)

        return torch.stack(imgs, dim=0), torch.stack(factors, dim=0) # (num_train, B, C, H, W), (num_train, B, -1)

    def img_from_idx(self, idx):
        return self.data[idx]

    def factor_from_idx(self, idx):
        return self.latents_classes[idx]

    def _load_data(self):
        dataset = np.zeros((24 * 4 * 183, 64, 64, 3))
        factors = np.zeros((24 * 4 * 183, 3))
        all_files = [x for x in os.listdir(self.path) if ".mat" in x]
        for i, filename in enumerate(all_files):
            data_mesh = self._load_mesh(filename)
            factor1 = np.array(list(range(4)))
            factor2 = np.array(list(range(24)))
            all_factors = np.transpose([
                np.tile(factor1, len(factor2)),
                np.repeat(factor2, len(factor1)),
                np.tile(i,
                        len(factor1) * len(factor2))
            ])
            indexes = self.index.features_to_index(all_factors)
            dataset[indexes] = data_mesh
            factors[indexes] = all_factors
        return dataset, factors

    def _load_mesh(self, filename):
        """Parses a single source file and rescales contained images."""
        with open(os.path.join(self.path, filename), "rb") as f:
            mesh = np.einsum("abcde->deabc", sio.loadmat(f)["im"])
        flattened_mesh = mesh.reshape((-1,) + mesh.shape[2:])
        rescaled_mesh = np.zeros((flattened_mesh.shape[0], 64, 64, 3))
        for i in range(flattened_mesh.shape[0]):
            pic = Image.fromarray(flattened_mesh[i, :, :, :])
            pic.thumbnail((64, 64), Image.ANTIALIAS)
            rescaled_mesh[i, :, :, :] = np.array(pic)
        return rescaled_mesh * 1. / 255