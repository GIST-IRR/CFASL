import numpy as np
import os
import torch
from src.seed import manual_seed
from dataset.utils import DisenDataLoader
from torchvision.transforms import ToTensor
import PIL
from tqdm import tqdm

# needs 3D Shapes image files not h5 file.
TRANSFORM = ToTensor()


class Fast_3DshapeDataLoader(DisenDataLoader):
    def __init__(self, path, shuffle_dataset=True, random_seed=42, split_ratio=0.0):
        super(Fast_3DshapeDataLoader, self).__init__(
            path, shuffle_dataset=True, random_seed=42, split_ratio=0.0
        )

        # def __call__(self, *args, **kwargs):
        IMGS = "imgs"
        self.img_dir = os.path.join(self.path, IMGS)
        self.data = [
            os.path.join(self.img_dir, dir) for dir in sorted(os.listdir(self.img_dir))
        ]
        self.latents_classes = np.load(os.path.join(self.path, "labels.npy"))
        self.factor_num = self.latents_classes.shape[-1]
        self.factor_dict = {0: 10, 1: 10, 2: 10, 3: 8, 4: 4, 5: 15}

    def __getitem__(self, idx):
        X = PIL.Image.open(self.data[idx])
        X = np.array(X) / 255.0
        X = np.transpose(X, (2, 0, 1))
        X = torch.Tensor(X)
        # X = TRANSFORM(X) # include normalization

        # data = torch.Tensor(self.data[idx])
        classes = torch.Tensor(self.latents_classes[idx])
        return X, classes

    def __len__(self):
        return len(self.data)

    def random_sampling_for_disen_global_variance(self, batch_size, replace=False):
        manual_seed(self.random_seed)
        g = np.random.Generator(np.random.PCG64(seed=np.random.randint(0, 2**32)))
        indices = g.choice(self.__len__(), batch_size, replace=replace)
        # return torch.Tensor(self.data[indices,:,:,:])#.permute(0,3,1,2)
        resized_imgs = []
        for idx in indices:
            resized_imgs.append(self.__getitem__(idx)[0])  # [3, 64, 64]

        return torch.stack(resized_imgs, dim=0)  # [batch, 3, 64, 64]

    def sampling_factors_and_img(self, batch_size, num_train):
        dataset_size = len(self.data)
        idxs = list(range(dataset_size))
        factors, imgs = [], []
        manual_seed(self.random_seed)
        for i in tqdm(range(num_train)):
            np.random.shuffle(idxs)
            factor_idxs = idxs[:batch_size]
            factors.append(
                torch.Tensor(self.latents_classes[factor_idxs, :])
            )  # (B, num factors)
            # imgs.append(torch.Tensor(self.data[factor_idxs, :, :, :])) # (B, C, H, W)
            # for resize
            resized_imgs = []
            for idx in factor_idxs:
                resized_imgs.append(self.__getitem__(idx)[0])

            resized_imgs = torch.stack(resized_imgs, dim=0)  # [batch, 1, 64, 64]
            imgs.append(resized_imgs)

        return torch.stack(imgs, dim=0), torch.stack(
            factors, dim=0
        )  # (num_train, B, C, H, W), (num_train, B, -1)

    def img_from_idx(self, idx):
        return self.data[idx]

    def factor_from_idx(self, idx):
        return self.latents_classes[idx]
