from __future__ import print_function
import os
import errno
import struct

import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
from torchvision.datasets.utils import download_url, check_integrity

from dataset.utils import DisenDataLoader
from torchvision.transforms import ToTensor
from tqdm import tqdm
from src.seed import manual_seed
import pdb
TRANSFORM = ToTensor()

class SmallNORBDataLoader(DisenDataLoader):
    """`MNIST <https://cs.nyu.edu/~ylclab/data/norb-v1.0-small//>`_ Dataset.
        Args:
            root (string): Root directory of dataset where processed folder and
                and  raw folder exist.
            train (bool, optional): If True, creates dataset from the training files,
                otherwise from the test files.
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If the dataset is already processed, it is not processed
                and downloaded again. If dataset is only already downloaded, it is not
                downloaded again.
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
            info_transform (callable, optional): A function/transform that takes in the
                info and transforms it.
            mode (string, optional): Denotes how the images in the data files are returned. Possible values:
                - all (default): both left and right are included separately.
                - stereo: left and right images are included as corresponding pairs.
                - left: only the left images are included.
                - right: only the right images are included.
        """

    dataset_root = "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/"
    data_files = {
        'train': {
            'dat': {
                "name": 'smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat',
                "md5_gz": "66054832f9accfe74a0f4c36a75bc0a2",
                "md5": "8138a0902307b32dfa0025a36dfa45ec"
            },
            'info': {
                "name": 'smallnorb-5x46789x9x18x6x2x96x96-training-info.mat',
                "md5_gz": "51dee1210a742582ff607dfd94e332e3",
                "md5": "19faee774120001fc7e17980d6960451"
            },
            'cat': {
                "name": 'smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat',
                "md5_gz": "23c8b86101fbf0904a000b43d3ed2fd9",
                "md5": "fd5120d3f770ad57ebe620eb61a0b633"
            },
        },
        'test': {
            'dat': {
                "name": 'smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat',
                "md5_gz": "e4ad715691ed5a3a5f138751a4ceb071",
                "md5": "e9920b7f7b2869a8f1a12e945b2c166c"
            },
            'info': {
                "name": 'smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat',
                "md5_gz": "a9454f3864d7fd4bb3ea7fc3eb84924e",
                "md5": "7c5b871cc69dcadec1bf6a18141f5edc"
            },
            'cat': {
                "name": 'smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat',
                "md5_gz": "5aa791cd7e6016cf957ce9bdb93b8603",
                "md5": "fd5120d3f770ad57ebe620eb61a0b633"
            },
        },
    }

    raw_folder = 'raw'
    processed_folder = 'processed'
    train_image_file = 'train_img'
    train_label_file = 'train_label'
    train_info_file = 'train_info'
    test_image_file = 'test_img'
    test_label_file = 'test_label'
    test_info_file = 'test_info'
    extension = '.pt'


    def __init__(self,
                 path,
                 download=False,
                 mode='left',
                 shuffle_dataset=True,
                 random_seed=42,
                 split_ratio=0.0
                 ):
        super(SmallNORBDataLoader, self).__init__(path,
                                                 shuffle_dataset=True,
                                                 random_seed=42,
                                                 split_ratio=0.0)

        self.root = os.path.expanduser(path)
        self.train = True  # training set or test set
        self.mode = mode

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found or corrupted.' + 'You can use download=True to download it')

        # load test or train set
        image_file = self.train_image_file if self.train else self.test_image_file
        label_file = self.train_label_file if self.train else self.test_label_file
        info_file = self.train_info_file if self.train else self.test_info_file

        # load labels
        self.labels = self._load(label_file).numpy()

        # load info files
        if self.mode == "left" or self.mode == "right":
            self.latents_classes = self._load(info_file).numpy()

            self.latents_classes[:, 2] = (self.latents_classes[:, 2]/2).astype(int)
            a = {4:0, 6:1, 7:2, 8:3, 9:4}
            label_0 = np.copy(self.latents_classes[:,0])
            for k, v in a.items():
                label_0[self.latents_classes[:,0] == k] = v
            self.latents_classes[:,0] = label_0

        else:
            # add left and right info as [0, 1], where 0: left, 1: right
            latents_classes = self._load(info_file).numpy()
            left, right = np.zeros((latents_classes.shape[0], 1)), np.ones((latents_classes.shape[0], 1))
            left_classes = np.concatenate((left, latents_classes), axis=1)
            right_classes = np.concatenate((right, latents_classes), axis=1)
            self.latents_classes = np.concatenate((left_classes, right_classes), axis=0)

        # load right set
        if self.mode == "left":
            self.data = self._load("{}_left".format(image_file)).unsqueeze(1).numpy()

        # load left set
        elif self.mode == "right":
            self.data = self._load("{}_right".format(image_file)).unsqueeze(1).numpy()

        elif self.mode == "all" or self.mode == "stereo":
            left_data = self._load("{}_left".format(image_file))
            right_data = self._load("{}_right".format(image_file))

            # load stereo
            if self.mode == "stereo":
                self.data = torch.stack((left_data, right_data), dim=1).unsqueeze(1).numpy()

            # load all
            else:
                self.data = torch.cat((left_data, right_data), dim=0).unsqueeze(1).numpy() #[dataset_size, 1, 96,96]

        self.factor_num = 5 if self.mode == "all" else 4
        self.factor_dict = {0:5, 1:9, 2:18, 3:6}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            mode ``all'', ``left'', ``right'':
                tuple: (image, target, info)
            mode ``stereo'':
                tuple: (image left, image right, target, info)
        """

        target = self.labels[index % 24300] if self.mode == "all" else self.labels[index]

        info = self.latents_classes[index % 24300] if self.mode == "all" else self.latents_classes[index]

        if self.mode == "stereo":
            img_left = self._transform(self.data[index, 0])
            img_right = self._transform(self.data[index, 1])
            return img_left, img_right, target, info

        img = self._transform(self.data[index]) # include normalize
        return img, info

    def __len__(self):
        return len(self.data)

    def _transform(self, img):
        # doing this so that it is consistent with all other data sets
        # to return a PIL Image
        img = np.squeeze(img)
        img = Image.fromarray(img, mode='L')

        img = img.resize(size=(64, 64))


        img = TRANSFORM(img) # for resize #TRANSFORM(img.resize(size=(64, 64)))
        return img  # .resize(size=(64, 64))

    def _load(self, file_name):
        return torch.load(os.path.join(self.root, self.processed_folder, file_name + self.extension))

    def _save(self, file, file_name):
        with open(os.path.join(self.root, self.processed_folder, file_name + self.extension), 'wb') as f:
            torch.save(file, f)

    ############################################# 수정 필요함 #################################################
    def random_sampling_for_disen_global_variance(self, batch_size, replace=False):
        manual_seed(self.random_seed)
        g = np.random.Generator(np.random.PCG64(seed=np.random.randint(0, 2 ** 32)))
        indices = g.choice(self.data.shape[0], batch_size, replace=replace)#g.choice(self.data.shape[0], batch_size, replace=replace)
        #return torch.from_numpy(self.data[indices,:,:,:]) * 1. / 255 #torch.Tensor(self.data[indices, :, :, :])  # .permute(0,3,1,2)

        # for resize
        resized_imgs = []
        for idx in indices:
            resized_imgs.append(self.__getitem__(idx)[0]) #[1, 64, 64]

        return torch.stack(resized_imgs, dim=0) # [batch, 1, 64, 64]

    def sampling_factors_and_img(self, batch_size, num_train):
        dataset_size = self.data.shape[0]#len(self.data)
        idxs = list(range(dataset_size))
        factors, imgs = [], []
        manual_seed(self.random_seed)
        for i in tqdm(range(num_train)):
            np.random.shuffle(idxs)
            factor_idxs = idxs[:batch_size]
            factors.append(torch.from_numpy(self.latents_classes[factor_idxs, :])) #factors.append(torch.Tensor(self.latents_classes[factor_idxs, :])) #(B, num factors)
            #imgs.append(torch.from_numpy(self.data[factor_idxs, :, :, :])* 1. / 255) #imgs.append(torch.Tensor(self.data[factor_idxs, :, :, :])) # (B, C, H, W)

            #for resize
            resized_imgs = []
            for idx in factor_idxs:
                resized_imgs.append(self.__getitem__(idx)[0])

            resized_imgs = torch.stack(resized_imgs, dim=0) # [batch, 1, 64, 64]
            imgs.append(resized_imgs)


        return torch.stack(imgs, dim=0), torch.stack(factors, dim=0) # (num_train, B, C, H, W), (num_train, B, -1)

    def img_from_idx(self, idx):
        return self.data[idx]

    def factor_from_idx(self, idx):
        return self.latents_classes[idx]

    ############################################# ########### #################################################


    # for download from url

    def _check_exists(self):
        """ Check if processed files exists."""
        files = (
            "{}_left".format(self.train_image_file),
            "{}_right".format(self.train_image_file),
            "{}_left".format(self.test_image_file),
            "{}_right".format(self.test_image_file),
            self.test_label_file,
            self.train_label_file
        )
        fpaths = [os.path.exists(os.path.join(self.root, self.processed_folder, f + self.extension)) for f in files]
        return False not in fpaths

    def _flat_data_files(self):
        return [j for i in self.data_files.values() for j in list(i.values())]

    def _check_integrity(self):
        """Check if unpacked files have correct md5 sum."""
        root = self.root
        for file_dict in self._flat_data_files():
            filename = file_dict["name"]
            md5 = file_dict["md5"]
            fpath = os.path.join(root, self.raw_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        """Download the SmallNORB data if it doesn't exist in processed_folder already."""
        import gzip

        if self._check_exists():
            return

        # check if already extracted and verified
        if self._check_integrity():
            print('Files already downloaded and verified')
        else:
            # download and extract
            for file_dict in self._flat_data_files():
                url = self.dataset_root + file_dict["name"] + '.gz'
                filename = file_dict["name"]
                gz_filename = filename + '.gz'
                md5 = file_dict["md5_gz"]
                fpath = os.path.join(self.root, self.raw_folder, filename)
                gz_fpath = fpath + '.gz'

                # download if compressed file not exists and verified
                download_url(url, os.path.join(self.root, self.raw_folder), gz_filename, md5)

                print('# Extracting data {}\n'.format(filename))

                with open(fpath, 'wb') as out_f, \
                        gzip.GzipFile(gz_fpath) as zip_f:
                    out_f.write(zip_f.read())

                os.unlink(gz_fpath)

        # process and save as torch files
        print('Processing...')

        # create processed folder
        try:
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        # read train files
        left_train_img, right_train_img = self._read_image_file(self.data_files["train"]["dat"]["name"])
        train_info = self._read_info_file(self.data_files["train"]["info"]["name"])
        train_label = self._read_label_file(self.data_files["train"]["cat"]["name"])

        # read test files
        left_test_img, right_test_img = self._read_image_file(self.data_files["test"]["dat"]["name"])
        test_info = self._read_info_file(self.data_files["test"]["info"]["name"])
        test_label = self._read_label_file(self.data_files["test"]["cat"]["name"])

        # save training files
        self._save(left_train_img, "{}_left".format(self.train_image_file))
        self._save(right_train_img, "{}_right".format(self.train_image_file))
        self._save(train_label, self.train_label_file)
        self._save(train_info, self.train_info_file)

        # save test files
        self._save(left_test_img, "{}_left".format(self.test_image_file))
        self._save(right_test_img, "{}_right".format(self.test_image_file))
        self._save(test_label, self.test_label_file)
        self._save(test_info, self.test_info_file)

        print('Done!')

    @staticmethod
    def _parse_header(file_pointer):
        # Read magic number and ignore
        struct.unpack('<BBBB', file_pointer.read(4))  # '<' is little endian)

        # Read dimensions
        dimensions = []
        num_dims, = struct.unpack('<i', file_pointer.read(4))  # '<' is little endian)
        for _ in range(num_dims):
            dimensions.extend(struct.unpack('<i', file_pointer.read(4)))

        return dimensions

    def _read_image_file(self, file_name):
        fpath = os.path.join(self.root, self.raw_folder, file_name)
        with open(fpath, mode='rb') as f:
            dimensions = self._parse_header(f)
            assert dimensions == [24300, 2, 96, 96]
            num_samples, _, height, width = dimensions

            left_samples = np.zeros(shape=(num_samples, height, width), dtype=np.uint8)
            right_samples = np.zeros(shape=(num_samples, height, width), dtype=np.uint8)

            for i in range(num_samples):
                # left and right images stored in pairs, left first
                left_samples[i, :, :] = self._read_image(f, height, width)
                right_samples[i, :, :] = self._read_image(f, height, width)

        return torch.ByteTensor(left_samples), torch.ByteTensor(right_samples)

    @staticmethod
    def _read_image(file_pointer, height, width):
        """Read raw image data and restore shape as appropriate. """
        image = struct.unpack('<' + height * width * 'B', file_pointer.read(height * width))
        image = np.uint8(np.reshape(image, newshape=(height, width)))
        return image

    def _read_label_file(self, file_name):
        fpath = os.path.join(self.root, self.raw_folder, file_name)
        with open(fpath, mode='rb') as f:
            dimensions = self._parse_header(f)
            assert dimensions == [24300]
            num_samples = dimensions[0]

            struct.unpack('<BBBB', f.read(4))  # ignore this integer
            struct.unpack('<BBBB', f.read(4))  # ignore this integer

            labels = np.zeros(shape=num_samples, dtype=np.int32)
            for i in range(num_samples):
                category, = struct.unpack('<i', f.read(4))
                labels[i] = category
            return torch.LongTensor(labels)

    def _read_info_file(self, file_name):
        fpath = os.path.join(self.root, self.raw_folder, file_name)
        with open(fpath, mode='rb') as f:

            dimensions = self._parse_header(f)
            assert dimensions == [24300, 4]
            num_samples, num_info = dimensions

            struct.unpack('<BBBB', f.read(4))  # ignore this integer

            infos = np.zeros(shape=(num_samples, num_info), dtype=np.int32)

            for r in range(num_samples):
                for c in range(num_info):
                    info, = struct.unpack('<i', f.read(4))
                    infos[r, c] = info

        return torch.LongTensor(infos)