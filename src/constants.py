from torch.optim import SGD, Adam

from models.decoder import CNN2DShapesDecoder, CNN3DShapesDecoder
from models.encoder import CNN2DShapesEncoder, CNN3DShapesEncoder

from dataset.dsprites import dstripeDataLoader
from dataset.car import _3DcarDataLoader
from dataset.smallnorb import SmallNORBDataLoader
from dataset.shapes3d import Fast_3DshapeDataLoader
from dataset.celebA import CelebADataLoader

DATA_CLASSES = {
    'dsprites': (CNN2DShapesEncoder, CNN2DShapesDecoder),
    'shapes3d': (CNN3DShapesEncoder, CNN3DShapesDecoder),
    'car': (CNN3DShapesEncoder, CNN3DShapesDecoder),
    'smallnorb': (CNN2DShapesEncoder, CNN2DShapesDecoder),
    'celeba': (CNN3DShapesEncoder, CNN3DShapesDecoder),
    'cdsprites': (CNN3DShapesEncoder, CNN3DShapesDecoder),
    'grid': (CNN2DShapesEncoder, CNN2DShapesDecoder),
}

DATA_HIDDEN_DIM = {
    'dsprites' : [256, 128],
    'shapes3d' : [256, 256],
    'car' : [256, 256],
    'smallnorb' : [256, 256],
    'celeba': [256, 256],
    'cdsprites' : [256, 256],
    'grid' : [256, 128],
}

DATA_STEPS= {
    'dsprites' : 300000,
    'shapes3d' : 500000,
    'car' : 300000,
    'smallnorb' : 500000,
    'celeba': 1000000,
    'cdsprites' : 600000,
    'grid' : 324,
}

DATALOADER = {
    'dsprites' : dstripeDataLoader,
    'shapes3d' : Fast_3DshapeDataLoader,
    'car' : _3DcarDataLoader,
    'smallnorb' : SmallNORBDataLoader,
    'celeba': CelebADataLoader,
}


OPTIMIZER = {
    'sgd': SGD,
    'adam': Adam,
}