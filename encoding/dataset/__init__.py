from .Kvasir_data import *


datasets = {
    'kvasir': KvasirDataset,
}


def get_segmentation_dataset(name, **kwargs):
    return datasets[name.lower()](**kwargs)
