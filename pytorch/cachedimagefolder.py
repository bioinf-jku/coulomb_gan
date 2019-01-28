import numpy as np
import torch
import pathlib
from PIL import Image
import torch.utils.data


class CachedImageFolder(torch.utils.data.Dataset):
    '''Similar to torchvision.datasets.ImageFolder, but for single folders.'''
    def __init__(self, filename, transform=None, dummy_label=False):

        self.data = np.load(filename, mmap_mode='r')
        self.transform = transform
        self.dummy_label = dummy_label

    def create_cache(self, filepath, output_name):
        path = pathlib.Path(filepath)
        imgs = list(path.glob('*.jpg')) + list(path.glob('*.png'))
        if len(imgs) == 0:
            raise(RuntimeError("Found *.jpg oder *.png in " + root))
        imgs = list(path.glob('*.jpg')) + list(path.glob('*.png'))
        data = np.empty([len(imgs), 64, 64, 3], dtype=np.uint8)
        for i, f in enumerate(imgs):
            img = Image.open(f)
            data[i] = np.array(img.convert('RGB'))
            img.close()
        np.save(output_name, allow_pickle=False)

    def __getitem__(self, index):
        img = np.array(self.data[index]) # convert from mmap to array
        if self.transform is not None:
            img = self.transform(img)
        if self.dummy_label:
            return img, 0
        else:
            return img

    def __len__(self):
        return len(self.data)
