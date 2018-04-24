#!/usr/bin/env python3

import os
import numpy as np
import pathlib
import scipy as sp
from scipy.misc import imread

import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as tvt
from torchvision.datasets.folder import default_loader
from torch.autograd import Variable
from torch.utils.data import DataLoader



def get_fid_network():
    net = models.inception_v3( pretrained=True)
    net.train(False)
    net.fc = torch.nn.Sequential()  # identity function
    net.transform_input=False
    return net


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.

    Returns:
    -- FID  : The Frechet Distance.
    -- mean : The squared norm of the difference of the means: ||mu_1 - mu_2||^2
    -- trace: The trace-part of the FID: Tr(C_1 + C_2 - 2*sqrt(C_1*C_2))
    """
    diff = mu1 - mu2
    eps = 1e-8

    # product might be almost singular
    covmean, _ = sp.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    offset = np.eye(sigma1.shape[0]) * eps
    covmean = sp.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real
    tr_covmean = np.trace(covmean)

    dist = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return dist


class SingleImageFolder(torch.utils.data.Dataset):
    '''Similar to torchvision.datasets.ImageFolder, but for single folders.'''
    def __init__(self, root, transform=None, loader=default_loader, dummy_label=False):
        path = pathlib.Path(root)
        imgs = list(path.glob('*.jpg')) + list(path.glob('*.png'))
        if len(imgs) == 0:
            raise(RuntimeError("Found *.jpg oder *.png in " + root))
        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.loader = loader
        self.dummy_label=dummy_label

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(str(path))
        if self.transform is not None:
            img = self.transform(img)
        if self.dummy_label:
            return img, 0
        else:
            return img

    def __len__(self):
        return len(self.imgs)


def calculate_statistics(dataset_with_labels, gpu_id=-1, n_threads=2):
    ''' Excepts dataset to be images with values in the range [0, 1]. '''
    pm = True if gpu_id > -1 else False
    data_loader = DataLoader(dataset_with_labels, batch_size=64, shuffle=False, num_workers=n_threads, pin_memory=pm)
    net = get_fid_network()
    mean = torch.Tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = torch.Tensor([0.229, 0.224, 0.225])[None, :, None, None]
    if gpu_id > -1:
        net = net.cuda(gpu_id)
        mean = mean.cuda(gpu_id)
        std = std.cuda(gpu_id)

    act = torch.FloatTensor(len(data_loader.dataset), 2048)
    i = 0
    for x, _ in data_loader:
        if gpu_id > -1:
            x = x.cuda(gpu_id)
        x -= mean
        x /= std
        x = Variable(x, volatile=True) # the following line would create Variable anyways
        x = F.upsample(x, 299, mode='bilinear')
        a = net(x)
        n = x.size()[0]
        act[i:i+n] = a.cpu().data
        i += n
    act = act.numpy()
    mu = np.mean(act, axis=0, dtype=np.float64)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def _handle_path(path, options):
    if path.endswith('.npz'):
        f = np.load(path)
        m, s = f['mu'][:], f['sigma'][:]
        f.close()
    else:
        ds = SingleImageFolder(path, transform=tvt.ToTensor(), dummy_label=True)
        m, s = calculate_statistics(ds, options.gpu, options.threads)
    return m, s


def calculate_fid_given_paths(paths, options):
    ''' Calculates the FID of two paths. '''

    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError("Invalid path: %s" % p)

    m1, s1 = _handle_path(paths[0], options)
    m2, s2 = _handle_path(paths[1], options)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value


if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("path", type=str, nargs=2, help='Path to the generated images or to .npz statistic files')
    parser.add_argument("--gpu", type=int, help='GPU to use (use -1 for CPU only)', default=-1)
    parser.add_argument("--threads", type=int, help='number of input threads', default=2)
    args = parser.parse_args()
    fid_value = calculate_fid_given_paths(args.path, args)
    print("FID: ", fid_value)
