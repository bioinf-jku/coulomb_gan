#!/usr/bin/env python3

import os
import sys
import time
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from utils import startup_bookkeeping, plot_tiles, get_timestamp

import torch
import torchvision.datasets as dset
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as tvt
import torch.optim as optim

def print_info(msg, do_print, flush=True):
    if do_print:
        print(msg, flush=flush)


import fid
import models


def calculate_squared_distances(a, b):
    '''returns the squared distances between all elements in a and in b as a matrix
    of shape #a * #b'''
    na = a.data.shape[0]
    nb = b.data.shape[0]
    dim = a.data.shape[-1]
    a = a.view([na, 1, -1])
    b = b.view([1, nb, -1])
    d = a-b
    return (d*d).sum(2)


def plummer_kernel(a, b, dimension, epsilon):
    r = calculate_squared_distances(a, b)
    r += epsilon*epsilon
    f1 = dimension-2
    return torch.pow(r, -f1 / 2)


def get_potentials(x, y, dimension, cur_epsilon):
    '''
    This is alsmost the same `calculate_potential`, but
        px, py = get_potentials(x, y)
    is faster than:
        px = calculate_potential(x, y, x)
        py = calculate_potential(x, y, y)
    because we calculate the cross terms only once.
    '''
    x_fixed = x.detach()
    y_fixed = y.detach()
    nx = x.data.shape[0]
    ny = y.data.shape[0]
    pk_xx = plummer_kernel(x_fixed, x, dimension, cur_epsilon)
    pk_yx = plummer_kernel(y, x, dimension, cur_epsilon)
    pk_yy = plummer_kernel(y_fixed, y, dimension, cur_epsilon)
    #pk_xx.view(-1)[::pk_xx.size(1)+1] = 1.0
    #pk_yy.view(-1)[::pk_yy.size(1)+1] = 1.0
    #for i in range(nx):
    #    pk_xx[i, i] = 1.0
    #for i in range(ny):
    #    pk_yy[i, i] = 1.0
    kxx = pk_xx.sum(0) / nx
    kyx = pk_yx.sum(0) / ny
    kxy = pk_yx.sum(1) / nx
    kyy = pk_yy.sum(0) / ny
    pot_x = kxx - kyx
    pot_y = kxy - kyy
    return pot_x, pot_y


def mean_squared_error(x, y):
    d = (x - y)**2
    return d.mean()


def calculate_fid(generator, fid_net, mu_fid, sigma_fid, n_samples, batch_size, gpu_id):
    fid_net.transform_input=False
    mean = torch.Tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = torch.Tensor([0.229, 0.224, 0.225])[None, :, None, None]
    if gpu_id != -1:
        mean = mean.cuda()
        std = std.cuda()
    mean, std = Variable(mean, volatile=True), Variable(std, volatile=True)

    n_iter = (n_samples // batch_size)+1
    n = n_iter * batch_size
    zz = torch.FloatTensor(batch_size, generator.latentsize)
    act = torch.FloatTensor(n, 2048)
    for i in range(n_iter):
        zz.normal_()
        z = Variable(zz, volatile=True)
        if gpu_id != -1:
            z = z.cuda()
        x = generator(z)
        x = (torch.clamp(x, -1.0, +1.0) + 1.0) / 2.0
        x -= mean
        x /= std
        x = F.upsample(x, 299, mode='bilinear')
        a = fid_net(x)
        act[(i*batch_size):(i+1)*batch_size] = a.data.cpu()
    act = act.numpy()
    mu = np.mean(act, axis=0, dtype=np.float64)
    sigma = np.cov(act, rowvar=False)
    return fid.calculate_frechet_distance(mu, sigma, mu_fid, sigma_fid)


def run(dataset, generator_type, discriminator_type, latentsize, kernel_dimension, epsilon, learning_rate, batch_size, options, logdir_base='/tmp'):
    run_name = '_'.join(['%s' % get_timestamp('%H%M'),
        'g%s' % generator_type,
        'd%s' % discriminator_type,
        'z%d' % latentsize,
        'l%1.0e' % learning_rate,
        'l2p%1.0e' % options.l2_penalty,
        #'d%d' % kernel_dimension,
        #'eps%3.2f' % epsilon,
        'lds%1.e' % options.discriminator_lr_scale,
    ])
    run_name += ("_l2pscale%1.e" % options.gen_l2p_scale) if options.gen_l2p_scale != 1.0 else ''
    run_name += "_M" if options.remember_previous else ''
    run_name += ("_dl%s" % options.disc_loss) if options.disc_loss != 'l2' else ''
    run_name += ("_%s" % options.logdir_suffix) if options.logdir_suffix else ''
    run_name = run_name.replace('+', '')

    subdir = "%s_%s" % (get_timestamp('%y%m%d'), dataset)
    logdir = Path(logdir_base) / subdir / run_name
    print_info("\nLogdir: %s\n" % logdir, options.verbosity > 0)
    if __name__ == "__main__" and options.sample_images is None:
        startup_bookkeeping(logdir, __file__)
        trainlog = open(logdir / 'logfile.csv', 'w')
    else:
        trainlog = None

    m, s = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    if dataset == 'lsun_bedroom':
        img_shape = [3, 64, 64]
        data = dset.LSUN(db_path='/publicdata/image/lsun/', classes=['bedroom_train'],
            transform=tvt.Compose([tvt.Scale(64), tvt.CenterCrop(64), tvt.ToTensor(), tvt.Normalize(m, s)]))
    elif dataset == 'lsun_tower':
        img_shape = [3, 64, 64]
        data = dset.LSUN(db_path='/publicdata/image/lsun/', classes=['tower_train'],
            transform=tvt.Compose([tvt.Scale(64), tvt.CenterCrop(64), tvt.ToTensor(), tvt.Normalize(m, s)]))
    elif dataset == 'lsun_church_outdoor':
        img_shape = [3, 64, 64]
        data = dset.LSUN(db_path='/publicdata/image/lsun/', classes=['church_outdoor_train'],
            transform=tvt.Compose([tvt.Scale(64), tvt.CenterCrop(64), tvt.ToTensor(), tvt.Normalize(m, s)]))
    elif dataset == 'celeba':
        img_shape = [3, 64, 64]
        data = fid.SingleImageFolder('/publicdata/image/celebA_cropped/', dummy_label=True,
            transform=tvt.Compose([tvt.ToTensor(), tvt.Normalize(m, s)]))
    elif dataset == 'cifar10':
        img_shape = [3, 32, 32]
        data = dset.CIFAR10(root='/tmp', download=True,
            transform=tvt.Compose([tvt.ToTensor(), tvt.Normalize(m, s)]))
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=options.threads, pin_memory=True)

    fid_stats_file = options.fid_stats % dataset.lower()
    assert Path(fid_stats_file).exists(), "Can't find training set statistics for FID (%s)" % fid_stats_file
    f = np.load(fid_stats_file)
    mu_fid, sigma_fid = f['mu'][:], f['sigma'][:]
    f.close()
    fid_net = fid.get_fid_network()
    if options.gpu > -1:
        fid_net = fid_net.cuda()

    generator = models.GENERATORS[generator_type](latentsize, img_shape)
    discriminator = models.DISCRIMINATORS[discriminator_type](img_shape)

    if options.resume_checkpoint:
        generator.load_state_dict(torch.load(options.resume_checkpoint + '.generator'))
        discriminator.load_state_dict(torch.load(options.resume_checkpoint + '.discriminator'))

    optim_g = optim.Adam(generator.parameters(), learning_rate)
    optim_d = optim.Adam(discriminator.parameters(), learning_rate*options.discriminator_lr_scale, weight_decay=options.l2_penalty)
    if options.gpu != -1:
        generator.cuda()
        discriminator.cuda()

    zz = torch.FloatTensor(batch_size, latentsize)
    max_iter = int(options.iterations * 1000)

    cur_iter = 0
    t0 = time.time()
    while cur_iter < max_iter:
        for yy, _ in dataloader:

            zz.normal_()
            z = Variable(zz, requires_grad=False)
            if options.gpu != -1:
                z = z.cuda()

            x = generator(z)
            disc_x = discriminator(x)
            loss_g = torch.mean(disc_x)
            optim_g.zero_grad()
            loss_g.backward()
            optim_g.step()


            y = Variable(yy, requires_grad=False)
            if options.gpu != -1:
                y = y.cuda()
            x = x.detach()

            disc_x = discriminator(x)
            disc_y = discriminator(y)
            pot_x, pot_y = get_potentials(x, y, kernel_dimension, epsilon)

            loss_d_x = mean_squared_error(pot_x, disc_x)
            loss_d_y = mean_squared_error(pot_y, disc_y)
            loss_d = loss_d_x + loss_d_y

            #loss_g = mean_squared_error(disc_x)
            optim_d.zero_grad()
            loss_d.backward()
            optim_d.step()


            if (cur_iter > 0) and (cur_iter % options.checkpoint_every == 0):
                fn_prefix =  str(logdir / ('model-%d' % cur_iter))
                torch.save(generator.state_dict(), fn_prefix + '.generator')
                torch.save(discriminator.state_dict(), fn_prefix + '.discriminator')

            if cur_iter % options.stats_every == 0:
                fid_value = calculate_fid(generator, fid_net, mu_fid, sigma_fid, 5*1024, batch_size, options.gpu)
                #fid_value = -1

                xx_img = (torch.clamp(x, -1.0, +1.0) + 1.0) / 2.0
                xx_img = xx_img.cpu().data.numpy()
                s = (cur_iter, fid_value, time.time() - t0, dataset, run_name)
                if batch_size > 64:
                    fig = plot_tiles(xx_img, 10, 10, local_norm="none", figsize=(6.6, 6.6), data_format='NCWH')
                else:
                    fig = plot_tiles(xx_img, 8, 8, local_norm="none", figsize=(6.6, 6.6), data_format='NCWH')
                fig.savefig(str(logdir / ('%09d.png' % cur_iter)))
                plt.close(fig)
                if trainlog:
                    print(', '.join([str(ss) for ss in s]), file=trainlog, flush=True)
                print_info("%9d  %3.2f -- %3.2fs %s %s" % s, options.verbosity > 0)
            cur_iter += 1


    if trainlog:
        trainlog.close()



def setup_argumentparser():
    default_logdir = "/publicwork/coulomb_gan/"
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", '-e', "--iterations", type=int, help='number of SGD updates (in thousand)', default=500)
    parser.add_argument("-b", "--batch_size", type=int, help='batch size', default=128)
    parser.add_argument("-z", "--latentsize", type=int, help='latent size', default=32)
    parser.add_argument("-l", "--learningrate", type=float, help='learning rate', default=1e-4)
    parser.add_argument("--gpu", type=int, help='GPU to use (use -1 for CPU only)', default=-1)
    parser.add_argument("-g", "--generator", default='dcgan', choices=models.GENERATORS.keys())
    parser.add_argument("-d", "--discriminator", default='dcgan', choices=models.DISCRIMINATORS.keys())
    parser.add_argument("--l2_penalty", type=float, help="L2 weight decay term", default=0.0)
    parser.add_argument("--gen_l2p_scale", type=float, help="L2 weight decay scaling term for generator", default=1.0)
    parser.add_argument("--discriminator_lr_scale", type=float, help="LR scaling for the discriminator", default=1)
    parser.add_argument("--dimension", type=int, help='Dimension for the kernel function', default=3)
    parser.add_argument("--epsilon", type=float, help='epsilon', default=1.0)
    parser.add_argument("--threads", type=int, help='number of input threads', default=2)
    parser.add_argument("--dataset", choices=['imagenet', 'celeba', 'lsun_bedroom', 'lsun_tower', 'lsun_church_outdoor', 'cifar10', 'imagenet64'], default='celebA')
    parser.add_argument("--resume_checkpoint", type=str, help='path to model from which to resume', default='')
    parser.add_argument("--checkpoint_every", type=int, help='how often to create a new checkpoint', default=25000)
    parser.add_argument("--stats_every", type=int, help='how often to print stats during training', default=5000)
    parser.add_argument("--logdir", type=str, help='directory for TF logs and summaries', default=default_logdir)
    parser.add_argument("--logdir_suffix", type=str, help='appendix to logdir', default="")
    parser.add_argument("--disc_loss", choices=['l2', 'l1'], default='l2')
    parser.add_argument("--immediate_return", help="only here for debugging purposes", action="store_true")
    parser.add_argument("--sample_images", help="just loads a model and samples images into the given directory, then exists", type=str, default=None)
    parser.add_argument("--sample_images_size", help="How many images to sample", type=int, default=20480)
    parser.add_argument("--remember_previous", help="Reevaluate Discriminator on previous iteration's generator points", action="store_true")
    parser.add_argument("--create_summaries", help="Add a summaries for Tensorboard", action="store_true")
    parser.add_argument("--inception_path", type=str, help='Path to Inception model', default='/publicwork/coulomb_gan')
    parser.add_argument("--fid_stats", type=str, help='Path to statistics for FID (%s will be replaced by the passed dataset)', default='./th_fid_stats_%s.npz')
    parser.add_argument("--verbosity", help="verbosity level", type=int, default=1)
    return parser


if __name__ == "__main__":
    parser = setup_argumentparser()
    args = parser.parse_args()

    with torch.cuda.device(args.gpu):
        run(args.dataset.lower(), args.generator, args.discriminator, args.latentsize,
            args.dimension, args.epsilon, args.learningrate, args.batch_size, args, args.logdir)
