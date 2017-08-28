#!/usr/bin/env python3
import os
import tensorflow as tf
import numpy as np
import scipy.misc
import fid
import pathlib
import urllib
import tempfile


def generate_images(run_name, output_dir, n_samples=20480, verbose=False):
    import train
    import tensorflow as tf

    path = pathlib.Path(run_name)
    if str(path.name).startswith('model-'):  # assume we're given a checkpoint
        ckpt = str(path.name)
        path = path.parent
        ckpt = path / ckpt
    else:
        with open(path/ 'checkpoint') as f:
            ckpt = f.readline().strip()[24:-1]

    with open(path / 'argv') as f:
        argv = f.read()

    argv += ' --resume_checkpoint %s' % ckpt
    argv += ' --sample_images %s' % output_dir
    argv += ' --sample_images_size %s' % n_samples
    argv += ' --verbosity 0'
    if verbose:
        print(argv)
    argv = argv.split()[1:]
    parser = train.setup_argumentparser()
    args = parser.parse_args(argv)
    tf.reset_default_graph()
    res = train.run(args.dataset, args.generator, args.discriminator, args.latentsize,
            args.dimension, args.epsilon, args.learningrate, args.batch_size, args, '/tmp')
    tf.reset_default_graph()
    return res


def evaluate_run(run_name, stats_path, inception_path, n_samples=20480, verbose=True):
    with tempfile.TemporaryDirectory() as tmpdir:
        if verbose:
            print("generating images", flush=True)
        generate_images(run_name, tmpdir, n_samples)
        if verbose:
            print("calculating FID", flush=True)
        fid_value = fid.calculate_fid_given_paths([tmpdir, stats_path], inception_path)
        if verbose:
            print("FID:", fid_value)
        return fid_value


if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("run_name", type=str, help='Path to the experiment run')
    #parser.add_argument("image_dir", type=str, help='Path to the generated images', default='./img')
    parser.add_argument("-s", "--stats", type=str, help='Inception statistics of real data', required=True)
    parser.add_argument("-i", "--inception", type=str, help='Path to Inception model (will be downloaded if not provided)', default=None)
    parser.add_argument("-n", "--nsamples", type=int, help='Number of samples', default=20480)
    parser.add_argument("--gpu", type=str, help='GPU to use (leave blank for CPU only)', default="")
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    evaluate_run(args.run_name, args.stats, args.inception, args.nsamples)

    #fid_value = calculate_fid(parser.image_dir, parser.stats, parser.inception, parser.unbatched)
    #print("FID:", fid_value)
