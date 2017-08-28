# Coulomb GAN

This repository contains an implementation of Coulomb GANs as depicted in the paper "Coulomb GANs: Provably Optimal Nash Equilibria via Potential Fields" by Unterthiner et al.

## Reproducing Paper results
As a first step, adjust the path to the datasets in `get_dataset_path` in the file `utils.py`.
Then, run using e.g.

    ./train.py --gpu 0 -z 16 --dataset lsun -d dcgan-big -i 1000 --l2_penalty 1e-7 --discriminator_lr_scale 0.5

Note that during training, the [FID](https://arxiv.org/abs/1706.08500) metric is calculated to estimate quality. But to save time, we use only 5k generated samples, which underestimates the FID. Thus to evaluate final model quality, use the `evaluate_fid.py` script.

Run `train.py --help` for further available options.

## Requirements
The implementation was developed in Python 3.6 using Tensorflow 1.2, but should work with any version of Python 3.5+ and Tensorflow 1.0+.
The implementation also assumes that training set statistics for the FID calculation are available, which can be downloaded from [the TTUR github repository](https://github.com/bioinf-jku/TTUR).

## Licensing
Note that the code is provided "as-is" under the terms of the General Public License v2. See `LICENSE.txt` for the full details.
