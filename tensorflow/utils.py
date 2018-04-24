import numpy as np
import tensorflow as tf
import os
import sys
import h5py
import matplotlib.pyplot as plt



def startup_bookkeeping(logdir, curent_file):
    '''
    Performs common operations at start of each run.

    Writes PID and argv into files in the logdir and copies a backup
    of the current source code files there as well.

    logdir is expected to be a pathlib.Path
    '''

    import shutil
    from pathlib import Path

    if not logdir.exists():
        logdir.mkdir(parents=True, exist_ok=True)

    with open(str(logdir / "pid.txt"), 'w') as f:  # write PID so we can abort from outside
        f.write(str(os.getpid()))
        f.write("\n")

    # make copy of current source files
    dest = logdir / 'sources'
    dest.mkdir(exist_ok=True)
    localdir = Path(curent_file).parent
    pyfiles = localdir.glob("*.py")
    for fn in localdir.glob("*.py"):
        shutil.copy(str(fn), str(dest / fn))

    with open(str(logdir / "argv"), 'w') as f:
        f.write(' '.join(sys.argv))
        f.write("\n")



def plot_tiles(data, nrows=8, ncols=8, ax=None, local_norm="maxabs", data_format='NWHC', **kwargs):
    '''
    Plots several images as tiles next to each other.
    Data is assumed to be in NWHC format. If it is shaped wrongly, we will
    try to heuristically determine the correct thing.

    TODO: for now, only square images are supported
    '''
    assert data_format in ('NWHC', 'NCWH'), "Unknown data format"

    n = data.shape[0]
    if len(data.shape) == 2:  # flattened array
        a = np.sqrt(data.shape[-1])
        if a == int(a): # size is square, so probably just 1 channel
            w, h, c = int(a), int(a), 1
        else: # try assuming 3 channels
            a = np.sqrt(data.shape[-1]//3)
            if a == int(a):
                w, h, c = int(a), int(a), 3
            else:
                raise RuntimeError("Unable to determine shape of images")
    elif len(data.shape) == 3: # single color channel
        w, h, c = data.shape[1], data.shape[2], 1
    elif len(data.shape) == 4:
        n, w, h, c = data.shape
    else:
        raise RuntimeError("Unable to determine shape of images")

    if data_format == 'NWHC':
        data = data.reshape(n, w, h, c)
    else:
        data = data.reshape(n, c, w, h)
        data = data.transpose(0, 2, 3, 1)  # rest of code always assumes NWHC


    assert c == 1 or c == 3, "Data must be in NWHC format"
    assert w == h, "Only square images are supported"

    ppi = w + 2  # +2 for borders
    imgshape = (nrows*ppi, ncols*ppi, c)
    # make sure border is black
    img = {"maxabs": lambda s: (data.min() / np.abs(data).max()) * np.ones(imgshape, dtype=data.dtype),
           "minmax": lambda s: np.zeros(imgshape, dtype=data.dtype),
           "none":   lambda s: np.ones(imgshape, dtype=data.dtype)*data.min()
            }[local_norm.lower()](None)

    n = min(nrows*ncols, data.shape[0])
    normfunc = {"maxabs": lambda d: d / np.abs(d).max(),
                "minmax": lambda d: (d - d.min()) / d.ptp(),
                "none":   lambda d: d}[local_norm.lower()]
    idx = 0
    for ri in range(nrows):
        for ci in range(ncols):
            if idx >= n:
                break
            img[ri*ppi+1:(ri+1)*ppi-1, ci*ppi+1:(ci+1)*ppi-1] = normfunc(data[idx])
            idx += 1
    if ax is None:
        fig = plt.figure(facecolor="black", **kwargs)
        fig.subplots_adjust(hspace=0, top=1, bottom=0,
                            wspace=0, left=0, right=1)
        ax = fig.gca()
    else:
        fig = None
    if c > 1:
        ax.imshow(img, interpolation="none")
    else:
        ax.imshow(img.reshape(nrows*ppi, ncols*ppi),
                    interpolation="none", cmap="Greys_r")
    ax.axis("off")
    return fig


def get_timestamp(fmt='%y%m%d_%H%M'):
    '''Returns a string that contains the current date and time.

    Suggested formats:
        short_format=%y%m%d_%H%M  (default)
        long format=%Y%m%d_%H%M%S
    '''
    import datetime
    now = datetime.datetime.now()
    return datetime.datetime.strftime(now, fmt)



def get_dataset_path(dataset_name):
    dataset = dataset_name.lower()
    if dataset == "celeba":
        n_samples = 202599
        dataset_directory = '/local00/bioinf/celebA_cropped/'
        if not os.path.exists(dataset_directory):
            dataset_directory = '/local10/bioinf/celebA_cropped/' # raptor
        if not os.path.exists(dataset_directory):
            print("Reading input files over network")
            dataset_directory = '/publicdata/image/celebA_cropped/'
        dataset_pattern = os.path.join(dataset_directory, '*.jpg')
        img_shape = [64, 64, 3]
    elif dataset == 'lsun':
        n_samples = 3033042
        dataset_directory = '/local00/bioinf/lsun_cropped/'
        if not os.path.exists(dataset_directory):
            dataset_directory = '/local10/k40data/tom/lsun_cropped/' # k40
        if not os.path.exists(dataset_directory):
            print("Reading input files over network")
            dataset_directory = '/publicdata/image/lsun/lsun_cropped/'
        dataset_pattern = os.path.join(dataset_directory, '*', '*.jpg')
        img_shape = [64, 64, 3]
    elif dataset == 'cifar10':
        n_samples = 60000
        dataset_directory = '/local00/bioinf/cifar10_img/original/train/'
        if not os.path.exists(dataset_directory):
            print("Reading input files over network")
            dataset_directory = '/publicdata/image/cifar10_img/original/train/'
        dataset_pattern = os.path.join(dataset_directory, '*.png')
        img_shape = [32, 32, 3]
    elif dataset == 'billion_word':
        n_samples = None
        dataset_directory = '/local00/bioinf/google_billion_word/'
        if not os.path.exists(dataset_directory):
            print("Reading input files over network")
            dataset_directory = '/publicdata/nlp/google_billion_word/1-billion-word-language-modeling-benchmark-r13output/'
        dataset_pattern = dataset_directory
        img_shape = None
    else:
        raise RuntimeError("Unknown dataset")

    return dataset_pattern, n_samples, img_shape

def load_text_dataset(path, batch_size, max_length, max_n_examples, max_vocab_size, dtype=tf.float32, shuffle=True, num_epochs=None):
    print("loading dataset...")

    lines = []

    finished = False

    for i in range(99):
        path_ = path + "training-monolingual.tokenized.shuffled/news.en-{}-of-00100".format(str(i+1).zfill(5))
        with open(str(path_), 'r', encoding='utf-8') as f:
            for line in f:
                line = line[:-1]
                line = tuple(line)

                if len(line) > max_length:
                    line = line[:max_length]

                lines.append(line + ( ("`",)*(max_length-len(line)) ) )

                if len(lines) == max_n_examples:
                    finished = True
                    break
        if finished:
            break
    # don't shuffle here, we'll shuffle in the input queue producer
    #np.random.shuffle(lines)

    import collections
    counts = collections.Counter(char for line in lines for char in line)

    charmap = {}
    inv_charmap = []

    for char,count in counts.most_common(max_vocab_size-1):
        if char not in charmap:
            charmap[char] = len(inv_charmap)
            inv_charmap.append(char)

    #now sort the charmap
    inv_charmap = sorted(inv_charmap)
    for i, char in enumerate(inv_charmap):
      charmap[char] = i+1

    # add the unk
    inv_charmap = ['unk'] + inv_charmap
    charmap['unk'] = 0

    lines_as_ints = np.zeros((max_n_examples, max_length), dtype=np.int32)
    for i,line in enumerate(lines):
        for j,char in enumerate(line):
            if char in charmap:
                lines_as_ints[i,j] = charmap[char]
            else:
                lines_as_ints[i,j] = charmap['unk']

    min_after_dequeue = batch_size * 1000
    capacity = min_after_dequeue + (4 * batch_size)
    input_producer = tf.train.input_producer(lines_as_ints, shuffle=shuffle, capacity=capacity, num_epochs=num_epochs)
    y_batch = tf.one_hot(input_producer.dequeue_many(batch_size), len(charmap))
    print("loaded %s lines in dataset" % (len(lines)))
    return y_batch, lines_as_ints, charmap, inv_charmap

def load_image_dataset(path, batch_size, img_shape, n_threads, allow_smaller_final_batch=False, dtype=tf.float32, num_epochs=None):
    with tf.variable_scope('input'):
        filelist = tf.train.match_filenames_once(path)
        filequeue = tf.train.string_input_producer(filelist, num_epochs=num_epochs, name="filename_queue")

        def read_sample(filequeue):
            reader = tf.WholeFileReader()
            fn, img = reader.read(filequeue)
            data = tf.image.decode_image(img)
            data = tf.reshape(data, img_shape)
            data.set_shape(img_shape)
            return [data]

        data = [read_sample(filequeue) for _ in range(n_threads)]
        min_after_dequeue = batch_size * 1000
        capacity = min_after_dequeue + (4 * batch_size * n_threads)
        y_batch = tf.train.shuffle_batch_join(data, batch_size=batch_size, capacity=capacity,
                                         min_after_dequeue=min_after_dequeue,
                                         allow_smaller_final_batch=allow_smaller_final_batch,
                                         name="minibatch_queue")
        # preprocessing
        y_batch = tf.cast(y_batch, dtype, 'y_batch')
        y_batch /= 127.5
        y_batch -= 1.0
    return y_batch


def calculate_squared_distances(a, b):
    '''returns the squared distances between all elements in a and in b as a matrix
    of shape #a * #b'''
    na = tf.shape(a)[0]
    nb = tf.shape(b)[0]
    nas, nbs = a.get_shape().as_list(), b.get_shape().as_list()
    a = tf.reshape(a, [na, 1, -1])
    b = tf.reshape(b, [1, nb, -1])
    a.set_shape([nas[0], 1, np.prod(nas[1:])])
    b.set_shape([1, nbs[0], np.prod(nbs[1:])])
    a = tf.tile(a, [1, nb, 1])
    b = tf.tile(b, [na, 1, 1])
    d = a-b
    return tf.reduce_sum(tf.square(d), axis=2)


def plummer_kernel(a, b, dimension, epsilon):
    r = calculate_squared_distances(a, b)
    r += epsilon*epsilon
    f1 = dimension-2
    return tf.pow(r, -f1 / 2)


def get_potentials(x, y, dimension, cur_epsilon):
    '''
    This is alsmost the same `calculate_potential`, but
        px, py = get_potentials(x, y)
    is faster than:
        px = calculate_potential(x, y, x)
        py = calculate_potential(x, y, y)
    because we calculate the cross terms only once.
    '''
    x_fixed = tf.stop_gradient(x)
    y_fixed = tf.stop_gradient(y)
    nx = tf.cast(tf.shape(x)[0], x.dtype)
    ny = tf.cast(tf.shape(y)[0], y.dtype)
    pk_xx = plummer_kernel(x_fixed, x, dimension, cur_epsilon)
    pk_yx = plummer_kernel(y, x, dimension, cur_epsilon)
    pk_yy = plummer_kernel(y_fixed, y, dimension, cur_epsilon)
    #pk_xx = tf.matrix_set_diag(pk_xx, tf.ones(shape=x.get_shape()[0], dtype=pk_xx.dtype))
    #pk_yy = tf.matrix_set_diag(pk_yy, tf.ones(shape=y.get_shape()[0], dtype=pk_yy.dtype))
    kxx = tf.reduce_sum(pk_xx, axis=0) / (nx)
    kyx = tf.reduce_sum(pk_yx, axis=0) / ny
    kxy = tf.reduce_sum(pk_yx, axis=1) / (nx)
    kyy = tf.reduce_sum(pk_yy, axis=0) / ny
    pot_x = kxx - kyx
    pot_y = kxy - kyy
    pot_x = tf.reshape(pot_x, [-1])
    pot_y = tf.reshape(pot_y, [-1])
    return pot_x, pot_y


def calculate_potential(x, y, a, dimension, epsilon):
    x = tf.stop_gradient(x)
    y = tf.stop_gradient(y)
    nx = tf.cast(tf.shape(x)[0], x.dtype)
    ny = tf.cast(tf.shape(y)[0], y.dtype)
    kxa = plummer_kernel(x, a, dimension, epsilon)
    kxa = tf.reduce_sum(kxa, axis=0) / nx
    kya = plummer_kernel(y, a, dimension, epsilon)
    kya = tf.reduce_sum(kya, axis=0) / ny
    p = kxa - kya
    p = tf.reshape(p, [-1])
    return p
