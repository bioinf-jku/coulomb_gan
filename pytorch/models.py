import numpy as np
from torch import nn
import torch.nn.functional as F


class BaseModule(nn.Module):
    '''nn.Module that keeps track of whether parameters need weight decay or not.'''
    def __init__(self):
        super(BaseModule, self).__init__()
        self.normal_params = []
        self.wd_params = []

    def _filter_parameters(self, module):
        def _process_module(l):
            if isinstance(l, nn.Conv2d) or \
               isinstance(l, nn.ConvTranspose2d) or \
               isinstance(l, nn.Linear):
                self.wd_params.append(l.weight)
                self.normal_params.append(l.bias)
            else:
                self.normal_params += l.parameters()

        if isinstance(module, nn.Sequential) or isinstance(module, nn.ModuleList):
            for l in module:
                _process_module(l)
        else:
            _process_module(module)


    def weight_decay_parameters(self):
        return self.wd_params

    def non_weight_decay_parameters(self):
        return self.normal_params



class DcganGenerator(nn.Module):
    def __init__(self, latentsize, img_shape):
        super(DcganGenerator, self).__init__()
        self.latentsize = latentsize
        self.init_res = 4
        n_filter = 1024
        img_res = self.init_res
        #self.projection = nn.Linear(latentsize, img_res*img_res*n_filter)
        layers = []
        layers.append(nn.ConvTranspose2d(latentsize, n_filter, 4, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(n_filter))
        layers.append(nn.ReLU(True))
        while img_res < img_shape[-1] // 2:
            #layers.append(nn.ConvTranspose2d(n_filter, n_filter // 2, 5, stride=2, bias=False))
            layers.append(nn.ConvTranspose2d(n_filter, n_filter // 2, 4, 2, 1, bias=False))
            layers.append(nn.BatchNorm2d(n_filter // 2))
            layers.append(nn.ReLU(True))
            n_filter //= 2
            img_res *= 2
        layers.append(nn.ConvTranspose2d(n_filter, img_shape[0], 4, 2, 1))
        layers.append(nn.Tanh())
        for l in layers:
            if isinstance(l, nn.ConvTranspose2d):
                nn.init.kaiming_uniform(l.weight.data, a=0)
        self.main = nn.Sequential(*layers)
        #self._filter_parameters(self.main)

    def forward(self, x):
        #x = self.projection(x)
        x = x.view(-1, self.latentsize, 1, 1)
        x = self.main(x)
        #x = 1.7159 * F.tanh(x * 2.0 / 3.0)
        x = F.tanh(x)
        return x


class DcganDiscriminator(nn.Module):
    def __init__(self, img_shape):
        super(DcganDiscriminator, self).__init__()
        n_hidden=64
        layers = []
        n_repeat = int(np.log2(int(img_shape[-1]) / 4))
        for i in range(n_repeat):
            n_in = img_shape[0] if i == 0 else n_hidden // 2
            layers.append(nn.Conv2d(n_in, n_hidden, 4, 2, 1, bias=False))
            if i > 0:
                layers.append(nn.BatchNorm2d(n_hidden))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            n_hidden *= 2
        layers.append(nn.Conv2d(n_hidden // 2, 1, 4, 1, 0))
        for l in layers:
            if isinstance(l, nn.Conv2d):
                nn.init.kaiming_uniform(l.weight.data, a=0.2)
        self.main = nn.Sequential(*layers)
        #self._filter_parameters(self.main)

    def forward(self, x):
        x = self.main(x)
        return x.view(-1, 1).squeeze(1)


class MyGenerator(BaseModule):
    def __init__(self, latentsize, img_shape):
        super(MyGenerator, self).__init__()
        self.latentsize = latentsize
        self.init_res = 4
        self.init_filter = 1024
        n_filter = self.init_filter
        img_res = self.init_res
        self.projection = nn.Linear(latentsize, img_res*img_res*n_filter)
        layers = []
        while img_res < img_shape[-1] // 2:
            layers.append(nn.ConvTranspose2d(n_filter, n_filter // 2, 5, 2, padding=2, output_padding=1))
            layers.append(nn.BatchNorm2d(n_filter // 2, momentum=0.99, eps=1e-3))
            layers.append(nn.ReLU())
            n_filter //= 2
            img_res *= 2
        layers.append(nn.ConvTranspose2d(n_filter, img_shape[0], 5, 2, padding=2, output_padding=1))
        for l in layers:
            if isinstance(l, nn.ConvTranspose2d):
                nn.init.kaiming_uniform(l.weight.data, a=0.0)
                l.bias.data[:] = 0.0
        self.main = nn.Sequential(*layers)
        self._filter_parameters(self.projection)
        self._filter_parameters(self.main)

    def forward(self, x):
        x = self.projection(x)
        x = F.relu(x)
        x = x.view(-1, self.init_filter, self.init_res, self.init_res)
        x = self.main(x)
        #x = 1.7159 * F.tanh(x * 2.0 / 3.0)
        x = F.tanh(x)
        return x


class MyDiscriminator(BaseModule):
    _n_initial_hidden = 64
    def __init__(self, img_shape):
        super(MyDiscriminator, self).__init__()
        n_hidden=self._n_initial_hidden
        layers = []
        n_repeat = int(np.log2(128 / 8))
        img_res = img_shape[-1]
        for i in range(n_repeat):
            n_in = img_shape[0] if i == 0 else n_hidden // 2
            layers.append(nn.Conv2d(n_in, n_hidden, 5, 2, padding=2))
            layers.append(nn.BatchNorm2d(n_hidden, momentum=0.99, eps=1e-3))
            layers.append(nn.LeakyReLU(0.2))
            n_hidden *= 2
            img_res //= 2
        for l in layers:
            if isinstance(l, nn.Conv2d):
                nn.init.kaiming_uniform(l.weight.data, a=0.2)
                l.bias.data[:] = 0.0
        self.main = nn.Sequential(*layers)
        self.projection = nn.Linear(img_res*img_res*(n_hidden//2), 1)
        self._filter_parameters(self.projection)
        self._filter_parameters(self.main)

    def forward(self, x):
        x = self.main(x)
        x = x.view(x.size()[0], -1)
        x = self.projection(x)
        return x.view(-1, 1).squeeze(1)


class MyBigDiscriminator(MyDiscriminator):
    _n_initial_hidden = 128


'''
class ResidualBlock2d(nn.Module):
    def __init__(n_in, n_hidden, n_out, resample=None):
        if resample == 'down':
            self.conv1 = nn.Conv2d(n_in, n_in, 3, 1, padding=1)
            self.conv2 = nn.Conv2d(n_in, n_out, 3, 1, padding=1)  # TODO: SAMPLE
            conv_shortcut = nn.Conv2d(n_in, n_out, 1)
        elif resample == 'up':





        super(ResidualBlock2d, self)
        assert resample in ('up', 'down')
        n = x.get_shape().as_list()[1]
        inputs = x

        layers = []
        layers.append(nn.BatchNorm2d(n_in, momentum=0.99, eps=1e-3))
        layers.append(nn.ReLU())
        if resample == 'up':
            layers.append(nn.UpsamplingNearest2d(scale_factor=2))
        else:
            layers.append(nn.AvgPool2d(2))
        layers.append(nn.Conv2d(n_in, n_hidden, 3, 2, padding=1))
        layers.append(nn.BatchNorm2d(n_hidden, momentum=0.99, eps=1e-3))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(n_hidden, n_out, 3, 2, padding=1))



        x = tf.layers.batch_normalization(x, training=train, scale=False, epsilon=1e-5, momentum=0.9, name=name+".bn.1")
        x = tf.nn.relu(x)
        if resample=='up':
            x = nn_upscale(x, 2)
        else:
            x = tf.layers.average_pooling2d(x, 2, 2)
        x = tf.layers.conv2d(x, n, 3, name=name+'.c1', kernel_regularizer=reg, padding='same')
        x = tf.layers.batch_normalization(x, training=train, scale=False, epsilon=1e-5, momentum=0.9, name=name+".bn.2")
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(x, n_out, 3, name=name+'.c2', kernel_regularizer=reg, padding='same')
        if resample=='up':
            y = nn_upscale(inputs, 2)
        else:
            y = tf.layers.average_pooling2d(inputs, 2, 2)
        y = tf.layers.conv2d(y, n_out, 1, name=name+'.shortcut', kernel_regularizer=reg, padding='same')
        return x + y

class SmallResnetGenerator(BaseModule):
    def __init__(self, latentsize, img_shape):
        super(MyGenerator, self).__init__()
        self.latentsize = latentsize
        self.init_res = 4
        self.init_filter = 1024
        n_filter = self.init_filter
        img_res = self.init_res
        self.projection = nn.Linear(latentsize, img_res*img_res*n_filter)
        layers = []
        while img_res < img_shape[-1] // 2:
            layers.append(nn.ConvTranspose2d(n_filter, n_filter // 2, 5, 2, padding=2, output_padding=1))
            layers.append(nn.BatchNorm2d(n_filter // 2, momentum=0.99, eps=1e-3))
            layers.append(nn.ReLU())
            n_filter //= 2
            img_res *= 2
        layers.append(nn.ConvTranspose2d(n_filter, img_shape[0], 5, 2, padding=2, output_padding=1))
        for l in layers:
            if isinstance(l, nn.ConvTranspose2d):
                nn.init.kaiming_uniform(l.weight.data, a=0.0)
                l.bias.data[:] = 0.0
        self.main = nn.Sequential(*layers)
        self._filter_parameters(self.projection)
        self._filter_parameters(self.main)

    def forward(self, x):
        x = self.projection(x)
        x = F.relu(x)
        x = x.view(-1, self.init_filter, self.init_res, self.init_res)
        x = self.main(x)
        x = 1.7159 * F.tanh(x * 2.0 / 3.0)
        return x
'''


GENERATORS = {'dcgan': DcganGenerator, 'my': MyGenerator}
DISCRIMINATORS = {'dcgan': DcganDiscriminator, 'my': MyDiscriminator, 'my-big': MyBigDiscriminator}
