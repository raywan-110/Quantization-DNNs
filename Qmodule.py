import torch
import torch.nn as nn
from utils import FakeQuantize
import torch.nn.functional as F
def quantize_tensor(x, scale, zero_point, num_bits=8, signed=False):
    '''input: x, real value tensor
       output: q_x, quantized value tensor'''
    if signed:
        qmin = -2.**(num_bits - 1)
        qmax = 2.**(num_bits - 1) - 1
    else:
        qmin = 0.
        qmax = 2.**num_bits - 1
    q_x = x / scale + zero_point
    q_x.clamp_(qmin, qmax).round_()
    return q_x.int()  


def dequantize_tensor(q, scale, zero_point):
    r = scale * (q - zero_point)
    return r


class Qparam:
    def __init__(self, num_bits=8):
        self.num_bits = num_bits
        self.scale = None
        self.zero_point = None
        self.rmin = None
        self.rmax = None

    def calcScaleZeroPoint(self, rmin, rmax, num_bits=8, signed=False):
        '''calculate S, Z parameters for certain range real numbers'''
        if signed:
            qmin = -2.**(num_bits - 1)
            qmax = 2.**(num_bits - 1) - 1
        else:
            qmin = 0.
            qmax = 2.**num_bits - 1
        scale = float((rmax - rmin) / (qmax - qmin))
        zero_point = qmax - rmax / scale
        zero_point.clamp_(qmin,qmax).round_()  # actually, zero_point should be uint8
        return scale, zero_point.int()

    def update(self, tensor):
        '''update all the params via input tensor'''
        if self.rmax is None or self.rmax < tensor.max():
            self.rmax = tensor.max()
        self.rmax = 0 if self.rmax < 0 else self.rmax  # guarantee that zero_point won't overflow
        if self.rmin is None or self.rmin > tensor.min():
            self.rmin = tensor.min()
        self.rmin = 0 if self.rmin > 0 else self.rmin

        self.scale, self.zero_point = self.calcScaleZeroPoint(
            self.rmin, self.rmax, self.num_bits)

    def quantize_tensor(self, x, signed=False):
        '''input: x, real value tensor
           output: q_x, quantized value tensor'''
        if signed:
            qmin = -2.**(self.num_bits - 1)
            qmax = 2.**(self.num_bits - 1) - 1
        else:
            qmin = 0.
            qmax = 2.**self.num_bits - 1
        q_x = x / self.scale + self.zero_point
        q_x.clamp_(qmin, qmax).round_()
        return q_x.int()  # quantized data type for bias

    def dequantize_tensor(self, q):
        r = self.scale * (q - self.zero_point)
        return r


class QModule(nn.Module):
    def __init__(self, qi=True, qo=True, num_bits=8):
        super().__init__()
        if qi:
            self.qi = Qparam(num_bits)  # quantizer for input tensor
        if qo:
            self.qo = Qparam(num_bits)  # quantizer for output tensor

    def freeze(self):
        '''freeze params of the network'''
        pass

    def quantize_inference(self, x):
        raise NotImplementedError('quantize_inference must be implemented.')


class QConv2d(QModule):
    def __init__(self, conv_module, qi=True, qo=True, num_bits=8):
        super(QConv2d,
              self).__init__(qi=qi, qo=qo,
                             num_bits=num_bits)  # quantizer for input output
        self.num_bits = num_bits
        self.conv_module = conv_module
        self.qw = Qparam(num_bits)  # quantizer for weights

    def freeze(self, qi=None, qo=None):
        '''some hidden module may not need to calculate rmin/rmax but reuse former
        qo as its own qi. for conv2d layer, it has its own qi qo'''

        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')

        if hasattr(self, 'qo') and qo is not None:
            raise ValueError('qo has been provided in init function.')
        if not hasattr(self, 'qo') and qo is None:
            raise ValueError('qo is not existed, should be provided.')

        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo

        # TODO implement it by bit shift
        self.M = self.qw.scale * self.qi.scale / self.qo.scale  # actually, it should be implement by bit shift
        self.conv_module.weight.data = self.qw.quantize_tensor(self.conv_module.weight.data)
        # TODO expand the product
        self.conv_module.weight.data = self.conv_module.weight.data - self.qw.zero_point

        self.conv_module.bias.data = quantize_tensor(
            self.conv_module.bias.data,
            self.qw.scale * self.qi.scale,
            zero_point=0,
            num_bits=32,
            signed=True
        )  # since Z = 0, r=Sq, the value range of q must contain negative numbers

    # used in QAT
    def forward(self, x):
        # statistics and update
        if hasattr(self, 'qi'):
            self.qi.update(x)
            # simulate quantization effects of input
            x = FakeQuantize.apply(x, self.qi)

        self.qw.update(self.conv_module.weight.data)
        # simulate quantization effects
        self.conv_module.weight.data = FakeQuantize.apply(self.conv_module.weight.data,self.qw)
        x = self.conv_module(x)
        # x = F.conv2d(x,
        #              FakeQuantize.apply(self.conv_module.weight, self.qw),
        #              self.conv_module.bias,
        #              stride=self.conv_module.stride,
        #              padding=self.conv_module.padding)

        # qo's params maybe useful for latter layers
        if hasattr(self, 'qo'):
            self.qo.update(x)
            # simulate quantization effects of input
            x = FakeQuantize.apply(x, self.qo)

        return x

    def quantize_inference(self, x):
        # use original formula to calculate
        x = x - self.qi.zero_point
        # calculate in 32 bits integer
        x = self.conv_module(x)
        x = (self.M * x).round() 
        x = x + self.qo.zero_point
        # clamp
        x.clamp_(0., 2**self.num_bits - 1.).round_()
        return x.int()

class QLinear(QModule):
    def __init__(self, fc_module, qi, qo, num_bits=8):
        super().__init__(qi=qi, qo=qo, num_bits=num_bits)
        self.num_bits = num_bits
        self.fc_module = fc_module
        self.qw = Qparam(num_bits)

    def freeze(self, qi=None, qo=None):
        '''some hidden module may not need to calculate rmin/rmax but reuse former
        qo as its own qi. for conv2d layer, it has its own qi qo'''

        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')

        if hasattr(self, 'qo') and qo is not None:
            raise ValueError('qo has been provided in init function.')
        if not hasattr(self, 'qo') and qo is None:
            raise ValueError('qo is not existed, should be provided.')

        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo

        # TODO implement it by bit shift
        self.M = self.qw.scale * self.qi.scale / self.qo.scale  # actually, it should be implement by bit shift
        self.fc_module.weight.data = self.qw.quantize_tensor(
            self.fc_module.weight.data)
        # TODO expand the product
        self.fc_module.weight.data = self.fc_module.weight.data - self.qw.zero_point

        self.fc_module.bias.data = quantize_tensor(
            self.fc_module.bias.data,
            self.qw.scale * self.qi.scale,
            zero_point=0,
            num_bits=32,
            signed=True
        )  # since Z = 0, r=Sq, the value range of q must contain negative numbers

    # used in QAT
    def forward(self, x):
        # statistics and update
        if hasattr(self, 'qi'):
            self.qi.update(x)
            # simulate quantization effects of input
            x = FakeQuantize.apply(x,self.qi)
        self.qw.update(self.fc_module.weight.data)

        # simulate quantization effects
        # self.fc_module.weight.data = FakeQuantize.apply(self.fc_module.weight.data, self.qw)
        # x = self.fc_module(x)  # no need to quantize bias
        x = F.linear(x, FakeQuantize.apply(self.fc_module.weight, self.qw),
                     self.fc_module.bias)
        # qo's params maybe useful for latter layers
        if hasattr(self, 'qo'):
            self.qo.update(x)
            # simulate quantization effects
            x = FakeQuantize.apply(x,self.qo)

        return x

    def quantize_inference(self, x):
        # use original formula to calculate
        x = x - self.qi.zero_point
        # calculate in integer
        x = self.fc_module(x)
        x = (self.M * x).round()
        x = x + self.qo.zero_point
        # from 32 bits rescale to 8 bits
        # max, min = x.max(), x.min()
        # max = 0 if max < 0 else max
        # min = 0 if min > 0 else min
        # scale = (max - min) / (2 ** self.num_bits - 1 - 0)
        # zero_point = (2 ** self.num_bits - 1) - max / scale
        # x = x / scale + zero_point.int()
        # clamp
        x.clamp_(0., 2**self.num_bits - 1.).round_()
        return x.int()


class QReLU(QModule):
    def __init__(self, relu_module, qi=False, num_bits=None):
        super(QReLU, self).__init__(qi=qi, num_bits=num_bits)
        self.relu_module = relu_module

    def freeze(self, qi=None):

        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')

        if qi is not None:
            self.qi = qi  # relu module reuse former layers' qo

    def forward(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = FakeQuantize.apply(x,self.qi)

        x = F.relu(x)
        return x

    def quantize_inference(self, x):
        x = x.clone()
        x[x < self.qi.zero_point] = self.qi.zero_point
        return x


class QMaxPooling2d(QModule):
    def __init__(self,
                 maxpooling2d_module,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 qi=False,
                 num_bits=None):
        super(QMaxPooling2d, self).__init__(qi=qi, num_bits=num_bits)
        self.maxpooling2d_module = maxpooling2d_module
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def freeze(self, qi=None):
        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')
        if qi is not None:
            self.qi = qi

    def forward(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = FakeQuantize.apply(x,self.qi)

        x = F.max_pool2d(x, self.kernel_size, self.stride, self.padding)

        return x

    def quantize_inference(self, x):
        x = F.max_pool2d(x.float(), self.kernel_size, self.stride, self.padding)
        return x.int()


class QConvBNReLU(QModule):
    def __init__(self, conv_module, bn_module, qi=True, qo=True, num_bits=8):
        super(QConvBNReLU,self).__init__(qi=qi, qo=qo, num_bits=num_bits)
        self.num_bits = num_bits
        self.conv_module = conv_module
        self.bn_module = bn_module
        self.qw = Qparam(num_bits=8)

    def fold_on(self, mean, std):
        '''get the mean and std of the pure outputs after conv layer
        and then reset the weights and bias'''
        if self.bn_module.affine:
            '''do the affine transform'''
            gamma_ = self.bn_module.weight / std  # weight is gamma
            weight = self.conv_module.weight * gamma_.view(self.conv_module.out_channels, 1, 1, 1)  # the number of gamma params corresponds to the number of output channels
            if self.conv_module.bias is not None:
                bias = gamma_ * self.conv_module.bias - gamma_ * mean + self.bn_module.bias
            else:
                bias = self.bn_module.bias - gamma_ * mean

        else:
            '''without affine transform'''
            gamma_ = 1 / std
            weight = self.conv_module.weight * gamma_
            if self.conv_module.bias is not None:
                bias = gamma_ * self.conv_module.bias - gamma_ * mean
            else:
                bias = -gamma_ * mean

        return weight, bias

    def forward(self, x):

        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = FakeQuantize.apply(x, self.qi)

        if self.training:
            y = self.conv_module(x)  # preparen for calculate mean and std

            y = y.permute(1,0,2,3)  # NCHW -> CNHW
            # y = y.contiguous().view(self.conv_module.out_channels,-1)  # C,N,H,W -> C,NHW
            y = y.reshape(self.conv_module.out_channels,-1)

            # calculate mean and std for each channels
            mean = y.mean(1).detach()
            var = y.var(1).detach()
            # update params in BN layer
            self.bn_module.running_mean = self.bn_module.running_mean * self.bn_module.momentum + \
                (1 - self.bn_module.momentum) * mean
            self.bn_module.running_var = self.bn_module.running_var * self.bn_module.momentum + \
                (1 - self.bn_module.momentum) * var
        else:
            # use fixed params
            mean = self.bn_module.running_mean
            var = self.bn_module.runnning_var

        std = torch.sqrt(var + self.bn_module.eps)
        weight, bias = self.fold_on(mean, std)  # return folded weights and bias
        self.qw.update(weight.data)  # update S Zp

        # use folded weight and bias to calculate
        x = F.conv2d(x,
                     FakeQuantize.apply(weight, self.qw),
                     bias,
                     stride=self.conv_module.stride,
                     padding=self.conv_module.padding)
        x = F.relu(x)

        if hasattr(self, 'qo'):
            self.qo.update(x)  # calculate S Zp after relu, the Zp will retain 0 all the time
            x = FakeQuantize.apply(x, self.qo)

        return x

    def freeze(self,qi=None, qo=None):
        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')

        if hasattr(self, 'qo') and qo is not None:
            raise ValueError('qo has been provided in init function.')
        if not hasattr(self, 'qo') and qo is None:
            raise ValueError('qo is not existed, should be provided.')

        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo

        # TODO implement it by bit shift
        self.M = self.qw.scale * self.qi.scale / self.qo.scale  # actually, it should be implement by bit shift

        weight, bias = self.fold_on(self.bn_module.running_mean,self.bn_module.running_var)
        # prepare folded weight and bias for inference 
        self.conv_module.weight.data = self.qw.quantize_tensor(weight.data)
        self.conv_module.weight.data = self.conv_module.weight.data - self.qw.zero_point
        self.conv_module.bias.data = quantize_tensor(bias, scale=self.qi.scale * self.qw.scale, zero_point=0, num_bits=32, signed=True)

    def quantize_inference(self, x):
        x = x - self.qi.zero_point
        x = self.conv_module(x)  # folded conv module
        x = (self.M * x).round()
        x = x + self.qo.zero_point
        x.clamp_(self.qo.zero_point, 2 ** self.num_bits - 1).round_()  # in fact, qo.zero_point = 0
        return x.int()





