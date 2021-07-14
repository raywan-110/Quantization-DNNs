import torch
import torch.nn as nn

from Qmodule import *


class MNIST_Net(nn.Module):
    def __init__(self, num_channels=1):
        super(MNIST_Net, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 40, 3, 1)
        self.conv2 = nn.Conv2d(40, 40, 3, 1)
        self.fc = nn.Linear(5 * 5 * 40, 10)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.maxpool2d_1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.maxpool2d_2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.maxpool2d_1(x)
        x = self.relu2(self.conv2(x))
        x = self.maxpool2d_2(x)
        x = x.view(-1, 5 * 5 * 40)
        x = self.fc(x)
        return x

    def quantize(self, num_bits=8):
        self.qconv1 = QConv2d(self.conv1, qi=True, qo=True, num_bits=num_bits)
        self.qrelu1 = QReLU(self.relu1)
        self.qmaxpool2d_1 = QMaxPooling2d(self.maxpool2d_1,kernel_size=2,stride=2,padding=0)
        self.qconv2 = QConv2d(self.conv2, qi=False, qo=True, num_bits=num_bits)
        self.qrelu2 = QReLU(self.relu2)
        self.qmaxpool2d_2 = QMaxPooling2d(self.maxpool2d_2,kernel_size=2,stride=2,padding=0)
        self.qfc = QLinear(self.fc, qi=False, qo=True, num_bits=num_bits)

    # forward and update QParams
    def quantize_forward(self, x):
        x = self.qrelu1(self.qconv1(x))
        x = self.qmaxpool2d_1(x)
        x = self.qrelu2(self.qconv2(x))
        x = self.qmaxpool2d_2(x)
        x = x.view(-1, 5 * 5 * 40)
        x = self.qfc(x)
        return x

    def freeze(self):
        self.qconv1.freeze()
        self.qrelu1.freeze(qi=self.qconv1.qo)
        self.qmaxpool2d_1.freeze(qi=self.qconv1.qo)
        self.qconv2.freeze(qi=self.qconv1.qo)
        self.qrelu2.freeze(qi=self.qconv2.qo)
        self.qmaxpool2d_2.freeze(qi=self.qconv2.qo)
        self.qfc.freeze(qi=self.qconv2.qo)

    def quantize_inference(self, x):
        # input should be quantized, and then all the calculations are performed on integer
        qx = self.qconv1.qi.quantize_tensor(x)
        qx = self.qconv1.quantize_inference(qx)
        qx = self.qrelu1.quantize_inference(qx)
        qx = self.qmaxpool2d_1.quantize_inference(qx)
        qx = self.qconv2.quantize_inference(qx)
        qx = self.qrelu2.quantize_inference(qx)
        qx = self.qmaxpool2d_2.quantize_inference(qx)
        qx = qx.view(-1, 5 * 5 * 40)
        qx = self.qfc.quantize_inference(qx)
        qx = self.qfc.qo.dequantize_tensor(qx)
        return qx

class MNIST_NetBN(nn.Module):

    def __init__(self, num_channels=1):
        super(MNIST_NetBN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 40, 3, 1)
        self.bn1 = nn.BatchNorm2d(40)
        self.relu1 = nn.ReLU()
        self.maxpool2d_1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(40, 40, 3, 1)
        self.bn2 = nn.BatchNorm2d(40)
        self.relu2 = nn.ReLU()
        self.maxpool2d_2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc = nn.Linear(5 * 5 * 40, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool2d_1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.maxpool2d_2(x)
        x = x.view(-1, 5 * 5 * 40)
        x = self.fc(x)
        return x

    def quantize(self, num_bits=8):
        self.qFconv1 = QConvBNReLU(self.conv1, self.bn1, qi=True, qo=True, num_bits=num_bits)
        self.qmaxpool2d_1 = QMaxPooling2d(self.maxpool2d_1,kernel_size=2, stride=2, padding=0)
        self.qFconv2 = QConvBNReLU(self.conv2, self.bn2, qi=False, qo=True, num_bits=num_bits)
        self.qmaxpool2d_2 = QMaxPooling2d(self.maxpool2d_2,kernel_size=2, stride=2, padding=0)
        self.qfc = QLinear(self.fc, qi=False, qo=True, num_bits=num_bits)

    def quantize_forward(self, x):
        x = self.qFconv1(x)
        x = self.qmaxpool2d_1(x)
        x = self.qFconv2(x)
        x = self.qmaxpool2d_2(x)
        x = x.view(-1, 5 * 5 * 40)
        x = self.qfc(x)
        return x

    def freeze(self):
        self.qFconv1.freeze()
        self.qmaxpool2d_1.freeze(self.qFconv1.qo)
        self.qFconv2.freeze(qi=self.qFconv1.qo)
        self.qmaxpool2d_2.freeze(self.qFconv2.qo)
        self.qfc.freeze(qi=self.qFconv2.qo)

    def quantize_inference(self, x):
        qx = self.qFconv1.qi.quantize_tensor(x)
        qx = self.qFconv1.quantize_inference(qx)
        qx = self.qmaxpool2d_1.quantize_inference(qx)
        qx = self.qFconv2.quantize_inference(qx)
        qx = self.qmaxpool2d_2.quantize_inference(qx)
        qx = qx.view(-1, 5 * 5 * 40)
        qx = self.qfc.quantize_inference(qx)
        out = self.qfc.qo.dequantize_tensor(qx)
        return out

class CIFAR10_Net(nn.Module):
    def __init__(self, num_channels=3):  # 初始化网络结构
        super(CIFAR10_Net, self).__init__()  # 多继承需用到super函数
        self.conv1 = nn.Conv2d(num_channels, 16, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()

    def forward(self, x):  # 正向传播过程
        x = self.relu1(self.conv1(x))  # input(3, 32, 32) output(16, 28, 28)
        x = self.pool1(x)  # output(16, 14, 14)
        x = self.relu2(self.conv2(x))  # output(32, 10, 10)
        x = self.pool2(x)  # output(32, 5, 5)
        x = x.view(-1, 32 * 5 * 5)  # output(32*5*5)
        x = self.relu3(self.fc1(x))  # output(120)
        x = self.relu4(self.fc2(x))  # output(84)
        x = self.fc3(x)  # output(10)
        return x

    def quantize(self, num_bits=8):
        self.qconv1 = QConv2d(self.conv1, qi=True, qo=True, num_bits=num_bits)
        self.qrelu1 = QReLU(self.relu1)
        self.qpoo11 = QMaxPooling2d(self.pool1)
        self.qconv2 = QConv2d(self.conv2, qi=False, qo=True, num_bits=num_bits)
        self.qrelu2 = QReLU(self.relu2)
        self.qpool2 = QMaxPooling2d(self.pool2)
        self.qfc1 = QLinear(self.fc1, qi=False, qo=True, num_bits=num_bits)
        self.qrelu3 = QReLU(self.relu3)
        self.qfc2 = QLinear(self.fc2, qi=False, qo=True, num_bits=num_bits)
        self.qrelu4 = QReLU(self.relu4)
        self.qfc3 = QLinear(self.fc3, qi=False, qo=True, num_bits=num_bits)

    def freeze(self):
        self.qconv1.freeze()
        self.qrelu1.freeze(qi=self.qconv1.qo)
        self.qpoo11.freeze(qi=self.qconv1.qo)
        self.qconv2.freeze(qi=self.qconv1.qo)
        self.qrelu2.freeze(qi=self.qconv2.qo)
        self.qpool2.freeze(qi=self.qconv2.qo)

        self.qfc1.freeze(qi=self.qconv2.qo)
        self.qrelu3.freeze(qi=self.qfc1.qo)
        self.qfc2.freeze(qi=self.qfc1.qo)
        self.qrelu4.freeze(qi=self.qfc2.qo)
        self.qfc3.freeze(qi=self.qfc2.qo)

    # forward and update QParams
    def quantize_forward(self, x):
        x = self.qrelu1(self.qconv1(x))  # input(3, 32, 32) output(16, 28, 28)
        x = self.qpoo11(x)  # output(16, 14, 14)
        x = self.qrelu2(self.qconv2(x))  # output(32, 10, 10)
        x = self.qpool2(x)  # output(32, 5, 5)
        x = x.view(-1, 32 * 5 * 5)  # output(32*5*5)
        x = self.qrelu3(self.qfc1(x))  # output(120)
        x = self.qrelu4(self.qfc2(x))  # output(84)
        x = self.qfc3(x)  # output(10)
        return x

    def quantize_inference(self, x):
        # input should be quantized, and then all the calculations are performed on integer
        qx = self.qconv1.qi.quantize_tensor(x)
        qx = self.qconv1.quantize_inference(qx)
        qx = self.qrelu1.quantize_inference(qx)
        qx = self.qpoo11.quantize_inference(qx)
        qx = self.qconv2.quantize_inference(qx)
        qx = self.qrelu2.quantize_inference(qx)
        qx = self.qpool2.quantize_inference(qx)
        qx = qx.view(-1, 32 * 5 * 5)
        qx = self.qfc1.quantize_inference(qx)
        qx = self.qrelu3.quantize_inference(qx)
        qx = self.qfc2.quantize_inference(qx)
        qx = self.qrelu4.quantize_inference(qx)
        qx = self.qfc3.quantize_inference(qx)
        qx = self.qfc3.qo.dequantize_tensor(qx)  # convert to float32
        return qx



