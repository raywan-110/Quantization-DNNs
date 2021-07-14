import torch
from torchvision import datasets, transforms
from torch.autograd import Function

def load_data_fashion_mnist(batch_size, resize=None):
    """download Fashion-MNIST dataset, then load into memory"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = datasets.FashionMNIST(root="../Data",
                                        train=True,
                                        transform=trans,
                                        download=True)
    mnist_test = datasets.FashionMNIST(root="../Data",
                                       train=False,
                                       transform=trans,
                                       download=True)
    return (torch.utils.data.DataLoader(mnist_train,
                                        batch_size,
                                        shuffle=True,
                                        num_workers=4),
            torch.utils.data.DataLoader(mnist_test,
                                        batch_size,
                                        shuffle=False,
                                        num_workers=4))

def load_data_cifar10(batch_size):
    """download Fashion-MNIST dataset, then load into memory"""
    trans = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    trans = transforms.Compose(trans)
    mnist_train = datasets.CIFAR10(root="../Data",
                                        train=True,
                                        transform=trans,
                                        download=True)
    mnist_test = datasets.CIFAR10(root="../Data",
                                       train=False,
                                       transform=trans,
                                       download=True)
    return (torch.utils.data.DataLoader(mnist_train,
                                        batch_size,
                                        shuffle=True,
                                        num_workers=4),
            torch.utils.data.DataLoader(mnist_test,
                                        batch_size,
                                        shuffle=False,
                                        num_workers=4))

class FakeQuantize(Function):

    @staticmethod
    def forward(ctx, x, qparam):
        x = qparam.quantize_tensor(x)
        x = qparam.dequantize_tensor(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        '''directly return the gradient of latter layer
        The return value corresponds to the input value, qparam doesn't need grad'''
        return grad_output, None
