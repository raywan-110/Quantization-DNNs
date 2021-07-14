import torch
import time
from Net import MNIST_Net, CIFAR10_Net, MNIST_NetBN
from utils import load_data_fashion_mnist, load_data_cifar10

def direct_quantize(model, test_loader):
    for i, (data, target) in enumerate(test_loader, 1):
        _ = model.quantize_forward(data)
        if i % 200 == 0:
            break
    print('direct quantization finish')

def quantize_inference(model, test_loader):
    correct = 0
    for _, (data, target) in enumerate(test_loader, 1):
        output = model.quantize_inference(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    print('\nTest set: Quant Model Accuracy: {:.0f}%\n'.format(100. * correct / len(test_loader.dataset)))

def full_inference(model, test_loader):
    correct = 0
    for _, (data, target) in enumerate(test_loader, 1):
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    print('\nTest set: Full Model Accuracy: {:.0f}%\n'.format(100. * correct / len(test_loader.dataset)))

if __name__ == '__main__':
    dataset = 'fmnist'
    test_batch_size = 64
    use_bn = True
    if dataset == 'cifar10':
        model = CIFAR10_Net(num_channels=3)
        model.load_state_dict(torch.load('./ckpt/cifar10_cnn.pt'))
        train_iter, test_iter = load_data_cifar10(batch_size=test_batch_size)
    else:
        if use_bn:
            model = MNIST_NetBN(num_channels=1)
            model.load_state_dict(torch.load('./ckpt/mnistBN_cnn.pt'))
        else:
            model = MNIST_Net(num_channels=1)
            model.load_state_dict(torch.load('./ckpt/mnist_cnn.pt'))
        train_iter, test_iter = load_data_fashion_mnist(batch_size=test_batch_size)

    # full precision inference
    begin = time.time()
    model.eval()
    with torch.no_grad():
        full_inference(model,test_iter)
    end = time.time()
    print('full inference runtime: ', end - begin)

    # quantize
    model.quantize(num_bits=4)
    direct_quantize(model,train_iter)  # statistics the value of rmin rmax and updates scale zero_point
    model.freeze()

    # quantize inference
    model.eval()
    begin = time.time()
    with torch.no_grad():
        quantize_inference(model,test_iter)
    end = time.time()
    print('runtime: ',end - begin)
