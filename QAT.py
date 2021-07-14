import torch
import torch.optim as optim
import time
from Net import CIFAR10_Net, MNIST_Net, MNIST_NetBN
from utils import load_data_fashion_mnist, load_data_cifar10
from PTQ import full_inference, quantize_inference


def quantize_aware_training(model, device, train_loader, optimizer, epoch):
    lossLayer = torch.nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model.quantize_forward(data)
        loss = lossLayer(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 50 == 0:
            print('Quantize Aware Training Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                loss.item()))


if __name__ == '__main__':
    dataset = 'fmnist'
    batch_size = 64
    test_batch_size = 64
    seed = 1
    epochs = 3
    momentum = 0.5
    lr = 0.01
    save_model = True
    from_scratch = False
    use_bn = True
    torch.manual_seed(seed)  # fix the seed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dataset == 'cifar10':
        train_iter, test_iter = load_data_cifar10(batch_size=batch_size)
        model = CIFAR10_Net(num_channels=3)
        if not from_scratch:
            model.load_state_dict(torch.load('./ckpt/cifar10_cnn.pt'))
        else:
            epochs = 15
    else:
        train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size)
        if not from_scratch:
            if use_bn:
                model = MNIST_NetBN(num_channels=1)
                model.load_state_dict(torch.load('./ckpt/mnistBN_cnn.pt'))
            else:
                model = MNIST_Net(num_channels=1)
                model.load_state_dict(torch.load('./ckpt/mnist_cnn.pt'))
        else:
            if use_bn:
                model = MNIST_NetBN(num_channels=1)
            else:
                model = MNIST_Net(num_channels=1)
            epochs = 15

    # full precision inference
    begin = time.time()
    model.eval()
    with torch.no_grad():
        full_inference(model,test_iter)
    end = time.time()
    print('full inference runtime: ', end - begin)

    # quantize
    model.quantize(num_bits=4)
    model.train()

    # quantize-aware training
    # optimizer = optim.Adam(model.parameters(), lr)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    print('begin Quantize Aware Training...')
    for epoch in range(1, epochs + 1):
        quantize_aware_training(model, device, train_iter, optimizer, epoch)

    # quantize inference
    model.freeze()
    model.eval()
    begin = time.time()
    with torch.no_grad():
        quantize_inference(model,test_iter)
    end = time.time()
    print('quantize inference runtime: ',end - begin)