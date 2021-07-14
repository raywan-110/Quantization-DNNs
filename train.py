import torch
import torch.optim as optim
import os
from Net import CIFAR10_Net, MNIST_Net, MNIST_NetBN
from utils import load_data_fashion_mnist, load_data_cifar10

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    lossLayer = torch.nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = lossLayer(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    lossLayer = torch.nn.CrossEntropyLoss(reduction='sum')
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += lossLayer(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.0f}%\n'.format(
        test_loss, 100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    dataset = 'fmnist'
    batch_size = 64
    test_batch_size = 64
    seed = 1
    epochs = 15
    lr = 0.001
    save_model = True
    use_bn = True
    torch.manual_seed(seed)  # fix the seed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dataset == 'cifar10':
        train_iter, test_iter = load_data_cifar10(batch_size=batch_size)
        model = CIFAR10_Net(num_channels=3)
    else:
        train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size)
        if use_bn:
            model = MNIST_NetBN(num_channels=1)
        else:
            model = MNIST_Net(num_channels=1)

    optimizer = optim.Adam(model.parameters(),lr)
    print('begin training...')
    for epoch in range(1, epochs + 1):
        train(model, device, train_iter, optimizer, epoch)
        test(model, device, test_iter)
    if save_model:
        if not os.path.exists('ckpt'):
            os.makedirs('ckpt')
        if use_bn:
            torch.save(model.state_dict(), 'ckpt/mnistBN_cnn.pt')
        else:
            torch.save(model.state_dict(), 'ckpt/mnist_cnn.pt')
