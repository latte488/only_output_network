from torch.utils.data import DataLoader
from torchvision import datasets, transforms

__root = '/tmp'

def loader(dataset, batch_size, train_transform, test_transform):
    train_dataset = dataset(
        root=__root,
        train=True,
        download=True,
        transform=train_transform
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    test_dataset = dataset(
        root=__root,
        train=False,
        download=True,
        transform=train_transform
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    return train_loader, test_loader

def mnist_loader(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    return loader(datasets.MNIST, batch_size, transform, transform)

def cifar10_loader(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    return loader(datasets.CIFAR10, batch_size, transform, transform)

def loader_test(loader):
    train_loader, test_loader = loader
    for train, test in zip(train_loader, test_loader):
        train_x, train_y = train
        test_x, test_y = test
        print(train_x.shape)
        print(train_y.shape)
        print(test_x.shape)
        print(test_y.shape)
        print(test_y)
        break

def prepare(batch_size):
    return cifar10_loader(batch_size)

num_classes=10

if __name__ == '__main__':
    loader_test(mnist_loader(48))
    loader_test(cifar10_loader(48))
