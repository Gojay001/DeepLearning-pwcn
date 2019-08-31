import torch
from torch import nn, optim
import torchvision
import torchvision.transforms as transforms
import os
import time

from nin import NIN


# Device configuration
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    #=========================================
    #        1. Load datasets
    #=========================================
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_datasets = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_datasets, batch_size=16, shuffle=True, num_workers=0)

    test_datasets = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(
        test_datasets, batch_size=16, shuffle=False, num_workers=0)

    print("Train numbers:{:d}".format(len(train_datasets)))
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    #=========================================
    # 2. Define a Convolutional Neural Network
    #=========================================
    model = NIN()
    model.to(device)

    #=========================================
    # 3. Define Loss function and optimizer
    #=========================================
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    #=========================================
    #     4. Train and Test the network
    #=========================================
    epochs = 1
    for epoch in range(epochs):

        #=================4.1 start training==================
        model.train()

        torch.cuda.empty_cache()
        start = time.time()
        index = 1
        sum_loss = 0.
        correct = 0.
        total = 0.

        for images, labels in train_loader:
            # clear the cuda cache
            torch.cuda.empty_cache()
            images = images.to(device)
            labels = labels.to(device, dtype=torch.long)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Backward and Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print training loss and accuracy
            _, predicted = torch.max(outputs.data, 1)
            sum_loss += loss.item()
            total += labels.size(0)
            correct += (predicted == labels).sum()
            if index % 100 == 0:
                print("Iter: %d ===> Loss: %.8f | Acc: %.3f%%"
                      % (index, sum_loss / index, 100. * correct / total))
            index += 1

        end = time.time()
        print("Epoch [%d/%d], Loss: %.8f, Time: %.1fsec!"
              % (epoch + 1, epochs + 1, sum_loss / index, (end - start)))

        #=====================4.2 start evalidating=======================
        model.eval()

        correct_prediction = 0.
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                # clear the cuda cache
                torch.cuda.empty_cache()
                images = images.to(device)
                labels = labels.to(device, dtype=torch.long)

                # print prediction
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                # total labels number
                total += labels.size(0)
                # add correct
                correct_prediction += (predicted == labels).sum().item()
                # print("total correct number: ", correct_prediction)

        accuracy = 100. * correct_prediction / total
        print("Accuracy: %.4f%%" % accuracy)
        scheduler.step()


if __name__ == '__main__':

    main()
