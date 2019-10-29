import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets
import visdom
import argparse
import os
import time
import numpy as np

from DataProcess.data_config import config
from DataProcess.data_loader import get_folders, MyDataSet
# from googlenet import googlenet


# Device configuration
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vis = visdom.Visdom()


def main():

    #=======================================
    #           1. Load dataset
    #=======================================
    # CIFAR data
    data_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([
                                 0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    train_set = datasets.CIFAR10(
        './data', train=True, transform=data_tf, download=True)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=64, shuffle=True)
    test_set = datasets.CIFAR10(
        './data', train=False, transform=data_tf, download=True)
    valid_loader = torch.utils.data.DataLoader(
        test_set, batch_size=128, shuffle=False)
    # custom datasets
    # train_folders, valid_folders = get_folders(
    #     config.train_data, config.valid_data)
    # train_datasets = MyDataSet(train_folders, transforms=None)
    # valid_datasets = MyDataSet(valid_folders, transforms=None, train=False)
    # train_loader = DataLoader(dataset=train_datasets,
    #                           batch_size=config.batch_size, shuffle=True)
    # valid_loader = DataLoader(dataset=valid_datasets,
    #                           batch_size=config.batch_size, shuffle=True)
    print("Train numbers:{:d}".format(len(train_loader)))
    print("Test numbers:{:d}".format(len(valid_loader)))

    #=======================================
    #   2. Define network and Load model
    #=======================================
    if config.pretrained:
        model = models.googlenet(num_classes=config.num_classes)
        checkpoint = torch.load(config.model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("load model success")
    else:
        model = models.googlenet(pretrained=True)
        # adjust last fc layer to class number
        channel_in = model.fc.in_features
        model.fc = nn.Linear(channel_in, config.num_classes)
    model.to(device)

    #=======================================
    # 3. Define Loss function and optimizer
    #=======================================
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr,
                           amsgrad=True, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    #=======================================
    #     4. Train and Test the network
    #=======================================
    best_accuracy = 0.
    epoch = 0
    resume = False

    # ====4.1 restart the training process====
    if resume:
        checkpoint = torch.load(config.model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]
        best_accuracy = checkpoint["best_accuracy"]
        # print checkpoint
        print('epoch:', epoch)
        print('best_accuracy:', best_accuracy)
        print("model's state_dict:")
        for param_tensor in model.state_dict():
            print(param_tensor, '\t', model.state_dict()[param_tensor].size())

    for epoch in range(1, config.epochs + 1):

        # ====4.2 start training====
        print("======start training=====")
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
            if index % 10 == 0:
                print("Iter: %d ===> Loss: %.8f | Acc: %.3f%%"
                      % (index, sum_loss / index, 100. * correct / total))
            index += 1

        end = time.time()
        print("Epoch [%d/%d], Loss: %.8f, Time: %.1fsec!"
              % (epoch, config.epochs, loss.item(), (end - start)))
        vis.line(X=[epoch], Y=[loss.item()], win='loss',
                 opts=dict(title='train loss'), update='append')

        # ====4.3 start evalidating====
        model.eval()

        correct_prediction = 0.
        total = 0

        with torch.no_grad():
            for images, labels in valid_loader:
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

        if accuracy > best_accuracy:
            if not os.path.exists(config.checkpoint):
                os.mkdir(config.checkpoint)
            # save networks
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'best_accuracy': accuracy,
            }, os.path.join("%s-%.3f.pth") % (config.best_models, accuracy))
            print("save networks for epoch:", epoch)
            best_accuracy = accuracy


if __name__ == '__main__':

    ''' parse_argument
    parser = argparse.ArgumentParser(description='Plants Disease Detection')
    parser.add_argument("-n", "--num_classes", default=6, type=int)
    parser.add_argument("-e", "--epochs", default=20, type=int)
    parser.add_argument("--net", default='resnet50', type=str)
    parser.add_argument("-d", "--depth", default=50, type=int)
    parser.add_argument("-l", "--lr", default=1e-3, type=float)
    parser.add_argument("-b", "--batch_size", default=16, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("-mn", "--model_name", default='model', type=str)
    parser.add_argument("-mp", "--model_path", default='./model', type=str)
    parser.add_argument("-p", "--pretrained", default=True, type=bool)
    parser.add_argument("-pm", "--pretrained_model", default='./model/resnet50.pth', type=str)
    args = parser.parse_args()
    '''

    main()
