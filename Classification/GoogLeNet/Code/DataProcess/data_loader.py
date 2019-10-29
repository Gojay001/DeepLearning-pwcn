from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from PIL import Image
import os
import random
import numpy as np
import json

from DataProcess.data_config import config


def get_folders(train_data, test_data):

    train_folders = [os.path.join(train_data, label)
                     for label in os.listdir(train_data)
                     if os.path.isdir(os.path.join(train_data, label))
                     ]
    test_folders = [os.path.join(test_data, label)
                    for label in os.listdir(test_data)
                    if os.path.isdir(os.path.join(test_data, label))
                    ]
    # random.seed(1)
    # random.shuffle(train_folders)
    # random.shuffle(test_folders)
    return train_folders, test_folders


class MyDataSet(Dataset):

    def __init__(self, folders, transforms=None, train=True, test=False):

        self.folders = folders
        self.imgs = []
        self.labels = []
        labels_dict = np.array(range(len(folders)))
        labels_dict = dict(zip(folders, labels_dict))

        for folder in folders:
            imgs = [os.path.join(folder, img) for img in os.listdir(folder)]
            labels = [labels_dict[folder] for img in os.listdir(folder)]
            self.imgs.extend(imgs)
            self.labels.extend(labels)

        if transforms is None:
            if test or not train:
                self.transforms = T.Compose([
                    # T.Resize(config.img_weight, config.img_height),
                    T.CenterCrop((config.img_weight, config.img_height)),
                    T.ToTensor()])
                # T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])

            else:
                self.transforms = T.Compose([
                    # T.Resize(config.img_weight, config.img_height),
                    T.CenterCrop((config.img_weight, config.img_height)),
                    T.RandomRotation(30),
                    T.RandomHorizontalFlip(),
                    T.RandomVerticalFlip(),
                    T.RandomAffine(45),
                    T.ToTensor()])
                # T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])

        else:
            self.transforms = transforms

    def __getitem__(self, index):

        filename = self.imgs[index]
        label = self.labels[index]
        img = Image.open(filename)
        img = self.transforms(img)
        return img, label

    def __len__(self):
        return len(self.imgs)


class TestDataSet(Dataset):

    def __init__(self, test_folder, transforms=None):

        self.folder = test_folder
        self.imgs = os.listdir(self.folder)

        if transforms is None:
            self.transforms = T.Compose([
                # T.Resize(config.img_weight, config.img_height),
                T.CenterCrop((config.img_weight, config.img_height)),
                T.ToTensor()])
            # T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])])
        else:
            self.transforms = transforms

    def __getitem__(self, index):

        filename = os.path.join(self.folder, self.imgs[index])
        img = Image.open(filename)
        img = self.transforms(img)
        return img

    def __len__(self):
        return len(self.imgs)


def get_img_dict(folder):

    images = os.listdir(folder)
    # images.sort(key=lambda x:int(x[:-4]))

    imgs_dict = []
    for img in images:
        img_dict = {'image_id': img}
        imgs_dict.append(img_dict)
    return imgs_dict


if __name__ == '__main__':
    train_folders, test_folders = get_folders(
        '../data/traffic-sign/train/', '../data/traffic-sign/valid/')

    # print basic info of datasets
    #-----------------------------
    train_datasets = MyDataSet(train_folders)
    img, label = train_datasets[0]
    print(img.size(), label)
    print(train_datasets.__len__())

    # print images and labels
    #------------------------------
    # train_datasets = MyDataSet(train_folders, transforms=None)
    # train_loader = DataLoader(dataset=train_datasets,
    #                           batch_size=1, shuffle=True)
    # for images, labels in train_loader:
    #     img = T.ToPILImage()(images[0])
    #     img.show()
    #     print('the label is: ', labels)
    #     exit()

    # print images dict to json
    #-------------------------------
    # test_datasets = TestDataSet('../data/traffic-sign/test/00000/')
    # data = get_img_dict('../data/traffic-sign/test/00000/')
    # with open('../submit/data.json', 'w') as f:
    #     json.dump(data, f)
    # with open('../submit/data.json', 'r') as f:
    #     data = json.load(f)
    # temp = data[4]
    # temp["disease_class"] = '1'
    # print(data[4])
