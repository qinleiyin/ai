import os
import csv
import shutil

path_csv = r'D:\data\kaggle_cifar10\trainLabels.csv'


def read_label_file(data_dir, label_file, train_dir, valid_ratio):
    with open(os.path.join(data_dir, label_file)) as fd:
        fd.readline()
        reader = csv.reader(fd)

        idx_label = dict(reader)

    labels = idx_label.values
    n_train_valid = len(os.listdir(os.path.join(data_dir, train_dir)))
    n_train = int(n_train_valid * (1 - valid_ratio))
    assert 0 < n_train < n_train_valid
    return n_train // n_train_valid, idx_label


def mkdir_if_not_exist(path):
    if not os.path.exists(os.path.join(*path)):
        os.makedirs(os.path.join(*path))


def reorg_train_valid(data_dir, train_dir, input_dir, n_train_per_label, idx_label):
    label_count = {}
    for img in os.listdir(os.path.join(data_dir, train_dir)):
        path_img = os.path.join(data_dir, train_dir, img)
        idx = img[:-4]
        label = idx_label[idx]
        mkdir_if_not_exist([data_dir, input_dir, "train_valid", label])
        shutil.copy(path_img, os.path.join(data_dir, input_dir, "train_valid", label))
        if label not in label_count or label_count[label] < n_train_per_label:
            mkdir_if_not_exist([data_dir, input_dir, "train", label])
            shutil.copy(path_img, os.path.join(data_dir, input_dir, "train", label))
            label_count[label] = label_count.get(label, 0) + 1
        else:
            mkdir_if_not_exist([data_dir, input_dir, "valid", label])
            shutil.copy(path_img, os.path.join(data_dir, input_dir, "valid", label))


def reorg_test(data_dir, test_dir, input_dir):
    mkdir_if_not_exist([data_dir, input_dir, "test", "unknown"])
    for img in os.listdir(os.path.join(data_dir, test_dir)):
        shutil.copy(os.path.join(data_dir, test_dir, img), os.path.join(data_dir, input_dir, "test", "unknown"))


def reorg_cifar10_data(data_dir, label_file, train_dir, test_dir, input_dir,
                       valid_ratio):
    n_train_per_label, idx_label = read_label_file(data_dir, label_file,
                                                   train_dir, valid_ratio)
    reorg_train_valid(data_dir, train_dir, input_dir, n_train_per_label,
                      idx_label)
    reorg_test(data_dir, test_dir, input_dir)

train_dir, test_dir, batch_size = 'train', 'test', 1
data_dir, label_file = r'D:\data\kaggle_cifar10', 'trainLabels.csv'
input_dir, valid_ratio = 'train_valid_test', 0.1
reorg_cifar10_data(data_dir, label_file, train_dir, test_dir, input_dir,
                   valid_ratio)

import torchvision.transforms as transforms
transforms_train = transforms.Compose([transforms.Resize(40),
                                       transforms.RandomResizedCrop(32, scale=(0.64, 1.0), ratio=(1.0,1.0)),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.4914, 0.4822, 0.4465],
                                                            [0.2023, 0.1994, 0.2010])
                                      ])
transforms_test = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize([0.4914, 0.4822, 0.4465],
                                                            [0.2023, 0.1994, 0.2010])
                                     ])

import torch
from PIL import Image

type_dict = {
    "truck": 0,
    "ship": 1,
    "horse": 2,
    "frog": 3,
    "dog": 4,
    "deer": 5,
    "cat": 6,
    "bird": 7,
    "automobile": 8,
    "airplane": 9
}


class ImageFolderDataset(torch.utils.data.Dataset):

    def __init__(self, data_dir, transforms):
        self.img_list = []
        image_size = (3, 256, 256)
        for label_dir in os.listdir(data_dir):
            label_path = os.path.join(data_dir, label_dir)
            for img_file in os.listdir(label_path):
                path_img_file = os.path.join(data_dir, label_dir, img_file)
                img_pil = Image.open(path_img_file).convert("RGB").resize((image_size[2], image_size[1]),
                                                                          Image.BILINEAR)
                img = transforms(img_pil)
                self.img_list.append((img, type_dict[label_dir]))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        return self.img_list[index]

train_ds = ImageFolderDataset(os.path.join(data_dir, input_dir, "train"), transforms_train)
valid_ds = ImageFolderDataset(os.path.join(data_dir, input_dir, "valid"), transforms_test)

train_iteror = torch.utils.data.DataLoader(train_ds, batch_size, shuffle=True)
valid_iteror = torch.utils.data.DataLoader(valid_ds, batch_size, shuffle=True)

import torch.nn as nn


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1_conv=False, strides=1, **kwargs):
        super(self.__class__, self).__init__(**kwargs)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        if use_1x1_conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None

    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)

        #         print(y.size(), x.size())
        y += x
        return self.relu(y)


class Flatten(nn.Module):
    def __init__(self, **kwargs):
        super(self.__class__, self).__init__(*kwargs)

    def forward(self, x):
        return x.view(x.size(0), -1)


def res_blk(in_features, out_features, num, first_blk=False):
    lays = []
    for i in range(num):
        if i == 0 and not first_blk:
            lays.append(Residual(in_features, out_features, use_1x1_conv=True, strides=2))
        else:
            lays.append(Residual(out_features, out_features))

    return lays


def resnet18():
    b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU())
    b2 = nn.Sequential(*res_blk(64, 64, 2, first_blk=True))
    b3 = nn.Sequential(*res_blk(64, 128, 2))
    b4 = nn.Sequential(*res_blk(128, 256, 2))
    b5 = nn.Sequential(*res_blk(256, 512, 2))

    net = nn.Sequential(b1, b2, b3, b4, b5, nn.AdaptiveMaxPool2d((1, 1)), Flatten(), nn.Linear(512, 10))
    return net


def init_weights(lay):
    if type(lay) in [nn.Linear, nn.Conv2d]:
        nn.init.xavier_uniform_(lay.weight)


def get_net():
    net = resnet18()
    net.apply(init_weights)
    return net


loss = nn.CrossEntropyLoss()


def train(net, train_iter, valid_iter, epocks, loss, lr, lr_period, lr_decay):
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    opt_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay, gamma=0.1)
    for epock in range(epocks):
        if epock != 0 and epock % lr_period == 0:
            opt_scheduler.step()

        for x, y in train_iter:
            optimizer.zero_grad()
            #             print(x.size())
            y_hat = net(x)
            print(y)
            print(y_hat)
            l = loss(y_hat, y)
            print(l)
            l.backward()
            optimizer.step()


train(get_net(), train_iteror, valid_iteror, 1, loss, 0.1, 80, 0.1)