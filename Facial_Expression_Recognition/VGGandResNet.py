'''Train Fer2013 with PyTorch.'''
# 10 crop for data enhancement
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import transforms as transforms
import numpy as np
import os
import argparse
import utils
from fer import FER2013
from torch.autograd import Variable
from models import *
import os

log_path = 'resnet34_learning_rate.txt'
log_path_Train = 'resnet34_train_results.txt'
log_path_PublicTest = 'resnet34_public_test_results.txt'
log_path_PrivateTest = 'resnet34_private_test_results.txt'
f = open(log_path, 'a')
f_Train = open(log_path_Train, 'a')
f_PublicTest = open(log_path_PublicTest, 'a')
f_PrivateTest = open(log_path_PrivateTest, 'a')
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    global Train_acc
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    if epoch > learning_rate_decay_start and learning_rate_decay_start >= 0:
        frac = (epoch - learning_rate_decay_start) // learning_rate_decay_every
        decay_factor = learning_rate_decay_rate ** frac
        current_lr = opt.lr * decay_factor
        utils.set_lr(optimizer, current_lr)  # set the decayed rate
    else:
        current_lr = opt.lr
    print('learning_rate: %s' % str(current_lr))
    f.write(str(current_lr) + '\n')
    torch.cuda.empty_cache()

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        utils.clip_gradient(optimizer, 0.1)
        optimizer.step()
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        utils.progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), correct.__float__()/total, correct, total))
    Train_acc = correct.__float__()/total
    print(str(Train_acc))
    f_Train.write(str(Train_acc)+'\n')


def PublicTest(epoch):
    global PublicTest_acc
    global best_PublicTest_acc
    global best_PublicTest_acc_epoch
    net.eval()
    PublicTest_loss = 0
    correct = 0
    total = 0
    torch.cuda.empty_cache()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(PublicTestloader):
            bs, ncrops, c, h, w = np.shape(inputs)
            inputs = inputs.view(-1, c, h, w)
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            # inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            outputs = net(inputs)
            outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops
            loss = criterion(outputs_avg, targets)
            PublicTest_loss += loss.data
            _, predicted = torch.max(outputs_avg.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            utils.progress_bar(batch_idx, len(PublicTestloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                               % (PublicTest_loss / (batch_idx + 1), correct.__float__() / total, correct, total))

    # Save checkpoint.
    PublicTest_acc = correct.__float__()/total
    f_PublicTest.write(str(PublicTest_acc) + '\n')
    print(str(PublicTest_acc))
    if PublicTest_acc > best_PublicTest_acc:
        print('Saving..')
        print("best_PublicTest_acc: %0.3f" % PublicTest_acc)
        state = {
            'net': net.state_dict() if torch.cuda.is_available() else net,
            'acc': PublicTest_acc,
            'epoch': epoch,
        }
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(state, os.path.join(path, 'PublicTest_model.t7'))
        best_PublicTest_acc = PublicTest_acc
        best_PublicTest_acc_epoch = epoch


def PrivateTest(epoch):
    global PrivateTest_acc
    global best_PrivateTest_acc
    global best_PrivateTest_acc_epoch
    net.eval()
    PrivateTest_loss = 0
    correct = 0
    total = 0
    torch.cuda.empty_cache()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(PrivateTestloader):
            bs, ncrops, c, h, w = np.shape(inputs)
            inputs = inputs.view(-1, c, h, w)
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            # inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            outputs = net(inputs)
            outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops
            loss = criterion(outputs_avg, targets)
            PrivateTest_loss += loss.item()
            _, predicted = torch.max(outputs_avg.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            utils.progress_bar(batch_idx, len(PublicTestloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (PrivateTest_loss / (batch_idx + 1), correct.__float__() / total, correct, total))
    # Save checkpoint.
    PrivateTest_acc = correct.__float__() / total
    print(str(PrivateTest_acc))
    f_PrivateTest.write(str(PrivateTest_acc) + '\n')
    if PrivateTest_acc > best_PrivateTest_acc:
        print('Saving..')
        print("best_PrivateTest_acc: %0.3f" % PrivateTest_acc)
        state = {
            'net': net.state_dict() if torch.cuda.is_available() else net,
	        'best_PublicTest_acc': best_PublicTest_acc,
            'best_PrivateTest_acc': PrivateTest_acc,
    	    'best_PublicTest_acc_epoch': best_PublicTest_acc_epoch,
            'best_PrivateTest_acc_epoch': epoch,
        }
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(state, os.path.join(path,'PrivateTest_model.t7'))
        best_PrivateTest_acc = PrivateTest_acc
        best_PrivateTest_acc_epoch = epoch


if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    parser = argparse.ArgumentParser(description='PyTorch Fer2013 CNN Training')
    parser.add_argument('--model', type=str, default='ResNet34', help='VGG19 ResNet18 34 50')
    parser.add_argument('--dataset', type=str, default='FER2013', help='FER2013')
    parser.add_argument('--bs', default=1024, type=int, help='batch size')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    opt = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    best_PublicTest_acc = 0  # best PublicTest accuracy
    best_PublicTest_acc_epoch = 0
    best_PrivateTest_acc = 0  # best PrivateTest accuracy
    best_PrivateTest_acc_epoch = 0
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    learning_rate_decay_start = 80  # 50
    learning_rate_decay_every = 5  # 5
    learning_rate_decay_rate = 0.9  # 0.9

    cut_size = 44
    total_epoch = 250

    path = os.path.join(opt.model)
    print(opt.model)

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(44),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.TenCrop(cut_size),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    ])

    # batch_size=opt.bs

    trainset = FER2013(split='Training', transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
    PublicTestset = FER2013(split='PublicTest', transform=transform_test)
    PublicTestloader = torch.utils.data.DataLoader(PublicTestset, batch_size=32, shuffle=False)
    PrivateTestset = FER2013(split='PrivateTest', transform=transform_test)
    PrivateTestloader = torch.utils.data.DataLoader(PrivateTestset, batch_size=32, shuffle=False)

    # Model
    if opt.model == 'VGG19':
        net = VGG('VGG19')
    elif opt.model == 'VGG16':
        net = VGG('VGG16')
    elif opt.model == 'VGG13':
        net = VGG('VGG13')
    elif opt.model == 'VGG16':
        net = VGG('VGG16')
    elif opt.model == 'ResNet18':
        net = ResNet18()
    elif opt.model == 'ResNet34':
        net = ResNet34()
    elif opt.model == 'ResNet50':
        net = ResNet50()

    if opt.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir(path), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(os.path.join(path, 'PrivateTest_model.t7'))

        net.load_state_dict(checkpoint['net'])
        best_PublicTest_acc = checkpoint['best_PublicTest_acc']
        best_PrivateTest_acc = checkpoint['best_PrivateTest_acc']
        best_PrivateTest_acc_epoch = checkpoint['best_PublicTest_acc_epoch']
        best_PrivateTest_acc_epoch = checkpoint['best_PrivateTest_acc_epoch']
        start_epoch = checkpoint['best_PrivateTest_acc_epoch'] + 1
    else:
        print('==> Building model..')

    if use_cuda:
        net.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)

    for epoch in range(start_epoch, total_epoch):
        train(epoch)
        PublicTest(epoch)
        PrivateTest(epoch)

        print("best_PublicTest_acc: %0.3f" % best_PublicTest_acc)
        print("best_PublicTest_acc_epoch: %d" % best_PublicTest_acc_epoch)
        print("best_PrivateTest_acc: %0.3f" % best_PrivateTest_acc)
        print("best_PrivateTest_acc_epoch: %d" % best_PrivateTest_acc_epoch)
