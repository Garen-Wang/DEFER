"""
这部分代码是用来训练 resnet50 早退点的
"""
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

from datetime import datetime as dt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet50WithEarlyExit(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, early_exit_enabled=False):
        super().__init__()
        self.in_planes = 64
        self.early_exit_enabled = early_exit_enabled
        self.early_exit_threshold = 0.5
        self.exit_loss_weights = [0.9, 0.7, 0.5, 0.3]

        # self.layer0 = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        # )
        # self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        # self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        # self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        ))
        self.layers.append(self._make_layer(block, 64, num_blocks[0], stride=1))
        self.layers.append(self._make_layer(block, 128, num_blocks[1], stride=2))
        self.layers.append(self._make_layer(block, 256, num_blocks[2], stride=2))
        self.layers.append(self._make_layer(block, 512, num_blocks[3], stride=2))

        # self.avgpool = nn.AvgPool2d(4)
        # self.flatten = nn.Flatten(1)
        # self.linear = nn.Linear(512*block.expansion, num_classes)

        # self.exit1 = self._make_exit(64 * 4 * 8 * 8, num_classes)
        # self.exit2 = self._make_exit(128 * 4 * 4 * 4, num_classes)
        # self.exit3 = self._make_exit(256 * 4 * 2 * 2, num_classes)
        # self.exit1 = self._make_exit(4096 * block.expansion, num_classes)
        # self.exit2 = self._make_exit(2048 * block.expansion, num_classes)
        # self.exit3 = self._make_exit(1024 * block.expansion, num_classes)
        # self.exit4 = self._make_exit(512 * block.expansion, num_classes)
        self.exits = nn.ModuleList()
        self.exits.append(self._make_exit(4096 * block.expansion, num_classes))
        self.exits.append(self._make_exit(2048 * block.expansion, num_classes))
        self.exits.append(self._make_exit(1024 * block.expansion, num_classes))
        self.exits.append(self._make_exit(512 * block.expansion, num_classes))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _make_exit(self, in_features, num_classes):
        return nn.Sequential(
            nn.AvgPool2d(4),
            nn.Flatten(1),
            nn.Linear(in_features, num_classes),
        )
    
    def exit_criterion_top1(self, x):
        with torch.no_grad():
            pk = nn.functional.softmax(x, dim=-1)
            top1 = torch.max(pk)
            return top1 > self.early_exit_threshold

    def forward(self, x):
        outputs = []
        x = self.layers[0](x)
        x = self.layers[1](x)
        outputs.append(self.exits[0](x))
        if self.early_exit_enabled and self.exit_criterion_top1(outputs[0]):
            return outputs[0]
        x = self.layers[2](x)
        outputs.append(self.exits[1](x))
        if self.early_exit_enabled and self.exit_criterion_top1(outputs[1]):
            return outputs[1]
        x = self.layers[3](x)
        outputs.append(self.exits[2](x))
        if self.early_exit_enabled and self.exit_criterion_top1(outputs[2]):
            return outputs[2]
        x = self.layers[4](x)
        outputs.append(self.exits[3](x))
        if self.early_exit_enabled:
            return outputs[3]
        return outputs


def save_model(model, path, file_prefix='', seed=None, epoch=None, opt=None, loss=None):
    timestamp = dt.now().strftime("%Y-%m-%d_%H%M%S")
    filenm = file_prefix + '-' + timestamp
    save_dict ={'timestamp': timestamp,
                'model_state_dict': model.state_dict()
                }

    if seed is not None:
        save_dict['seed'] = seed
    if epoch is not None:
        save_dict['epoch'] = epoch
        filenm += f'{epoch:03d}'
    if opt is not None:
        save_dict['opt_state_dict'] = opt.state_dict()
    if loss is not None:
        save_dict['loss'] = loss

    if not os.path.exists(path):
        os.makedirs(path)

    filenm += '.pth'
    file_path = os.path.join(path, filenm)

    torch.save(save_dict, file_path)

    print("Saved to:", file_path)
    return file_path

def train_backbone(model, train_loader, valid_loader, criterion, epochs):
    lr = 0.001
    exp_decay_rates = [0.99, 0.999]
    backbone_params = [
        {'params': model.layers.parameters()},
        {'params': model.exits[-1].parameters()},
    ]
    optimizer = optim.Adam(backbone_params, lr=lr, betas=exp_decay_rates)

    best_val_loss = [1.0, '']
    for epoch in range(epochs):
        model.train()
        print(f'train backbone, epoch {epoch}')

        train_loss_sum = 0.
        correct = 0
        total = 0
        for idx, (inputs, target) in enumerate(train_loader):
            inputs, target = inputs.to(device), target.to(device)
            outputs = model(inputs)
            loss = criterion(outputs[-1], target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            _, predicted = outputs[-1].max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            if (idx + 1) % 20 == 0:
                print('[train] batch_idx=%d, loss=%.3f, acc=%.3f%%' % (idx, train_loss_sum / (idx+1), 100.0 * correct / total))
            

        model.eval()

        valid_loss_sum = 0.
        correct = 0
        total = 0
        with torch.no_grad():
            for idx, (inputs, target) in enumerate(valid_loader):
                inputs, target = inputs.to(device), target.to(device)
                outputs = model(inputs)
                loss = criterion(outputs[-1], target)

                valid_loss_sum += loss.item()
                _, predicted = outputs[-1].max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

                if (idx + 1) % 20 == 0:
                    print('[val] batch_idx=%d, loss=%.3f, acc=%.3f%%' % (idx, valid_loss_sum / (idx+1), 100.0 * correct / total))

        valid_loss_avg = valid_loss_sum / len(valid_loader)
        print('[val] loss=%.3f, acc=%.3f%%' % (valid_loss_sum / total, 100.0 * correct / total))

        checkpoint = save_model(model, './checkpoints', f'backbone_{epoch+1}', opt=optimizer)

        if valid_loss_avg < best_val_loss[0]:
            best_val_loss[0] = valid_loss_avg
            best_val_loss[1] = ''
    print(f'best val loss: {best_val_loss[0]}')
    return checkpoint
    
def joint_train(model, train_loader, valid_loader, criterion=nn.CrossEntropyLoss(), backbone_epochs=50, joint_epochs=100, pretrain_backbone=True):
    if pretrain_backbone:
        checkpoint = train_backbone(model, train_loader, valid_loader, criterion, backbone_epochs)
        model.load_state_dict(torch.load(checkpoint)['model_state_dict'])
    else:
        print('????????')
        model.load_state_dict(torch.load('./checkpoints/backbone_50-2023-10-23_230355.pth')['model_state_dict'])
        pass

    lr = 0.001
    exp_decay_rates = [0.99, 0.999]
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=exp_decay_rates)

    best_val_losses = [[1.0, 1.0, 1.0, 1.0], '']
    for epoch in range(joint_epochs):
        print(f'joint train, epoch {epoch}')
        model.train()

        train_losses_sum = [0.0, 0.0, 0.0, 0.0]
        corrects = [0, 0, 0, 0]
        total = 0
        for batch_idx, (inputs, target) in enumerate(train_loader):
            inputs, target = inputs.to(device), target.to(device)
            outputs = model(inputs)

            losses = [model.exit_loss_weights[i] * criterion(outputs[i], target) for i in range(len(outputs))]
            optimizer.zero_grad()
            for loss in losses[:-1]:
                loss.backward(retain_graph=True)
            losses[-1].backward()
            optimizer.step()

            for i in range(4):
                train_losses_sum[i] += losses[i].item()
                _, predicted = outputs[i].max(1)
                corrects[i] += predicted.eq(target).sum().item()
            total += target.size(0)

            if (batch_idx + 1) % 20 == 0:
                print('[train] batch_idx=%d:' % (batch_idx))
                for i in range(4):
                    print('    exits[%d]: loss=%.3f, acc=%.3f%%' % (i, train_losses_sum[i] / (batch_idx+1), 100.0 * corrects[i] / total))
        
        model.eval()
        valid_losses_sum = [0.0, 0.0, 0.0, 0.0]
        corrects = [0, 0, 0, 0]
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, target) in enumerate(valid_loader):
                inputs, target = inputs.to(device), target.to(device)
                outputs = model(inputs)

                for i in range(4):
                    loss = criterion(outputs[i], target)
                    valid_losses_sum[i] += loss.item()
                    _, predicted = outputs[i].max(1)
                    corrects[i] += predicted.eq(target).sum().item()
                total += target.size(0)
                
                if (batch_idx + 1) % 20 == 0:
                    print('[val] batch_idx=%d:' % (batch_idx))
                    for i in range(4):
                        print('    exits[%d]: loss=%.3f, acc=%.3f%%' % (i, valid_losses_sum[i] / (batch_idx+1), 100.0 * corrects[i] / total))

            valid_losses_avg = [0.0, 0.0, 0.0, 0.0]
            for i in range(4):
                valid_losses_avg[i] = valid_losses_sum[i] / len(valid_loader)
                print('[val] exits[%d]: loss=%.3f, acc=%.3f%%' % (i, valid_losses_avg[i], 100.0 * corrects[i] / total))
            checkpoint = save_model(model, './checkpoints', f'early_{epoch+1}', opt=optimizer)

            if sum(valid_losses_avg) < sum(best_val_losses[0]):
                best_val_losses[0] = valid_losses_avg
                best_val_losses[1] = checkpoint

    print(f'best val losses: {best_val_losses[0]}')


def main():
    # dataset cifar10
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_set = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=128, shuffle=True, num_workers=6)

    test_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=100, shuffle=False, num_workers=6)
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

    # model
    model = ResNet50WithEarlyExit(Bottleneck, [3, 4, 6, 3])
    model.to(device)
    # x = torch.randn(1, 3, 32, 32)
    # y = model(x)
    # print(y.size())

    joint_train(model, train_loader, test_loader, pretrain_backbone=False)

if __name__ == '__main__':
    main()