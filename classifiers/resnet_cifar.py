'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


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


class ResNet(nn.Module):
    def __init__(self, block, in_channel, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        z = out.view(out.size(0), -1)
        out = self.linear(z)
        return z, out, z

    def predict(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        z = out.view(out.size(0), -1)
        out = self.linear(z)
        x = torch.argmax(out, dim=1)
        return z, x, z

    def predict_proba(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        logits = torch.softmax(out, dim=1)
        return logits
    
    def get_feature(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out        


def ResNet18(in_channel=3):
    return ResNet(BasicBlock, in_channel, [2, 2, 2, 2])


def ResNet34(in_channel=3):
    return ResNet(BasicBlock, in_channel, [3, 4, 6, 3])


def ResNet50(in_channel=3):
    return ResNet(Bottleneck, in_channel, [3, 4, 6, 3])


def ResNet101(in_channel=3):
    return ResNet(Bottleneck, in_channel, [3, 4, 23, 3])


def ResNet152(in_channel=3):
    return ResNet(Bottleneck, in_channel, [3, 8, 36, 3])

def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())


if __name__ == '__main__':
  model = ResNet18()
  # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
  #                     weight_decay=1e-4)
  # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
  # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
  optimizer = torch.optim.SGD(model.parameters(), lr=0.1,
                        momentum=0.9, weight_decay=5e-4)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
  optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
  scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma= 0.1)
  criterion = nn.CrossEntropyLoss()
  if torch.cuda.is_available():
    model = model.cuda()
    # criterion = criterion.cuda()
  
  is_train = True
  if is_train:
    loss_list = []
    num_epochs = 200
    for epoch in range(num_epochs):
      train_loss = 0
      for batch_id, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        x, y = x.cuda(), y.cuda()
        output = model(x)

        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        train_loss += loss
      
      correct = 0
      total = 0
      with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
      
      acc = 100.*correct/total
      scheduler.step()

      print("Epoch: {}, LOSS: {}, ACC: {}".format(epoch, train_loss/batch_id, acc))
      loss_list.append(train_loss/batch_id)
    
    plt.plot(range(len(loss_list)), loss_list, label='Training loss')
    plt.legend()
    plt.show()
    plt.savefig("output/loss_cifar10.png")
    plt.close()

    torch.save(model.state_dict(), 'output/CNN_{}.pth'.format(dataset_name))
    print("Training done.")