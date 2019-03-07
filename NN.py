import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms


trainset = torchvision.datasets.MNIST(root='./data', download=True, train=True, transform=transforms.ToTensor())

trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', download=True, train=False, transform=transforms.ToTensor())

testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=True, num_workers=2)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1,8,3, padding=1)
        self.conv2 = nn.Conv2d(8,16,5)
        self.conv3 = nn.Conv2d(16,20,5)
        self.pool1 = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(20 * 6 * 6,120)
        self.fc2 = nn.Linear(120,10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 20*6*6)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = Net()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net.to(device)
print('your device is:', device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=1e-3)
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)


epochs = 100
for epoch in range(epochs):

    running_loss = 0.
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        #print('inputs:', inputs.shape, 'labels:', labels.shape)
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        #print('outputs:', outputs.shape, 'labels:', labels.shape)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')



correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the {total} test images: {100 * correct / total}%')

torch.save({
            #'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            #'loss': loss,
            }, 'data/MNIST_NN.pt')

print('successfully saved')