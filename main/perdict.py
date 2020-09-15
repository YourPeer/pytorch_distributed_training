import torch
print(torch.cuda.is_available())
from torchvision.datasets import CIFAR10
import torchvision.models as models
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
from model import Net
from utils import eval_net
import os
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transform = transforms.Compose([transforms.ToTensor(),normalize])
testset = CIFAR10(root='../data', train=False,download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,shuffle=True, num_workers=8)
model=Net().cuda()
state_dict = torch.load('../checkpoints/best_accuracy.pth')
model.load_state_dict(state_dict['state_dict'])
print("Beginning Testing")
correct = 0
total = 0
for data in testloader:
    images, labels = data
    images=images.cuda()
    labels=labels.cuda()
    with torch.no_grad():
        outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct // total))