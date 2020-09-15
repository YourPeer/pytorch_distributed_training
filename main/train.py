import torch
from torchvision.datasets import CIFAR10
import torchvision.models as models
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
from model import Net
from utils import eval_net
import os
import argparse
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.backends.cudnn as cudnn
parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')
parser.add_argument('--data',default='../data', help='path to dataset')
parser.add_argument('--checkpoint',default='../checkpoints/best_accuracy.pth',help='path to checkpoint')

def create_dataloader(data_dir,batch_size=32,num_workers=2):
    #data transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.ToTensor(),normalize])
    #load dataset
    trainset=CIFAR10(data_dir, train=True, transform=transform, target_transform=None, download=False)
    testset = CIFAR10(data_dir, train=False,download=False, transform=transform)
    #change to dataloader
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False, num_workers=num_workers)
    return trainloader,testloader

def load_model(fine_turn=False,checkpoint_dir=None):
    model=Net()
    if fine_turn:
        state_dict=torch.load(checkpoint_dir)
        model.load_state_dict(state_dict['state_dict'])
    return model

def optimizer_and_lossfn(model,learning_rate=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    return criterion,optimizer



def train_one_epoch(model,trainloader,testloader,optimizer,criterion,epoch,accuracy):
    running_loss=0
    for i, data in enumerate(trainloader):
        # get the inputs
        inputs, labels = data
        # warp them in Variable
        inputs, labels =inputs.cuda(), labels.cuda()
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        outputs = model(inputs)
        # loss
        loss = criterion(outputs, labels)
        # backward
        loss.backward()
        # update weights
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        if i % 10 == 9:  # print every 200 mini-batches
            eval_accuracy=eval_net(model, testloader).item()
            print('[%d, %5d] loss: %.3f  accuracy:%.3f' %(epoch + 1, i + 1, running_loss / 20,accuracy))
            running_loss = 0.0
            if eval_accuracy>accuracy:
                accuracy=eval_accuracy
                dict={'state_dict':model.state_dict(),'accuracy':accuracy}
                torch.save(dict,'../checkpoints/best_accuracy.pth')
    return accuracy

def train(args):
    #nproc_per_node ???????????????,?????????? GPU ??

    model = load_model(fine_turn=False, checkpoint_dir='../checkpoints/best_accuracy.pth')
    model.cuda()
    criterion, optimizer = optimizer_and_lossfn(model)
    criterion.cuda()
    cudnn.benchmark = True
    trainloader, testloader = create_dataloader(args.data)

    accuracy=0
    model.train()
    for i,epoch in enumerate(range(1)):
        accuracy=train_one_epoch(model,trainloader,testloader,optimizer,criterion,epoch,accuracy)
        print('The '+str(i)+' epoch accuracy is: '+str(accuracy))

def main():
    args = parser.parse_args()
    import time
    start=time.time()
    train(args)
    end=time.time()
    print("Training time is: "+str(end-start))
    # mp.spawn(train, nprocs=1, args=(1, args))

if __name__=="__main__":
    main()
    print("Finished Training")