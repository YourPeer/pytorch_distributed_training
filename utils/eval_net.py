import torch
def eval_net(model,testloader):
    model.eval()
    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        images = images.cuda()
        labels = labels.cuda()
        with torch.no_grad():
            outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    model.train()
    return 100 * correct // total