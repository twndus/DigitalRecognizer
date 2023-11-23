'''
test.py
'''
import os, random, string

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

from dataset.mnistdataset import MNISTDataset
from model.model import SimpleMLP

def acc(model, dataloader, datalen, device):
    # performance
    model.eval()
    
    tot_corr = 0
    with torch.no_grad():
        for data, label in dataloader:

            data, label = data.float().to(device), label.float().to(device)
            pred = model.forward(data.view((-1, 28*28)))

            y_pred = torch.argmax(pred,1)
            y_label = torch.argmax(label,1)

            correct = (y_pred == y_label).sum().item()
            tot_corr += correct

    return tot_corr/datalen


def main(model_path):

    BATCH_SIZE = 256
    device = torch.device('mps')
    
    # dataset
    mnist_train_data = MNISTDataset(path='data/train.csv', train=True, 
        transform=transforms.ToTensor())
    mnist_test_data = MNISTDataset(path='data/test.csv', train=False, 
        transform=transforms.ToTensor())

    # dataloader
    train_dataloader = DataLoader(dataset=mnist_train_data, 
        batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(dataset=mnist_test_data, 
        batch_size=BATCH_SIZE, shuffle=True)
    
    # get model
    model = SimpleMLP().to(device)
    model.load_state_dict(torch.load(model_path))
    train_acc = acc(model, train_dataloader, len(mnist_train_data), device)
    print(f"train acc: {train_acc}")

if __name__ == '__main__':
    model_save_dir = './output'
    modelname = 'mnist-mlp-qxint.pt'
    main(os.path.join(model_save_dir, modelname))
