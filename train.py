'''
train.py
'''
import numpy as np

import torch, torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset.mnistdataset import MNISTDataset
from model.model import SimpleMLP

# fix random seeds for reproducibility
SEED = 42#20231118 
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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

    print(f'acc: {tot_corr/datalen}')

def main():
    
    ## connection with wandb
    #run = wandb.init(project='digital-recognizer', config={"learning_rate": })

    # dataset
    mnist_train_data = MNISTDataset(path='data/train.csv', train=True, 
        transform=transforms.ToTensor())
    mnist_test_data = MNISTDataset(path='data/test.csv', train=False, 
        transform=transforms.ToTensor())

    # dataloader
    train_dataloader = DataLoader(dataset=mnist_train_data, batch_size=32, 
        shuffle=True)
    test_dataloader = DataLoader(dataset=mnist_test_data, batch_size=32, 
        shuffle=True)

    # model train
    epochs = 10
    learning_rate = 1e-3
    device = torch.device('mps')

    model = SimpleMLP().to(device)
    model.init_params()
    model.train()

    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for e in range(epochs):
        tot_loss = 0
        for b, (data, label) in enumerate(train_dataloader):

            data, label = data.float().to(device), label.float().to(device)
            pred = model.forward(data.view((-1, 28*28)))

            optim.zero_grad()
            loss = loss_fn(pred, label)
            loss.backward()
            optim.step()

            tot_loss += loss.item()
            #print(f'[step {e}-{b}] loss: {loss.item()}')

        print(f'[step {e}] loss: {tot_loss}')
        acc(model, train_dataloader, len(mnist_train_data), device)
        
if __name__ == '__main__':
    main()
