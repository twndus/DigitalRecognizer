'''
train.py
'''
import numpy as np

import torch, torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

import wandb

from dataset.mnistdataset import MNISTDataset
from model.model import SimpleMLP

# fix random seeds for reproducibility
SEED = 20231118 
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

    return tot_corr/datalen

def main():
    
    ## connection with wandb
    EPOCHS = 10
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 32

    config = {"epochs": EPOCHS, "batch_size": BATCH_SIZE, 
        "learning_rate": LEARNING_RATE}
    run = wandb.init(project='digital-recognizer', config=config)

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

    # model train
    device = torch.device('mps')

    model = SimpleMLP().to(device)
    model.init_params()
    model.train()

    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for e in range(EPOCHS):
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

        train_acc = acc(model, train_dataloader, len(mnist_train_data), device)
        print(f'[step {e}] loss: {tot_loss}, acc: {train_acc}')
        wandb.log({'accuracy': train_acc, 'loss': loss})
        
if __name__ == '__main__':
    main()
