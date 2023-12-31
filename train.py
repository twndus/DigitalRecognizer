'''
train.py
'''
import os, random, string 

import numpy as np

import torch, torchvision
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

import wandb

from dataset.mnistdataset import MNISTDataset
from model.model import SimpleMLP, SimpleCNN

# fix random seeds for reproducibility
SEED = 20231118 
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def acc(model, dataloader, datalen, device, model_2d):
    # performance
    model.eval()
    
    tot_corr = 0
    with torch.no_grad():
        for data, label in dataloader:
            data, label = data.float().to(device), label.float().to(device)
            if model_2d:
                pred = model.forward(data)
            else:
                pred = model.forward(data.view((-1, 28*28)))

            y_pred = torch.argmax(pred,1)
            y_label = torch.argmax(label,1)

            correct = (y_pred == y_label).sum().item()
            tot_corr += correct

    return tot_corr/datalen

def main(modelname, model_2d, monitor, model_save, model_save_path):
    
    ## connection with wandb
    EPOCHS = 10 
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 32

    if monitor:
        config = {"epochs": EPOCHS, "batch_size": BATCH_SIZE, 
            "learning_rate": LEARNING_RATE}
        run = wandb.init(project='digital-recognizer', config=config)

    # dataset
    mnist_train_data = MNISTDataset(path='data/train.csv', train=True, 
        transform=transforms.ToTensor())
    mnist_test_data = MNISTDataset(path='data/test.csv', train=False, 
        transform=transforms.ToTensor())
    
    # split train to train and val
    mnist_train_data, mnist_val_data = random_split(mnist_train_data, [.8,.2])

    # dataloader
    train_dataloader = DataLoader(dataset=mnist_train_data, 
        batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(dataset=mnist_val_data, 
        batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(dataset=mnist_test_data, 
        batch_size=BATCH_SIZE, shuffle=True)

    # model train
    device = torch.device('mps')

    if modelname == 'SimpleMLP':
        model = SimpleMLP().to(device)
    elif modelname == 'SimpleCNN':
        model = SimpleCNN(hconv_channels=[32, 64]).to(device)
    else:
        raise ValueError(f"{modelname} not implemented. check model/model.py")

    model.init_params()
    model.train()

    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for e in range(EPOCHS):
        tot_loss = 0
        for b, (data, label) in enumerate(train_dataloader):
            data, label = data.float().to(device), label.float().to(device)
            if model_2d:
                pred = model.forward(data)
            else:
                pred = model.forward(data.view((-1, 28*28)))

            optim.zero_grad() # reset gradients to avoid confusing.
            loss = loss_fn(pred, label) # compute loss with logit and 1hot targe
            loss.backward() # compute partial derivatives
            optim.step() # update with optimizer

            tot_loss += loss.item()

        train_acc = acc(model, train_dataloader, len(mnist_train_data), device,
            model_2d)
        val_acc = acc(model, val_dataloader, len(mnist_val_data), device, model_2d)
        print(f'[step {e}] loss: {tot_loss}, train acc: {train_acc:.4f}, val acc: {val_acc:.4f}')

        if monitor:
            wandb.log({'train_accuracy': train_acc, 'val_accuracy':val_acc, 'loss': loss})

    if model_save:
        torch.save(model.state_dict(), model_save_path)
        
if __name__ == '__main__':

    # modelname
    ##'SimpleMLP', 'SimpleCNN'
    modelname = 'SimpleCNN'
    print(f'model: {modelname}')
    
    if modelname == 'SimpleMLP':
        model_2d = False
    else:
        model_2d = True

    # wandb monitoring
    monitor = False#True#False

    # model save
    model_save = False#True
    model_save_dir = './output'
    savename = f"mnist-{modelname}-{''.join(random.sample(string.ascii_lowercase, 5))}.pt"
    os.makedirs('./output', exist_ok=True)

    main(modelname, model_2d, monitor, model_save, 
         os.path.join(model_save_dir, savename))
