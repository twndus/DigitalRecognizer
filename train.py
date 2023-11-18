'''
train.py
'''
import numpy as np

import torch, torchvision
from torch.utils.data import DataLoader

from dataset.mnistdataset import MNISTDataset
from torchvision.transforms import ToTensor

# fix random seeds for reproducibility
SEED = 20231118 
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def main():
    
    ## connection with wandb
    #run = wandb.init(project='digital-recognizer', config={"learning_rate": })

    # dataset
    mnist_train_data = MNISTDataset(path='data/train.csv', train=True, transform=ToTensor())
    mnist_test_data = MNISTDataset(path='data/test.csv', train=False, transform=ToTensor())

    # dataloader
    train_dataloader = DataLoader(dataset=mnist_train_data, batch_size=4, shuffle=True)
    test_dataloader = DataLoader(dataset=mnist_test_data, batch_size=32, shuffle=True)

    # model train
    pass

if __name__ == '__main__':
    main()
