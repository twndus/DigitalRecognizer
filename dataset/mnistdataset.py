'''
DataLoader for DigitalRecognizer

 - data downloaded from Kaggle Competition DigitalRecognizer
   (https://www.kaggle.com/competitions/digit-recognizer/data)
 - data format: csv.
'''
import os, random
import numpy as np
import pandas as pd

import torch, torchvision
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

class MNISTDataset(Dataset):

    def __init__(self, path, train, transform):
        
        self.data = pd.read_csv(path)
        self.train = train
        self.transform = transform
        
        if self.train:
            self.X = np.reshape(self.data.iloc[:,1:].values, (-1, 28, 28))
            self.y = self.data['label'].values
        else:
            self.X = np.reshape(self.data.values, (-1, 28, 28))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.train:
            y = self.y[idx]
            ret = {'img': x, 'label': y}
        else:
            ret = {'img': x}
        return ret

if __name__ == '__main__':
    
    mnist_train_data = MNISTDataset(path='data/train.csv', train=True, transform=ToTensor())
    mnist_test_data = MNISTDataset(path='data/test.csv', train=False, transform=ToTensor())
    
    idx = random.randint(0,len(mnist_test_data))
    print(f'train length: {len(mnist_train_data)}, {idx}st item: {mnist_train_data[idx]}')
    print(f'test length: {len(mnist_test_data)}, {idx}st item: {mnist_test_data[idx]}')
