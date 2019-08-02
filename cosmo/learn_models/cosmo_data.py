from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import pandas as pd
import numpy as np
import collections
from pathlib import Path

## example usage:
#  data_dir = '/home/yren/data/cosmo_data/npy/'
#  train_data, test_data = Cosmo3D(data_dir, train=True), Cosmo3D(data_dir, train=False)

class Cosmo3D(Dataset):
    def __init__(self, base_folder, train=True, transform=None, target_transform=None, download=False):
        """
            base_folder contains *.npy files and train.csv and test.csv
        """
        super(Cosmo3D, self).__init__()
        base_folder = Path(base_folder)
        assert (base_folder.exists() and base_folder.is_dir())
        if train: 
            assert (base_folder/'train.csv').exists()
            self.df = pd.read_csv(base_folder/'train.csv')
        else:
            assert (base_folder/'test.csv').exists()
            self.df = pd.read_csv(base_folder/'test.csv')
        self.base_folder      = base_folder
        self.train            = train
        self.transform        = transform
        self.target_transform = target_transform
        if download:
            raise NotImplementedError
        
    def __getitem__(self, index):
        data_file = self.base_folder / self.df.iloc[index]['fname']
        x, y = np.load(data_file), int(self.df.iloc[index]['class'])
        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)
        return x,y

    def __len__(self):
        return len(self.df)

def np_norm(x): return np.expand_dims((x-8)/28.871186, axis=0)

def get_data(data_dir, bsz, num_workers=4, pin_memory=True,\
             amount=None, seed=None):
    """
        create train data loader and validation data loader
        support subset sampling by assigning "amount" 

        NOTE: to make sure the member models of an ensemble have the same subset,
        a random seed should need to be specified.
    """
    def shuffle_idx():
        idx = np.arange(len(train_data))
        np_rnd_state = np.random.get_state()
        np.random.seed(seed)
        np.random.shuffle(idx)
        np.random.set_state(np_rnd_state)
        return idx

    # data_dir = '/home/yren/data/cosmo_data/npy/'
    assert (Path(data_dir).exists() and Path(data_dir).is_dir())
    train_data = Cosmo3D(data_dir, transform=np_norm)
    test_data = Cosmo3D(data_dir, train=False, transform=np_norm)
    if not amount:
        # return all training data and all testing data
        train_loader = DataLoader(train_data, batch_size=bsz, \
                                shuffle=True, num_workers=num_workers, \
                                pin_memory=pin_memory)
        test_loader = DataLoader(test_data, batch_size=2*bsz)
        return train_loader, test_loader
    elif not isinstance(amount, collections.Sequence) or len(amount) == 1:
        # amount is a number or a sequence with one element
        # return x amount for training and n-x for dev
        if isinstance(amount, collections.Sequence): amount = amount[0]
        if amount < 1: amount = int(len(train_data)*amount)
        assert ( 0 < amount < len(train_data) )
        idx = shuffle_idx()
        train_loader = DataLoader(train_data, batch_size=bsz, \
                                  sampler=SubsetRandomSampler(idx[:amount]), \
                                  pin_memory=pin_memory,
                                  num_workers=num_workers)
        dev_loader   = DataLoader(train_data, batch_size=2*bsz, \
                           sampler=SubsetRandomSampler(idx[amount:]), \
                           pin_memory=pin_memory, num_workers=num_workers)
        test_loader = DataLoader(test_data, batch_size=2*bsz)
        return train_loader, dev_loader, test_loader
    else:
        # first number for train, 
        # second number for dev, and
        # third number (if present) for test
        train_amount, dev_amount = amount[0], amount[1]
        if train_amount < 1 : train_amount = int(len(train_data)*train_amount)
        if dev_amount < 1   : dev_amount = int(len(train_data)*dev_amount)
        #print(train_amount, dev_amount, len(train_data) )
        assert train_amount + dev_amount <= len(train_data), \
               "Train Amount : {0} and Dev Amount : {1} is greater than the \
               Data Size : {2}".format(train_amount, dev_amount,
                                       len(train_data) ) 
        idx = shuffle_idx()
        train_loader = DataLoader(train_data, batch_size=bsz, \
                           sampler=SubsetRandomSampler(idx[:train_amount]), \
                           pin_memory=pin_memory, num_workers=num_workers)
        dev_loader   = DataLoader(train_data, batch_size=2*bsz, \
                           sampler=SubsetRandomSampler(\
                               idx[train_amount:(train_amount+dev_amount)]), \
                           pin_memory=pin_memory, num_workers=num_workers)
        test_loader = DataLoader(test_data, batch_size=2*bsz)
        return train_loader, dev_loader, test_loader
