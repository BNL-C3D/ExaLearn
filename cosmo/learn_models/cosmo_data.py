from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import pandas as pd
import numpy as np
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
        assert(base_folder.exists() and base_folder.is_dir())
        if train: 
            assert(base_folder/'train.csv').exists()
            self.df = pd.read_csv(base_folder/'train.csv')
        else:
            assert(base_folder/'test.csv').exists()
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

def get_data(data_dir, bsz, num_workers=8, pin_memory=True,\
             amount=None, seed=None):
    """
        create train data loader and validation data loader
        support subset sampling by assigning "amount" 

        NOTE: to make sure the member models of an ensemble have the same subset,
        a random seed should need to be specified.
    """
    # data_dir = '/home/yren/data/cosmo_data/npy/'
    assert(Path(data_dir).exists() and Path(data_dir).is_dir())
    train_data = Cosmo3D(data_dir, transform=np_norm)
    test_data = Cosmo3D(data_dir, train=False, transform=np_norm)
    if not amount:
        train_loader = DataLoader(train_data, batch_size=bsz, \
                                shuffle=True, num_workers=num_workers, \
                                pin_memory=pin_memory)
        test_loader = DataLoader(test_data, batch_size=2*bsz)
    else:
        if amount < 1: amount = int(len(train_data)*amount)
        assert( 0 < amount < len(train_data) )
        idx = np.arange(len(train_data))
        np.random.seed(seed)
        np.random.shuffle(idx)
        train_loader = DataLoader(train_data, batch_size=bsz, \
                                  sampler=SubsetRandomSampler(idx[:amount]), \
                                  pin_memory=pin_memory)
        test_loader = DataLoader(test_data, batch_size=2*bsz)

    return train_loader, test_loader
