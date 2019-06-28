from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import numpy as np

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


