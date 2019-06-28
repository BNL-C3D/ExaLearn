import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product
import random

def create_data_set( data_dir, split=0.9, seed=None ) :
    data_dir = Path(data_dir)
    assert(data_dir.exists() and data_dir.is_dir())
    fname, para1, para2, para3, hsh, subidx =[], [], [], [], [], []
    for f in data_dir.iterdir():
        tmp = f.stem.split('_')
        fname.append(f.name)
        para1.append(float(tmp[1]))
        para2.append(float(tmp[2]))
        para3.append(float(tmp[3]))
        hsh.append(tmp[4])
        subidx.append(int(tmp[5]))
    data = {'fname':fname, 'omega-m': para1, 'omega-L': para2, 'sigma_8': para3, 'uniq-id':hsh, 'subidx': subidx}
    df = pd.DataFrame.from_dict(data)
    dic = {}
    for idx, x in enumerate(product(df['omega-m'].unique(), df['omega-L'].unique())):
        dic[x] =idx
    df['class'] = [dic[(x,y)] for x,y in zip(df['omega-m'], df['omega-L'])]
    if not seed:
        seed = random.randint(0,1<<20)
    prv_state = np.random.get_state()
    np.random.seed(seed)
    idx = np.arange(len(df))
    np.random.shuffle(idx)
    np.random.set_state(prv_state)
    split = int(len(df)*split)
    df_train, df_test = df.iloc[idx[:split]], df.iloc[idx[split:]]
    return df_train, df_test

## Examples to use the code: 
# data_dir = '/home/yren/data/cosmo_data/npy/'
# df_train, df_test = create_data_set(data_dir, 0.9, seed=8)
# df_train.to_csv(data_dir+'train.csv')
# df_test.to_csv(data_dir+'test.csv')
