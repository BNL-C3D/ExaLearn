import torch 
import torchvision
import numpy as np
from torchvision import transforms
import torch.nn as nn
import random

class ActiveLearn():
    def __init__(self, seed=None):
        self.selected = set()
        if not seed: 
            seed = random.randint(1,1<<20)
        np.random.seed(seed)
    
    
    def entropy(self, x):
        x = torch.sigmoid(x)
        x = x/torch.sum(x,dim=1).unsqueeze(-1)
        x = - x * torch.log(x)
        x = torch.sum(x,dim=1)
        return x
    
    def random_select_k_samples(self, data_loader, k=None):
        cnt = 0
        bsz = data_loader.batch_size
        if not k: k = bsz
        idx = np.arange(len(data_loader.dataset))
        np.random.shuffle(idx)
        ans = []
        for i in idx:
            if cnt == k: break;
            if i in self.selected: continue
            ans.append(i)
            self.selected.add(i)
            cnt += 1
        return ans
    
    def select_k_samples(self, model, data_loader, k=None):
        device=next(iter(model.parameters())).device
        ent = []
        cnt = 0
        bsz = data_loader.batch_size
        if not k: k = bsz
        with torch.no_grad():
            model.eval()
            for x,y in data_loader:
                cnt += len(x)
                x = x.to(device)
                z = model(x)
                ent.append(self.entropy(z))
            model.train()
        
        ary = np.empty(cnt)
        for idx, a in enumerate(ent):
            ary[idx*bsz:min(cnt, (idx+1)*bsz)] = a.cpu().data.numpy()
        
        res = np.argsort(ary)
        cnt = 0
        ans = []
        for idx in res[::-1]:
            if idx not in self.selected:
                self.selected.add(idx)
                ans.append(idx)
                cnt += 1
                if cnt == k: break
        
        return ans

class RandomLearn():
    def __init__(self, seed=None):
        self.selected = set()
        if not seed: 
            seed = random.randint(1,1<<20)
        np.random.seed(seed)
    
    def random_select_k_samples(self, data_loader, k=None):
        cnt = 0
        bsz = data_loader.batch_size
        if not k: k = bsz
        idx = np.arange(len(data_loader.dataset))
        np.random.shuffle(idx)
        ans = []
        for i in idx:
            if cnt == k: break;
            if i in self.selected: continue
            ans.append(i)
            self.selected.add(i)
            cnt += 1
        return ans
    
    def select_k_samples(self, model, data_loader, k=None):
        device=next(iter(model.parameters())).device
        cnt = 0
        bsz = data_loader.batch_size
        if not k: k = bsz
        idx = np.arange(len(data_loader.dataset))
        np.random.shuffle(idx)
        
        ans = []
        for i in idx:
            if cnt == k: break;
            if i in self.selected: continue
            ans.append(i)
            self.selected.add(i)
            cnt += 1
        return ans

