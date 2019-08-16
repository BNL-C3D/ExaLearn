import numpy as np

class AvgMeter():
    r"""
        example: 
            1. calculate average loss
                ```
                avg = AvgMeter(mk(nn.CrossEntropyLoss()))
                xx = torch.rand((5,10)).float()
                yy = torch.FloatTensor(5).uniform_(0, 10).long()
                avg(yy.shape[0], xx, yy)
                ```
    """
    
    def __init__(self, func):
        self.cnt, self.cum = None, None
        self.func = func
    
    def update(self, *args, **kwags):
        bsz, tmp = self.func(*args, **kwags) 
        self.cnt = (bsz + self.cnt) if self.cnt is not None else bsz
        self.cum = (tmp + self.cum) if self.cum is not None else tmp
        return self.cum / self.cnt
    
    def __call__(self, *args, **kwags):
        return self.update(*args, **kwags)

    def reset(self):
        self.cnt, self.cum = None, None

    def eval(self):
        return self.cum / self.cnt if self.cnt is not None else 0


def mk(loss):
    r"""
        Avg Meter Adapter for Loss functions
        make loss function to return (bsz, value)
    """

    def rt(output, y):
        return y.shape[0], loss(output,y)

    return rt


def accuracy(output, y):
    r"""
        output is a tensor of size (Bsz, ClsSz)
        y is a tensor of size (Bsz)
        return overall accuracy counts
    """
    return y.shape[0], (output.max(1)[1] == y ).sum().item()


def accuracy_per_class(output, y):
    r"""
        output is a tensor of size (Bsz, ClsSz)
        y is a tensor of size (Bsz)
        return: accuracy counts per class
    """
    num_class = output.size(1)
    cnt_per_class = np.zeros(num_class) ## num of classes
    acc_per_class = np.zeros(num_class) ## num of classes
    np.add.at(cnt_per_class, y.cpu().numpy(), 1)
    np.add.at(acc_per_class, y.cpu().numpy(), (output.max(1)[1] == y).cpu().numpy())
    return cnt_per_class, acc_per_class

