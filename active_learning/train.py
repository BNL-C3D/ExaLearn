import torch 
import torchvision
import numpy as np
from torchvision import transforms
import torch.nn as nn
import matplotlib.pyplot as plt

def plot_selected( selected_idx, dataset ):
    assert( 4*8 == len(selected_idx) )
    fig, axes = plt.subplots( 4, 8, figsize=(8,4))
    axes = axes.flatten()
    for ax, idx in zip(axes, selected_idx):
        ax.imshow(np.asarray(dataset[idx][0]), cmap='Greys')
        ax.set_axis_off()
    return fig

def train_model(model, lr, num_epoch, train_loader, test_loader, quiet=True, device='cuda'):
    model.to(device)
    optim = torch.optim.SGD(model.parameters(),lr=lr)
    critr = nn.CrossEntropyLoss()
    test_acc = []
    for epoch in range(num_epoch):
        tot_, crt_ = 0, 0
        for i, (images, labels) in enumerate(train_loader):  
            # Move tensors to the configured device
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = critr(outputs, labels)
            
            # Backward and optimize
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            _, prd_ = torch.max(outputs.data, 1)
            tot_ += labels.size(0)
            crt_ += (prd_ == labels).sum().item()
    
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        if not quiet:
            print('Accuracy of the network after epoch {:02d} : {:.4f} % for train and {:.4f} % for test'.\
                  format(1+epoch, 100*crt_/tot_, 100 * correct / total))
        test_acc.append(correct/total)
    return test_acc

def active_learn(model_func, learner_func, num_para, lr, num_epoch, batch_size, train_loader_fixed, test_loader ):
    track_selection = []
    track_accuracy = []
    actlearn = learner_func()
    ## define initial number of samples, number of samples selected each iter and budget size
    # n_seed, n_iter, n_tot, num_iter = 32, 32, 1024, 31
    n_seed, n_iter, n_tot, num_iter = num_para
    assert( n_tot-n_seed == num_iter*n_iter)
    init_data = actlearn.random_select_k_samples(train_loader_fixed, n_seed)
    tmp_sampler = torch.utils.data.SubsetRandomSampler(list(actlearn.selected))
    tmp_train_loader = torch.utils.data.DataLoader(dataset=train_loader_fixed.dataset, 
                                                   batch_size=batch_size,
                                                   sampler=tmp_sampler)
    track_selection.append(init_data)
    model = model_func()
    acc   = []
    tmp_acc = train_model(model, lr, num_epoch, tmp_train_loader, test_loader)
    acc.append(tmp_acc)
    for _ in range(num_iter):
        newly_selected = actlearn.select_k_samples(model, train_loader_fixed, n_iter)
        track_selection.append(newly_selected)
        tmp_sampler = torch.utils.data.SubsetRandomSampler(list(actlearn.selected))
        tmp_train_loader = torch.utils.data.DataLoader(dataset=train_loader_fixed.dataset, 
                                                   batch_size=batch_size,
                                                  sampler=tmp_sampler)
        model = model_func()
        tmp_acc = train_model(model, lr, num_epoch, tmp_train_loader, test_loader)
        acc.append(tmp_acc)
        print("---------{:2d}:{:.4f}----------".format(_+1, 100*tmp_acc[-1]))
        
    plt.plot([x[-1] for x in acc])
    return acc
