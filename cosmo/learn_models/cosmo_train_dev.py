import torch
import numpy as np
import cosmo_data
import cosmo_models
from torch.utils.tensorboard import SummaryWriter
torch.cuda.set_device(0)

learning_rate = 0.001
batch_size = 8

train_data, test_data = cosmo_data.get_data('/home/yren/data/cosmo_data/npy/',
                                            bsz=batch_size, num_workers=8, pin_memory=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = cosmo_models.ResNet3d(Bottleneck3d, [3,4,6,3], groups=32, width_per_group=4)
model.to(device)
optim = torch.optim.Adam(model.parameters(), lr = learning_rate)
critc = torch.nn.CrossEntropyLoss()

epochs = 100
exp_name = 'exp_' + 'resnet3d_lr' + str(learning_rate) + "_bsz_"+str(batch_size)
writer = SummaryWriter('./runs/'+exp_name+'/')

for epoch in range(epochs):
    print("epoch==",epoch)
    tot_loss_train, tot_loss_test = 0, 0
    for x,y in train_loader:
        x,y = x.to(device).float(), y.to(device)
        output = model(x)
        loss   = critc(output, y)
        tot_loss_train += loss.item()
        optim.zero_grad()
        loss.backward()
        optim.step()
    writer.add_scalar('train_loss', tot_loss_train/len(train_loader), epoch)
        
    with torch.no_grad():
        acc, tot = 0,0
        for x,y in test_loader:
            x,y = x.to(device).float(), y.to(device)
            output = model(x)
            loss = critc(output, y)
            tot_loss_test += loss.item()
            tot += y.size(0)
            acc += (output.max(1)[1]==y).sum().item()
    writer.add_scalar('test_loss', tot_loss_test/len(test_loader), epoch)
    writer.add_scalar('test_acc', acc/tot, epoch)

    if (epoch+1)%20 == 0:
        torch.save({
            'epoch':epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict()
        },'./save/'+exp_name+'_epoch'+str(epoch)+'.pt')
