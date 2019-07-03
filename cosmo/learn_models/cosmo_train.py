import torch
import sys
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np
import cosmo_data
from cosmo_models import CosmoNet, ResNet3d, Bottleneck3d
import argparse

def get_model( arch ):
    if arch == "cosmoflow":
        return CosmoNet()
    elif arch == "resnext3d":
        return ResNet3d(Bottleneck3d, [3,4,6,3], groups=32, width_per_group=4)
    else:
        print(arch, "not supported", file=sys.stderr)
        raise NotImplementedError

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    required.add_argument("-n", "--experiment-name", help="experiment name", type=str, required=True)
    required.add_argument("-a", "--arch", help="select model architecture",
                          action='store', choices=['cosmoflow', 'resnext3d'],
                          type=str, required=True)
    required.add_argument("-o", "--optim", help="select optimizer",
                          action='store', choices=['SGD', 'Adam'],
                          type=str, required=True)
    required.add_argument("-i", "--input-data", help="directory containing input data", type=str, required=True)
    required.add_argument("-l", "--learning-rate", help="learning rate", type=float, required=True)
    required.add_argument("-b", "--batch-size", help="batch size", type=int, required=True)
    required.add_argument("-e", "--epochs", help="number of epochs", type=int, required=True)
    optional.add_argument("--no-pin-memory", help="Not using pin memory",
                          action='store_false', default=True, required=False)
    optional.add_argument("--gpu-device-id", help="GPU device id [=0]", type=int,
                          required=False, default=0)
    optional.add_argument("--num-workers", help="number of data loading workers [=8]",
                          type=int, required=False, default=8)
    optional.add_argument("--save", help="directory to save model checkpoints",
                          type=str, default="./save/", required=False)
    optional.add_argument("--tensorboard", help="directory to save tensorboard \
                          summary", type=str, default="./runs/", required=False)
    args = parser.parse_args() 

    print("pin memory = ", args.no_pin_memory)

    learning_rate, batch_size = args.lr, args.batch_size
    torch.cuda.set_device(args.gpu_device_id)
    train_loader, test_loader = cosmo_data.get_data(args.input_data,
                                                bsz=args.batch_size,
                                                num_workers=args.num_workers,
                                                pin_memory=no_pin_memory)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = get_model(args.arch)
    model.to(device)
    if args.optim == "Adam":
        optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim == "SGD":
        optim = torch.optim.SDG(model.parameters(), lr=args.lr)
    else:
        print(args.optim, "not supported", file=sys.stderr)
        raise NotImplementedError
    critc = torch.nn.CrossEntropyLoss()
    exp_name = args.experiment_name
    epochs = args.epochs
    writer = SummaryWriter(Path(args.tensorboard)/exp_name)

    for epoch in range(epochs):
        tot_loss_train, tot_loss_test = 0, 0
        cnt = 0
        for x,y in train_loader:
            x,y = x.to(device).float(), y.to(device)
            output = model(x)
            loss   = critc(output, y)
            tot_loss_train += loss.item()
            cnt += y.size(0)
            optim.zero_grad()
            loss.backward()
            optim.step()
        writer.add_scalar('train_loss', tot_loss_train/len(train_loader), epoch)
            
        acc, tot = 0,0
        with torch.no_grad():
            for x,y in test_loader:
                x,y = x.to(device).float(), y.to(device)
                output = model(x)
                loss = critc(output, y)
                tot_loss_test += loss.item()
                tot += y.size(0)
                acc += (output.max(1)[1]==y).sum().item()
        writer.add_scalar('test_loss', tot_loss_test/tot/len(test_loader), epoch)
        writer.add_scalar('test_acc', acc/tot, epoch)
    
        if (epoch+1)%20 == 0:
            torch.save({
                'epoch':epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict()
            },Path(args.save)/exp_name/('epoch'+str(epoch)+'.pt'))
