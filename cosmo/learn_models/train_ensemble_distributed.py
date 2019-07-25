import os, sys, argparse, time
from pathlib import Path
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import cosmo_data
from cosmo_models import CosmoNet, ResNet3d, Bottleneck3d

NUM_CLASS = 6

def get_model( arch ):
    if arch == "cosmoflow":
        return CosmoNet(NUM_CLASS)
    elif arch == "resnext3d":
        return ResNet3d(Bottleneck3d, [3,4,6,3], NUM_CLASS, groups=32, width_per_group=4)
    elif arch == "resnext3dsmall":
        return ResNet3d(Bottleneck3d, [1,2,3,2], NUM_CLASS, groups=32, width_per_group=4)
    else:
        print(arch, "not supported", file=sys.stderr)
        raise NotImplementedError

def train_model(exp_name, model, train_loader, test_loader, optim, critc,\
                device, writer, args):
    best_acc = 0
    model.train()
    for epoch in range(epochs):
        tot_loss_train, tot_loss_test = 0, 0
        start_time = time.time()
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
        writer.add_scalar('epoch_time', time.time()-start_time, epoch)
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
        
        # only save the best modeL
        if acc > best_acc:
            best_acc = acc
            torch.save({
                'epoch':epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict()
            },Path(args.save)/exp_name/('best.pt'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    required.add_argument("-n", "--experiment-name", help="experiment name",\
                          type=str, required=True)
    required.add_argument("-a", "--arch", help="select model architecture",
                          action='store',
                          choices=['cosmoflow', 'resnext3d', 'resnext3dsmall'],
                          type=str, required=True)
    required.add_argument("-o", "--optim", help="select optimizer",\
                          action='store', choices=['sgd', 'adam'],\
                          type=str, required=True)
    required.add_argument("-i", "--input-data", \
                          help="directory containing input data", type=str, required=True)
    required.add_argument("-l", "--learning-rate", help="learning rate",\
                          type=float, required=True)
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
    optional.add_argument("--use-subset-data", help="train on a subset of the \
                          training data.", type=float, default=None, required=False)
    optional.add_argument("--randseed", help="random seed for selecting a subset of the \
                          training data.", type=int, default=None, required=False)
    optional.add_argument('--local_rank', default=0, type=int)
    optional.add_argument('--skip_train', action='store_true', default=False)
    optional.add_argument('--backend', default='nccl', type=str,\
                          choices=['nccl', 'ddl'], required=False)
    args = parser.parse_args() 
    
    ## check if distributed 
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    args.gpu = 0
    args.world_size = 1
    if args.distributed:
        args.gpu = args.local_rank % torch.cuda.device_count()
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend=args.backend, init_method='env://')
        args.world_size = torch.distributed.get_world_size()
    else:
        args.world_size = torch.cuda.device_count()
    

    ## setup hyper parameters, data loaders, optimizers and models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lr, batch_size, epochs = args.learning_rate, args.batch_size, args.epochs
    train_loader, test_loader = cosmo_data.get_data(args.input_data,\
                                                    bsz=args.batch_size,\
                                                    num_workers=args.num_workers,\
                                                    pin_memory=args.no_pin_memory,\
                                                    amount=args.use_subset_data,\
                                                    seed=args.randseed)
    model = get_model(args.arch).to(device)
    if args.optim == "adam":
        optim = torch.optim.Adam(model.parameters(), lr=lr)
    elif args.optim == "sgd":
        optim = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        print(args.optim, "not supported", file=sys.stderr)
        raise NotImplementedError
    critc = torch.nn.CrossEntropyLoss()

    ## setup tensorboard to monitor the training process
    exp_name = args.experiment_name+"_rank_"+str(dist.get_rank()).zfill(4)
    writer = SummaryWriter(Path(args.tensorboard)/exp_name)
    (Path(args.tensorboard)/exp_name).mkdir(parents=True, exist_ok=True)
    (Path(args.save)/exp_name).mkdir(parents=True, exist_ok=True)
    

    ## training process
    if not args.skip_train:
        train_model(exp_name, model, train_loader, test_loader, optim, \
                    critc, device, writer, args)


    ## at the end of training, do ensemble eval
    ckpt = torch.load(Path(args.save)/exp_name/'best.pt')
    model.load_state_dict(ckpt['model_state_dict'])
    with torch.no_grad():
        model.eval()
        acc, acc_opt1, acc_opt2, tot = 0.0, 0.0, 0.0, 0
        acc_per_class, acc_per_class_opt1, acc_per_class_opt2, tot_per_class =\
            [np.zeros(NUM_CLASS) for _ in range(4)]
        for x,y in test_loader:
            x,y = x.to(device).float(), y.to(device)
            output = model(x)
            tot += y.size(0)
            
            # option 1, add sigmoid values directly
            y1 = output.clone()
            dist.reduce(y1, 0)
            if dist.get_rank() == 0:
                acc_opt1 += (y1.max(1)[1]==y).sum().item()

            # option 2, binary first and then add (0,1)s
            y2 = torch.zeros(output.shape, device=device)
            y2[range(y2.size(0)), output.max(1)[1]] = 1
            dist.reduce(y2, 0)
            if dist.get_rank() == 0:
                acc_opt2 += (y2.max(1)[1]==y).sum().item()

            # local acc
            acc += (output.max(1)[1]==y).sum().item()

            # acc per class
            np.add.at(tot_per_class, y.cpu().numpy(), 1)
            np.add.at(acc_per_class, y.cpu().numpy(),
                      (output.max(1)[1]==y).cpu().numpy())
            if dist.get_rank() == 0:
                np.add.at(acc_per_class_opt1, y.cpu().numpy(),
                          (y1.max(1)[1]==y).cpu().numpy())
                np.add.at(acc_per_class_opt2, y.cpu().numpy(),
                          (y2.max(1)[1]==y).cpu().numpy())

        if dist.get_rank() == 0:
            print("opt 1 acc = ", acc_opt1 / tot)
            print("opt 2 acc = ", acc_opt2 / tot)
            print("tot per class", tot_per_class)
            print("per class opt1 = "," ".join('{:.4f}'.format(_) for _ in
                                               acc_per_class_opt1/tot_per_class))

            print("per class opt2 = "," ".join('{:.4f}'.format(_) for _ in
                                                acc_per_class_opt2/tot_per_class))


        print("rank {0} has acc {1}".format(dist.get_rank(), acc/tot))
        print("rank {0} has acc_per_class {1}".format(dist.get_rank(),\
                                                      " ".join('{:.4f}'.format(_) for _ in
                                                      acc_per_class/tot_per_class)))

        ## count the number per class
        train_tot_per_class = np.zeros(NUM_CLASS)
        for x,y in train_loader:
            np.add.at(train_tot_per_class, y.numpy(), 1)
        print("rank: {0} train tot per_class {1}".format(dist.get_rank(),
                                                         train_tot_per_class))



