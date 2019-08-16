import os, sys, argparse, time, json
from pathlib import Path
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import cosmo_data
from cosmo_models import get_model
from cosmo_train import train_model, eval_model, train
from parser import get_parser

## NUM_CLASS = 6
## 
## def train_model(exp_name, model, train_loader, test_loader, optim, critc,\
##                 device, writer, args):
##     best_acc = 0
##     model.train()
##     cum_instances = 0
##     cur_intervals = args.save_intervals if args.save_intervals else 0
##     for epoch in range(epochs):
##         tot_loss_train, tot_loss_test = 0, 0
##         start_time = time.time()
##         cnt = 0
##         for x,y in train_loader:
##             x,y = x.to(device).float(), y.to(device)
##             output = model(x)
##             loss   = critc(output, y)
##             tot_loss_train += loss.item()
##             cnt += y.size(0)
##             optim.zero_grad()
##             loss.backward()
##             optim.step()
##         writer.add_scalar('epoch_time', time.time()-start_time, epoch)
##         writer.add_scalar('train_loss', tot_loss_train/len(train_loader), epoch)
## 
##             
##         cum_instances += cnt
##         if args.save_intervals == None or cum_instances > cur_intervals:
##             cur_intervals += args.save_intervals if args.save_intervals else 0
##             acc, tot = 0,0
##             acc_per_class,  tot_per_class =\
##                 [np.zeros(NUM_CLASS) for _ in range(2)]
##             with torch.no_grad():
##                 for x,y in test_loader:
##                     x,y = x.to(device).float(), y.to(device)
##                     output = model(x)
##                     loss = critc(output, y)
##                     tot_loss_test += loss.item()
##                     tot += y.size(0)
##                     acc += (output.max(1)[1]==y).sum().item()
##                     np.add.at(tot_per_class, y.cpu().numpy(), 1)
##                     np.add.at(acc_per_class, y.cpu().numpy(),
##                               (output.max(1)[1]==y).cpu().numpy())
##             writer.add_scalar('test_loss', tot_loss_test/tot/len(test_loader), epoch)
##             writer.add_scalar('test_acc', acc/tot, epoch)
##             
##             # only save the best modeL
##             if acc > best_acc:
##                 best_acc = acc
##                 torch.save({
##                     'epoch':epoch,
##                     'model_state_dict': model.state_dict(),
##                     'optimizer_state_dict': optim.state_dict(),
##                     'validation_accuracy' : acc/tot,
##                     'per_class_validation_accuracy' : acc_per_class/tot_per_class
##                 },Path(args.save)/exp_name/('best.pt'))
## 

if __name__ == "__main__":
    NUM_CLASS = 6
    args = get_parser()
    
    ## setup hyper parameters, data loaders, optimizers and models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lr, batch_size, epochs = args.learning_rate, args.batch_size, args.epochs
    
    ## Get Data
    if args.train_dev_split is not None:
        train_loader, valid_loader, test_loader = cosmo_data.get_data(args.input_data,
                                                bsz=args.batch_size,
                                                num_workers=args.num_workers,
                                                pin_memory=args.no_pin_memory,
                                                    amount=args.train_dev_split, 
                                                                seed=args.randseed)
    elif args.use_subset_data is not None:
        subset_idx = None
        with open(args.use_subset_data, 'r') as f:
            subset_idx = json.load(f)
        train_loader, valid_loader, test_loader = cosmo_data.get_subset_data(args.input_data,
                                                    subset_idx,
                                                    bsz=args.batch_size,
                                                    num_workers=args.num_workers,
                                                    pin_memory=args.no_pin_memory,
                                                    )
    else:
        train_loader, test_loader = cosmo_data.get_data(args.input_data,\
                                                    bsz=args.batch_size,\
                                                    num_workers=args.num_workers,\
                                                    pin_memory=args.no_pin_memory,\
                                                    )

    model = get_model(args.arch, NUM_CLASS).to(device)
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
    main_exp_name = args.experiment_name
    writer = SummaryWriter(Path(args.tensorboard)/exp_name)
    (Path(args.tensorboard)/exp_name).mkdir(parents=True, exist_ok=True)
    (Path(args.save)/exp_name).mkdir(parents=True, exist_ok=True)
    (Path(args.output)/exp_name).mkdir(parents=True, exist_ok=True)
    if dist.get_rank() == 0:
        (Path(args.output)/main_exp_name).mkdir(parents=True, exist_ok=True)
    

    ## training process
    if not args.skip_train:
        print("valid_loader", len(valid_loader))
        train(exp_name, model, train_loader, valid_loader, optim, \
              critc, device, args, writer=writer)
        # train_model(exp_name, model, train_loader, test_loader, optim, \
        #             critc, device, writer, args)


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
            res = {"overall_accuracy"        : acc_opt1/tot,
                   "overall_accuracy_opt2"   : acc_opt2/tot,
                   "per_class_test_accuracy" : (acc_per_class_opt1/tot_per_class).tolist(),
                   "per_class_test_accuracy_opt2" : (acc_per_class_opt2/tot_per_class).tolist(),
                   "model_arch"              : args.arch,
                   "batch_size"              : batch_size,
                   "learning_rate"           : lr,
                   "optimizer"               : args.optim
                   }

            with open(Path(args.output)/main_exp_name/('output.json'), 'w') as f:
                json.dump(res, f)

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




