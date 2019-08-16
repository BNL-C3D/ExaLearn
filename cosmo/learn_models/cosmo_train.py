import torch
import sys, json, time
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np
import cosmo_data
from cosmo_models import get_model
import meters
from meters import AvgMeter
from parser import get_parser


def train_model(model, train_loader, optim, critc, device, meters=None):
    r"""
        train model for 1 epoch
        meters are list of AvgMeters

        return: tot_train_loss
    """
    model.train()
    for x, y in train_loader:
        x, y = x.to(device).float(), y.to(device)
        output = model(x)
        loss   = critc(output, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        if meters:  # update avg meters
            for mt in meters: mt(output, y)

def eval_model(model, data_loader, device, meters=None):
    r"""
        evaluate model on data_loader (validate, or test)
        return: accuracy
    """
    model.eval()
    with torch.no_grad():
        for x,y in data_loader:
            x,y = x.to(device).float(), y.to(device)
            output = model(x)
            if meters: 
                for mt in meters: mt(output, y)


def train(exp_name, model, train_loader, valid_loader, optim, critc,\
                device, args, writer=None):

    best_acc = 0
    train_meters = [AvgMeter(meters.accuracy),
                    AvgMeter(meters.mk(critc))]

    valid_meters = [AvgMeter(meters.accuracy),
                    AvgMeter(meters.mk(critc)),
                    AvgMeter(meters.accuracy_per_class)]

    cum_instances = 0
    cur_intervals = args.save_intervals if args.save_intervals else 0
    for epoch in range(args.epochs):
        start_time = time.time()
        for mt in train_meters: mt.reset()
        train_model(model, train_loader, optim, critc, device, meters=train_meters)
        if writer:
            writer.add_scalar('epoch_time', time.time()-start_time, epoch)
            writer.add_scalar('train_acc' , train_meters[0].eval(), epoch)
            writer.add_scalar('train_loss', train_meters[1].eval(), epoch)

            
        cum_instances += train_meters[0].cnt
        if args.save_intervals is None or cum_instances > cur_intervals:
            cur_intervals += args.save_intervals if args.save_intervals else 0
            for mt in valid_meters: mt.reset()
            eval_model(model, valid_loader, device, meters=valid_meters)
            # print("eval")
            # for mt in valid_meters:
            #     print(mt.cum, mt.cnt, mt.eval())
            if writer:
                writer.add_scalar('validate_acc', valid_meters[0].eval(), epoch)
                writer.add_scalar('validate_loss', valid_meters[1].eval(), epoch)
            
            # only save the best modeL
            if valid_meters[0].eval() > best_acc:
                best_acc = valid_meters[0].eval()
                torch.save({
                    'epoch':epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    'validation_accuracy' : best_acc,
                    'per_class_validation_accuracy' : valid_meters[2].eval()
                },Path(args.save)/exp_name/('best.pt'))

if __name__ == "__main__":
    NUM_CLASS = 6
    args = get_parser()

    exp_name = args.experiment_name
    lr, batch_size, epochs = args.learning_rate, args.batch_size, args.epochs
    train_loader, valid_loader, test_loader = cosmo_data.get_data(args.input_data,
                                                bsz=args.batch_size,
                                                num_workers=args.num_workers,
                                                pin_memory=args.no_pin_memory,
                                                    amount=args.train_dev_split, 
                                                                seed=args.randseed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = get_model(args.arch, NUM_CLASS)
    critc  = torch.nn.CrossEntropyLoss()
    model.to(device)

    if args.optim == "adam":
        optim = torch.optim.Adam(model.parameters(), lr=lr)
    elif args.optim == "sgd":
        optim = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        print(args.optim, "not supported", file=sys.stderr)
        raise NotImplementedError

    writer = SummaryWriter(Path(args.tensorboard)/exp_name)
    (Path(args.tensorboard)/exp_name).mkdir(parents=True, exist_ok=True)
    (Path(args.save)/exp_name).mkdir(parents=True, exist_ok=True)
    (Path(args.output)/exp_name).mkdir(parents=True, exist_ok=True)

    ## Train:
    train(exp_name, model, train_loader, valid_loader, optim, critc, \
          device, args, writer=writer)

    ## Test:
    ckpt = torch.load(Path(args.save)/exp_name/'best.pt')
    model.load_state_dict(ckpt['model_state_dict'])
    with torch.no_grad():
        tot_loss_test, tot, acc = 0, 0, 0
        acc_per_class,  tot_per_class =\
            [np.zeros(NUM_CLASS) for _ in range(2)]
        for x,y in test_loader:
            x,y = x.to(device).float(), y.to(device)
            output = model(x)
            loss = critc(output, y)
            tot_loss_test += loss.item()
            tot += y.size(0)
            acc += (output.max(1)[1]==y).sum().item()
            np.add.at(tot_per_class, y.cpu().numpy(), 1)
            np.add.at(acc_per_class, y.cpu().numpy(),
                      (output.max(1)[1]==y).cpu().numpy())

    # print("testing")
    # print(acc_per_class)
    # print(tot_per_class)
    print("Saving:")
    res = {"overall_accuracy"        : acc/tot,
           "validation_accuracy"     : ckpt['validation_accuracy'],
           "per_class_test_accuracy" : (acc_per_class/tot_per_class).tolist(),
           "peak_epoch"              : ckpt['epoch'],
           "model_arch"              : args.arch,
           "batch_size"              : batch_size,
           "learning_rate"           : lr,
           "optimizer"               : args.optim
           }

    with open(Path(args.output)/exp_name/('output.json'), 'w') as f:
        json.dump(res, f);
