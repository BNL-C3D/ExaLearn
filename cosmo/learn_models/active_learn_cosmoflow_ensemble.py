import json
import subprocess
import numpy as np
from pathlib import Path
from active_learn_utils import calc_budget_per_class, sample_given_budget_excluding
import cosmo_data

# CHANGE save intervels! #

def output_index_file(filename, *, train_idx, valid_idx):
    with open(filename,'w') as f:
        dic = {'train' : train_idx, 'validation' : valid_idx}
        json.dump(dic,f)


def load_index_file(filename):
    with open(filename, 'r') as f:
        dic = json.load(filename)
    return dic['train'], dic['validation']


def retrieve_weights(filename):
    """ Path('./output/'+expname_template.format(expname=exp_name, np=num_gpus,
                                                     stage=1, budget=stage_bgts[0],
                                                     run=nth_run) +
                                                     '/output.json')
    """
    with open(filename, 'r') as f:
        prv_output = json.load(f)
        weights = prv_output['per_class_valid_accuracy']
        weights = 1 / (np.asarray(weights)+np.finfo(float).eps)
        return weights


num_classes = 6
stage_bgts  = [120,45,45,45]
num_stages  = len(stage_bgts)
nth_run     = 0
rnd_seed    = [39, 17, 27, 12]
num_gpus    = 4
exp_name    = "cosmoflow_ensemble"
output_fld  = './output/test_active_learning/'
data_dir    = "/home/yren/data/cosmo_data/npy" 
expname_template = "{expname}_np_{np:d}_lr_1e4_bsz16_stage_{stage}_bgt_{budget}_run{run:d}"
idxfile_template = "idxfile_stage_{:d}.json"
cmd_template = \
       "-m torch.distributed.launch --nnodes 1 \
       --node_rank 0 --master_addr localhost --master_port 8989 \
       --nproc_per_node {np:d} \
       train_ensemble_distributed.py \
       -n {expname} \
       -i {input} \
       -a cosmoflow -o adam -l 0.0001 -b 16 -e {epoch:d} \
       --output {output}\
       --num-workers 3 \
       --use-subset-data {idxfile}\
       --tensorboard ./runs/{expname}/ \
       --save-intervals 1000"

train_data  = cosmo_data.Cosmo3D(data_dir, transform=cosmo_data.np_norm)
valid_idx   = sample_given_budget_excluding([100 for _ in range(6)], train_data, [])
train_idx   = []

for kth_stage in range(num_stages):
    current_exp = expname_template.format(expname=exp_name, np=num_gpus,
                                       stage=kth_stage,
                                       budget=stage_bgts[kth_stage],
                                       run=nth_run)

    #### GET BUDGET PER CLASS ####
    if kth_stage == 0:
        # initially uniformly pick
        budget = calc_budget_per_class(stage_bgts[0], [1.0/num_classes for _ in
                                          range(num_classes)]);
    else:
        # biased sample on inaccurate classes
        prv_exp_name = expname_template.format(expname=exp_name, 
                                               np=num_gpus,
                                               stage=kth_stage-1,
                                               budget=stage_bgts[kth_stage-1],
                                               run=nth_run)
        prv_output_filename = Path(output_fld)/prv_exp_name/'output.json'
        #prv_output_filename = Path(output_fld)/(expname_template.format(expname=exp_name, np=num_gpus,
        #                                         stage=kth_stage-1, budget=stage_bgts[kth_stage-1],
        #                                         run=nth_run) + '/output.json')
        weights = retrieve_weights(prv_output_filename)
        budget  = calc_budget_per_class(stage_bgts[kth_stage], weights)
    
    new_train_idx   = sample_given_budget_excluding(budget, train_data,
                                                    valid_idx+train_idx)
    train_idx = train_idx + new_train_idx 
    idxfile   = idxfile_template.format(kth_stage)
    output_index_file(idxfile, train_idx=train_idx, valid_idx=valid_idx)

    print("Launching {}th training".format(kth_stage))

    #### Launch Distributed Ensemble Training
    cmd = cmd_template.format(expname=current_exp, output=output_fld, input=data_dir,
                              np=num_gpus, budget=stage_bgts[kth_stage], run=nth_run,
                              epoch=15/(sum(stage_bgts[:kth_stage+1])/2000),
                              #epoch=10,
                              stage=kth_stage, idxfile=idxfile)
    rtn = subprocess.run(['python'+" " + cmd], shell=True)

    print("{}th_stage completed".format(kth_stage))

