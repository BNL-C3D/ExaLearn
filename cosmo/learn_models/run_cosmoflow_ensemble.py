import json
import subprocess
import active_learn_utils
import cosmo_data

num_stages = 3
rnd_seed   = [39, 17, 27]
num_gpus   = 4
exp_name   = "cosmoflow_ensemble"
frac       = 0.03

cmd = "-m torch.distributed.launch --nnodes 1 \
       --node_rank 0 --master_addr localhost --master_port 8989 \
       --nproc_per_node {np:d} \
       train_ensemble_distributed.py -n \
       {expname}_np_{np:d}_lr_1e4_bsz16_frac_{frac:.2f}_stage_{stage}_run{run:d} \
       -i /home/yren/data/cosmo_data/npy \
       -a cosmoflow -o adam -l 0.0001 -b 16 -e {epoch:d} \
       --num-workers 3 \
       --use-subset-data {idxfile}\
       --tensorboard ./runs/{expname}/ \
       --save-intervals 2000"

cmd0 = cmd.format(expname=exp_name, np=num_gpus, frac=frac, run=0, epoch=300,stage=1, idxfile="idx.json")
print(cmd0)

# data_dir = "/home/yren/data/cosmo_data/npy" 
# train_data = cosmo_data.Cosmo3D(data_dir, transform=cosmo_data.np_norm)
# train_idx = active_learn_utils.sample_given_budget_excluding([14 for _ in
#                                                               range(6)],
#                                                              train_data, [])
# valid_idx = active_learn_utils.sample_given_budget_excluding([94 for _ in
#                                                               range(6)],
#                                                              train_data,
#                                                              train_idx)
# with open('./idx.json','w') as f:
#     dic = {'train' : train_idx, 
#            'validation' : valid_idx}
#     json.dump(dic,f)

#print("trian idx", train_idx)
rtn = subprocess.run(['python'+" " + cmd0], shell=True)
print(rtn)
