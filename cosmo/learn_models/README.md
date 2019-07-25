## accuracy improvement w.r.t. amount of training data

## accuracy improvement w.r.t. number of models in ensemble

## accuracy disparity among classes
rank 0 has acc 0.8846153846153846
rank 1 has acc 0.8717948717948718
rank 2 has acc 0.9775641025641025
rank 3 has acc 0.6698717948717948

opt 1 acc =  0.9230769230769231
opt 2 acc =  0.9294871794871795

tot per class [49. 60. 55. 56. 49. 43.]
rank: 0 train tot per_class [ 84.  94.  91.  90. 100. 102.]
rank: 1 train tot per_class [ 84.  94.  91.  90. 100. 102.]
rank: 2 train tot per_class [ 84.  94.  91.  90. 100. 102.]
rank: 3 train tot per_class [ 84.  94.  91.  90. 100. 102.]

                         | class-0   |    class-1  |    class-2   |     class-3  |    class-4   |    class-5  
-------------------------+-----------+-------------+--------------+--------------+--------------+---------------
rank_0_has_acc_per_class | 1.0000    |    0.9000   |     0.5091   |     0.9464   |     1.0000   |     1.0000
rank_1_has_acc_per_class | 1.0000    |    0.8833   |     0.8727   |     0.7500   |     0.7551   |     1.0000
rank_2_has_acc_per_class | 1.0000    |    1.0000   |     0.9636   |     0.9107   |     1.0000   |     1.0000
rank_3_has_acc_per_class | 1.0000    |    0.9000   |     0.6364   |     0.4821   |     0.0204   |     1.0000
per_class_opt1           | 1.0000    |    0.9667   |     0.9091   |     0.9464   |     0.7143   |     1.0000
per_class_opt2           | 1.0000    |    0.9000   |     0.9455   |     0.7679   |     1.0000   |     1.0000

