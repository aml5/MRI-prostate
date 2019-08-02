import keras
import os
import D3_utils
from yogi import Yogi
from keras import optimizers
import metrics
from keras import callbacks
from  cyclical_learning_rate import CyclicLR
import D3_utils
import math
import loss_functions
#######
# General configuration/Preprocessing
#######
initial_gpu = '0,1'#,2'
use_multiprocessing=False
parallel=True
number_of_gpus = 2
data_path='./train'
verbose=False
prefix_for_segmentation='segmentation'
preprocessing=None
standard_volume=[24,384,384]
normalize='CLAHE' ### ZSCORE, MAX, CLAHE
type_of_sharpness="Classic" ###TwoLevel, EDGE_ENHANCE, Convolute
clip_limit=0.05
D3_channel = 1
select_model = 'Prostate_D3_Segmentation'
output_channel = 0 
directory_mode=False
ShowResult=True
verbose_Test= False
train_valid_ratio=0.8
Onlive=False #False
nbins=1000
test_mode=True
img_data_train ='MRI_Prostate_Segmentation/img_train_set.npy'
mask_data_train = 'MRI_Prostate_Segmentation/mask_train_set.npy'
img_data_valid ='MRI_Prostate_Segmentation/img_valid_set.npy'
mask_data_valid = 'MRI_Prostate_Segmentation/mask_valid_set.npy'
#%%

#%%
#########
# Model Configuration
########
def lr_scheduler(epoch, lr):
    decay_rate = 0.999 # random.uniform(0.8,0.999)
    decay_step = 10
    if epoch % decay_step == 0 and epoch:
        return lr * decay_rate
    return lr
def step_decay(epoch):
	initial_lrate = lr
	drop = 0.5
	epochs_drop = 3.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

############
#   Hyperparameter for each model
############
#Model 1:
lr_decay = callbacks.LearningRateScheduler(schedule=lr_scheduler)#lambda epoch: configuration.lr * (0.9 ** configuration.epochs))
lr=1e-3
clr_00 = CyclicLR(base_lr=1e-6, max_lr=6e-3,
                                step_size=3200, mode='exp_range')
#Model 2:
#NONE
#from keras_gradient_noise import add_gradient_noise      
#opt_noise = add_gradient_noise(optimizers.Adam)
optimizer = optimizers.Adam() #opt_noise(lr, amsgrad=True)
HED=True
losses = {
	"o1": loss_functions.cross_entropy_balanced,
	"o2": loss_functions.cross_entropy_balanced,
    "o3": loss_functions.cross_entropy_balanced,
    "o4": loss_functions.cross_entropy_balanced,
    "o5": loss_functions.cross_entropy_balanced,
    "ofuse": loss_functions.cross_entropy_balanced,
    "output": loss_functions.dice_coef_loss,
}
metrics_lst = { "ofuse": [loss_functions.ofuse_pixel_error],
            "output": [metrics.dic_coef_v2]
}
hyperparameters =   {'Prostate_D3_Segmentation':
                        {
                        'gpu' : '2',
                        'number_of_class': 1,
                        'name' : 'Prostate Segmentation',
                        'optimizer': optimizers.Adam(lr),
                        'CheckPoint' : './result_D3_Model_Segmentation/weights-{epoch:02d}.h5',
                        'monitor' : 'val_output_dic_coef_v2',
                        'mode' : 'max',
                        'epochs' : 50,
                        'n_filter' : 16,
                        'AdditionalCallbacks' : [], #clr_00
                        'RunNormalize': False,
                        'train_sample_size' : 3200,
                        'valid_sample_size' : 10,
                        'clip_region': None,
                        'input_shape': (24,384,384,1),
                        'batch_size' : 4,
                        'RunAugmentation' : True,
                        'loss' : losses, #loss_functions.cross_entropy_balanced, # loss_functions.dice_coef_loss, #'binary_crossentropy', #
                        'metrics' :metrics_lst,# [loss_functions.ofuse_pixel_error,metrics.dic_coef_v2],#, metrics.dic_coef_v2, metrics.jaccard_distance], # metrics.auc_roc
                        'activation_last' : 'sigmoid'
                        },
                    'D3_Model_Detection':
                        {
                        'gpu' : '1',
                        'number_of_class' : 1, 
                        'name' : 'Significance Detection',
                        'optimizer': Yogi(lr=1e-4),
                        'CheckPoint' : './result_D3_Model_Detection/weights-{epoch:02d}.h5',
                        'monitor' : 'val_dice_coef_v2',
                        'mode' : 'max',
                        'epochs' : 50,
                        'RunNormalize': True,
                        'AdditionalCallbacks' : [clr_00],
                        'clip_region': [2,18],                     
                        'train_sample_size' : 16000,
                        'valid_sample_size' : 1600,
                        'input_shape': (16,144,144,1),
                        'batch_size' : 16,
                        'loss' : 'categorical_crossentropy',
                        'metrics' : ['mse', 'acc', metrics.dice_coef, metrics.recall_at_thresholds, metrics.precision_at_thresholds, metrics.auc_roc],
                        'activation_last' : 'softmax'
                        }
                    }
print(hyperparameters['Prostate_D3_Segmentation'])