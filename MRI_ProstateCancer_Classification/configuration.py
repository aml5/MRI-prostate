import os
from keras import optimizers
from keras import callbacks
import cyclical_learning_rate
import utils
from keras import backend as K
import tensorflow as tf
import math
#GPU
CUDA_VISIBLE_DEVICES="0,1"
gpu=2
parallel=True
######
# Function
######
def CC_loss(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(tf.multiply(xm,ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den

    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return 1-r

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_true_f = K.round(y_true_f)
    y_pred_f = K.round(y_pred_f)
    intersection = K.sum(K.abs(y_true_f * y_pred_f), axis=-1)
    return (2. * intersection + smooth) / (K.sum(y_true_f,-1) + K.sum(y_pred_f,-1) + smooth)

#Results
save_dir="./T2_Ax_result_ProstateCancerClassification_NormalvsCancer_02"
if os.path.exists(save_dir) is not True:
    os.mkdir(save_dir)
#Input
test_mode=False
input_shape=(16,144,144)
batch_size=16
number_input_channel=1
#Output
final_activation='sigmoid'
n_class=1
#Model
lr=1e-4
epochs=50
decay = lr/epochs
counter_s=0
metrics=["acc",dice_coef]#{'prediction' :dice_coef, 'autoencoder': 'mse'}#['mse', 'binary_accuracy', dice_coef]
loss="binary_crossentropy" #[CC_loss, 'binary_crossentropy']#CC_loss, 'binary_crossentropy']
model_name="DataInspector3D"
loss_weights=None #[0.03,1.5]
import yogi
optimizer =optimizers.adam(lr=lr)#yogi.Yogi(lr=lr)#, clipvalue=0.6)# yogi.Yogi(lr=lr)#yogi.Yogi(lr=lr)# optimizers.RMSprop(lr=lr)#yogi.Yogi(lr=lr) 

def step_decay(epoch):
	initial_lrate = lr
	drop = 0.5
	epochs_drop = 3.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate
#Callbacks
lr_decay = callbacks.LearningRateScheduler(schedule=step_decay)#lambda epoch: lr * (0.9999 ** epoch))
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', mode="min", factor=0.5, patience=3, min_lr=1e-10,verbose=1)
checkpoint = callbacks.ModelCheckpoint(save_dir + '/weights-{epoch:02d}.h5', monitor='val_dice_coef',
                                           save_best_only=False, save_weights_only=True, verbose=1, mode="max") 

iterations=100
clr = cyclical_learning_rate.CyclicLR(base_lr=lr, max_lr=lr*6,
                                step_size=2*iterations, Peak_Location='LowFirst')#, mode='exp_range')
    
    
list_for_callbacks=[checkpoint, lr_decay,reduce_lr]#,clr]#,clr,lr_decay,reduce_lr]


#Normalization
Normalization="tanh"#'tanh'#"tanh"
clip_limit=0.05
nbins=1000
#Augmentation
shuffle=True
run_augmentations=True
augmentation_factor_for_training=4
augmentation_factor_for_validation=1
type_of_augmentation=None
dict_parameter=None
#Multithread
workers=10
max_queue_size=10
use_multiprocessing=False
#Cohort
dataset_hdf5_path="./Data/T2_Ax.h5"
class_weight=None
convert_to_categorical=False
binarize=True
Two_output=False
threshold_to_binary=1
#%%%
#Determine the right cases
import h5py
import keras
import numpy as np
import random
random.seed(123)
img_hdf5 = keras.utils.HDF5Matrix(dataset_hdf5_path, 'img')
#%%
study_cohort  =[]
for i in range(len(img_hdf5)):
    mean_vle =np.mean(img_hdf5[i])
    
    if i in [200, 603, 770, 783, 826, 842, 957, 1478, 1053, 1137]:
        continue
    if mean_vle>50:
        study_cohort.append(i)
random.shuffle(study_cohort)
print("number of study cohort:", len(study_cohort))
train_length =int(round(len(study_cohort)*0.7))
training_set_index = study_cohort[0:train_length]
print("number of training:", len(training_set_index))

validation_set_index = study_cohort[train_length:train_length+100]
test_set_index=  study_cohort[train_length+100:]
print("number of validation:", len(validation_set_index))
print("number of test:", len(test_set_index))
np.save("train_set_index.npy",training_set_index)
np.save("validation_set_index.npy",validation_set_index)
np.save("test_set_index.npy",test_set_index)