
#%%
weight_i = 54
import keras
import tensorflow as tf
import numpy as np
from numpy.fft import fft, fftshift
import cv2
########
#
##
import os
import keras.backend as K
import tensorflow as tf

import yogi
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]='0'# configuration.CUDA_VISIBLE_DEVICES
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_squared_norm) / (0.5 + s_squared_norm)
    return scale * x
        
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_true_f = K.round(y_true_f)
    y_pred_f = K.round(y_pred_f)
    intersection = K.sum(K.abs(y_true_f * y_pred_f), axis=-1)
    return (2. * intersection + smooth) / (K.sum(y_true_f,-1) + K.sum(y_pred_f,-1) + smooth)

def recall_at_thresholds(y_pred, y_true,threshold=[0.5]):
    value, update_op = tf.metrics.recall_at_thresholds(y_pred, y_true, threshold)
    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'recall_at_thresholds' in i.name.split('/')[1]]
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value

def precision_at_thresholds(y_pred, y_true, threshold=[0.5]):
    value, update_op = tf.metrics.precision_at_thresholds(y_pred, y_true, threshold)
    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'precision_at_thresholds' in i.name.split('/')[1]]
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value

import matplotlib.cm as cm
from vis.visualization import visualize_cam
from keras import models
import cyclical_learning_rate
metrics=['mse', 'acc', dice_coef, recall_at_thresholds, precision_at_thresholds]
#./result_recent/weights-08.h5
imgs = np.load('img_valid_data_3d_t2_tse_tra.npy', mmap_mode='r')
imgs_class = np.load('outcome_valid_data_3d_t2_tse_tra.npy', mmap_mode='r')


Y=imgs_class
Y_neg = list(np.where(Y==False)[0].flatten())
Y_pos = list(np.where(Y==True)[0].flatten())
print(Y_neg)
print(Y_pos)
print(Y[Y_neg])
print(Y[Y_pos])
import random
counter = 0
for i in range(0,40):
        random.seed(
                counter
            )
        N_indexes_randomly_selected = random.sample(Y_neg, 16//2)
        P_indexes_randomly_selected = random.sample(Y_pos, 16//2)
        print(N_indexes_randomly_selected)
        print(P_indexes_randomly_selected)
        counter+=1
        if counter>=10:
                counter=0
print(imgs_class)

def Normalize(data):
        X_train_pos_ = data
        for index in range(X_train_pos_.shape[0]):
                X_train_pos = X_train_pos_[index]
                mean_ = np.mean(X_train_pos)
                std_ = np.std(X_train_pos)
                data_ = np.tanh((X_train_pos-mean_)/std_) 
                X_train_pos_[index] = data_
        return X_train_pos_ 
def get_output_layer(model, layer_name):
        # get the symbolic outputs of eachimagggggggggkey" layer (we gave them unique names).
        layer_dict = dict([(layer.name, layer) for layer in model.layers])
        layer = layer_dict[layer_name]
        return layer
def Generate_positive_negative_lists(Y, Positive_Value=True, Negative_Value=False):
    Y_neg = list(np.where(Y==Negative_Value)[0].flatten())
    Y_pos = list(np.where(Y==Positive_Value)[0].flatten())
    random.shuffle(Y_neg)
    random.shuffle(Y_pos)
    return Y_pos, Y_neg
def GenerateEqualPositiveAndNegativeValue(X,Y, batch_size=16, RunNormalize=True, max_iteration_per_epoch=10000):
        P_indexes, N_indexes = Generate_positive_negative_lists(Y)
        steps=X.shape[0]//batch_size
        counter  = 0
        while 1:
                if counter>=max_iteration_per_epoch:
                        counter =0
                for i in range(0,steps):
                        random.seed(
                        counter
                        )
                        N_indexes_randomly_selected = random.sample(N_indexes, batch_size//2)
                        P_indexes_randomly_selected = random.sample(P_indexes, batch_size//2)
                        list_selectred = N_indexes_randomly_selected + P_indexes_randomly_selected
                        convert_selected = P_indexes_randomly_selected + N_indexes_randomly_selected
                        #random.shuffle(list_selectred)
                        x_batch_tmp = X[list_selectred].copy()
                        x_batch_tmp_c = X[convert_selected].copy()
                        y_batch_tmp = Y[list_selectred].copy()
                        counter += 1 
                        if RunNormalize:
                                for index in range(x_batch_tmp.shape[0]):
                                        x_batch_tmp[index] = Normalize(x_batch_tmp[index])
                                        x_batch_tmp_c[index] = Normalize(x_batch_tmp_c[index])

                        yield x_batch_tmp, y_batch_tmp
def NormalizeImages(img3D):
        fim = img3D
        for index,img in enumerate(fim):
                fim[index] = Normalize(img)
        return fim

def softmax(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex / K.sum(ex, axis=axis, keepdims=True)
from keras.engine.topology import Layer
from keras.constraints import min_max_norm
 
class RotationThetaWeightLayer(Layer): # a scaled layer
    def __init__(self,**kwargs):
        super(RotationThetaWeightLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        self.W1 = self.add_weight(name='kernel', 
                                      shape=(1,),
                                      initializer='uniform',
                                      trainable=True,
                                      constraint=min_max_norm(min_value=0.78, max_value=4))
        self.W2 = self.add_weight(name='kernel', 
                                      shape=(1,),
                                      initializer='uniform',
                                      trainable=True,
                                      constraint=min_max_norm(min_value=0.78, max_value=4))
        super(RotationThetaWeightLayer, self).build(input_shape)

    def call(self, x):
        assert isinstance(x, list)
        a, b = x
        
        return K.cos(3.14159265359/self.W1) * (-2) * K.exp(-(a**2+b**2)) + K.sin(3.14159265359/self.W2) * (-2) * b * K.exp(-(a**2+b**2))


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

print('./result_recent/weights-{:02d}.h5'.format(weight_i))
model = models.load_model('./result_recent/weights-{:02d}.h5'.format(weight_i), custom_objects={
                'dice_coef':dice_coef,
                'recall_at_thresholds': recall_at_thresholds,
                'precision_at_thresholds': precision_at_thresholds,
                'SineReLU' : cyclical_learning_rate.SineReLU,
                'Yogi': yogi.Yogi,
                'RotationThetaWeightLayer': RotationThetaWeightLayer,
                'CC_loss': CC_loss,
                'softmax': softmax,
                'squash': squash})

from  sklearn.metrics import confusion_matrix,classification_report
predictions = []
class_g = []
rund_d = True
counter = 0

_, prediction  = model.predict(NormalizeImages(imgs.reshape(41,16,144,144,1).copy()), batch_size=16)
        
predictions = prediction>0.5

confusion_matrix(imgs_class,predictions,labels=[0, 1]) 
print(classification_report(imgs_class,predictions,labels=[0, 1]))

_, prediction  = model.predict(NormalizeImages(imgs.reshape(41,16,144,144,1).copy()), batch_size=16)
for c, p in zip(imgs_class,prediction):
        print (c,p)

#%%
import math
case_id = 3
final_conv_layer_1 = get_output_layer(model, "input_centered_and_divided_normalized")
final_conv_layer_2 = get_output_layer(model, "stroma")
final_conv_layer_3 = get_output_layer(model, 'RotationInvariant')#
final_conv_layer_4 = get_output_layer(model, 'Y_AUTO_NORMAL')#
final_conv_layer_5 = get_output_layer(model, 'input_centered_by_autoencoder')#


get_output = K.function([model.layers[0].input], \
                                        [final_conv_layer_1.output,final_conv_layer_2.output,final_conv_layer_3.output,final_conv_layer_4.output,final_conv_layer_5.output,  model.layers[-1].output])
                
[conv_outputs_1,conv_outputs_2, conv_outputs_3, conv_outputs_4, conv_outputs_5,prediction] = get_output([Normalize(imgs[case_id].reshape(1,16,144,144,1).copy())])
print(prediction)
print(1/(1+math.exp(-prediction)))
print(imgs_class[case_id])

#%%
print(np.min(conv_outputs_4))
print(np.median(conv_outputs_4))
print(np.mean(conv_outputs_4))
print(np.max(conv_outputs_4))
#%%
print('start')
for i in range(16):
    plt.imshow(conv_outputs_1[0,i,:,:,0],  cmap='gray', vmin=np.min(conv_outputs_1),vmax=np.max(conv_outputs_1))
    plt.show()
    plt.imshow(conv_outputs_2[0,i,:,:,0], cmap='gray', vmin=np.min(conv_outputs_2),vmax=np.max(conv_outputs_2))
    plt.show()
    plt.imshow(conv_outputs_3[0,i,:,:,0], cmap='gray', vmin=np.min(conv_outputs_3),vmax=np.max(conv_outputs_3))
    plt.show()
    plt.imshow(conv_outputs_4[0,i,:,:,0], cmap='gray', vmin=np.min(conv_outputs_4),vmax=np.max(conv_outputs_4))
    plt.show()
    plt.imshow(conv_outputs_5[0,i,:,:,0], cmap='gray', vmin=np.min(conv_outputs_5),vmax=np.max(conv_outputs_5))
    plt.show()
    plt.imshow(imgs[case_id,i,:,:,0], cmap='gray', vmin=np.min(imgs),vmax=np.max(imgs))
    plt.show()