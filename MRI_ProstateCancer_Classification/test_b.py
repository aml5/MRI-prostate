#%%
import keras
import tensorflow as tf
import numpy as np
from numpy.fft import fft, fftshift
import cv2
########
#
##
import os
#os.chdir('/Volumes/eminaga/NAS/Projects/CoRegistration/')
import keras.backend as K
import tensorflow as tf

import yogi
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]='0'# configuration.CUDA_VISIBLE_DEVICES
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

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


#%%
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
#%%
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
#%%
model.summary()
#%%
print(imgs_class)
#%%
import math
case_id = 3
final_conv_layer = get_output_layer(model, "input_centered_and_divided_normalized")
final_conv_layer_2 = get_output_layer(model, "stroma")
final_conv_layer_a = get_output_layer(model, 'stroma')#
final_conv_layer_r = get_output_layer(model, 'auto_encoder')#


get_output = K.function([model.layers[0].input], \
                                        [final_conv_layer.output,final_conv_layer_2.output,final_conv_layer_a.output,final_conv_layer_r.output,  model.layers[-1].output])
                
[conv_outputs_0,conv_outputs_1, autoencoder_x, RotationInvariant_d,prediction] = get_output([Normalize(imgs[case_id].reshape(1,16,144,144,1).copy())])
autoencoder_x_c = autoencoder_x
print(prediction)
print(1/(1+math.exp(-prediction)))
print(imgs_class[case_id])
#conv_outputs_1 = conv_outputs_1>=np.mean(conv_outputs_1)
#%%
print(np.min(autoencoder_x))
print(np.max(autoencoder_x))
print(np.mean(autoencoder_x))
print(np.median(autoencoder_x))
#%%
conv_outputs_1 = conv_outputs_1_copy.copy()>=0.999

#%%
autoencoder_x = autoencoder_x_c.copy()
print(imgs_class[case_id])
autoencoder_x = (autoencoder_x - np.min(autoencoder_x))/(np.max(autoencoder_x)-np.min(autoencoder_x))
print(np.min(autoencoder_x))
print(np.max(autoencoder_x))
print(np.mean(autoencoder_x))
autoencoder_x = autoencoder_x>=np.mean(autoencoder_x)
print(np.min(autoencoder_x))
print(np.max(autoencoder_x))
print(np.mean(autoencoder_x))
#%%
print(np.mean(RotationInvariant_d))
print(np.median(RotationInvariant_d))
print(np.std(RotationInvariant_d))
print(np.max(RotationInvariant_d))
print(np.min(RotationInvariant_d))
#%%
import math
def primeFactors(n): 
        numbers = []
        # Print the number of two's that divide n 
        while n % 2 == 0:
                numbers.append(2)
        
                n = n / 2
          
        # n must be odd at this point 
        # so a skip of 2 ( i = i + 2) can be used 
        for i in range(3,int(math.sqrt(n))+1,2): 
          
        # while i divides n , print i ad divide n 
                while n % i== 0: 
                        numbers.append(i)
                        n = n / i
              
        # Condition if n is a prime 
        # number greater than 2 
        if n > 2: 
                numbers.append(n)
        return  numbers
x = primeFactors(1296)


# A Python program to print all  
# permutations of given length 
from itertools import permutations, combinations,combinations_with_replacement
  
# Get all permutations of length 2 
# and length 2 
unique_num = np.unique(x)
print(unique_num)
perm = combinations_with_replacement([2,3], 4) 
print(perm)
# Print the obtained permutations
#%% 
ds = list(perm)
print(ds)
print(len(ds))
print(len(ds)//2)

#%%
from functools import reduce
comb_1=[2,2,2,2]
kernel_size_1 = (1, reduce(lambda x, y: x*y, comb_1))
print(kernel_size_1)
#%%
for i in list(perm): 
        print(i)

import numpy as np
print(np.unique(x))
#%%
def factors(n):
        while n>1:
                for i in [1,2,3,5,7,11,47]:
                        if n % i ==0:
                                v = n // i
                                n = v
                                yield i
                                break

for factor in factors(1296):
        print(factor)
#%%
print('start')
for i in range(16):
        plt.imshow(conv_outputs_0[0,i,:,:,0],  cmap='gray', vmin=np.min(conv_outputs_0),vmax=np.max(conv_outputs_0))
        plt.show()
        plt.imshow(conv_outputs_1[0,i,:,:,0], cmap='gray', vmin=np.min(conv_outputs_1),vmax=np.max(conv_outputs_1))
        plt.show()
        plt.imshow(RotationInvariant_d[0,i,:,:,0], cmap='gray', vmin=np.min(RotationInvariant_d),vmax=np.max(RotationInvariant_d))
        plt.show()
        plt.imshow(autoencoder_x[0,i,:,:,0], cmap='gray', vmin=np.min(autoencoder_x),vmax=np.max(autoencoder_x))
        plt.show()
        plt.imshow(imgs[case_id,i,:,:,0], cmap='gray', vmin=np.min(imgs),vmax=np.max(imgs))
        plt.show()
#%%      
'''  
for i in range(16):
        #plt.imshow((imgs[case_id,i,:,:,0]*((conv_outputs_1[0,i,:,:,0])*conv_outputs_0[0,i,:,:,0]))/(imgs[case_id,i,:,:,0]**(1-autoencoder_x[0,i,:,:,0])))
        #plt.show()
        plt.imshow((imgs[case_id,i,:,:,0]*((conv_outputs_1[0,i,:,:,0])*conv_outputs_0[0,i,:,:,0]))/(1+autoencoder_x[0,i,:,:,0]))
        plt.show()
        plt.imshow((1-conv_outputs_0[0,i,:,:,0])*conv_outputs_1[0,i,:,:,0])
        plt.show()
        plt.imshow(1-conv_outputs_1[0,i,:,:,0])
        plt.show()
        plt.imshow(imgs[case_id,i,:,:,0])
        pltstroma.show()
'''      
#%%
#/result_recent/weights-32.h5 weights-26
#/result_recent/weights-04.h5
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
                
                

#%%
#%% weights-24
#/result_recent/weights-39.h
#weights-32.h5
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
i= 1
print('./result_recent/weights-{:02d}.h5'.format(i))
model = models.load_model('./result_recent/weights-{:02d}.h5'.format(i), custom_objects={
                'dice_coef':dice_coef,
                'recall_at_thresholds': recall_at_thresholds,
                'precision_at_thresholds': precision_at_thresholds,
                'SineReLU' : cyclical_learning_rate.SineReLU,
                'Yogi': yogi.Yogi,
                'RotationThetaWeightLayer': RotationThetaWeightLayer,
                'CC_loss': CC_loss,
                'softmax': softmax})
#%%
#weights-09
for i in [1]:
        

        '''
        #Get the 512 input weights to the softmax.
        final_conv_layer = get_output_layer(model, "reshape_1")

        get_output = K.function([model.layers[0].input], \
                        [final_conv_layer.output, 
        model.layers[-1].output])
        '''
        
        from  sklearn.metrics import confusion_matrix,classification_report
        predictions = []
        class_g = []
        rund_d = True
        counter = 0

        _, prediction  = model.predict(NormalizeImages(imgs.reshape(41,16,144,144,1).copy()), batch_size=16)
        
        predictions = prediction>0.5
        #for i in range(imgs.shape[0]):
        #        autoencoder, prediction  = model.predict(Normalize(imgs[i].reshape(1,16,144,144,1).copy()))
        #        #print(imgs_class[i],prediction)
        #        predictions.append(np.argmax(prediction))


        confusion_matrix(imgs_class,predictions,labels=[0, 1]) 
        print(classification_report(imgs_class,predictions,labels=[0, 1]))
#%%
_, prediction  = model.predict(NormalizeImages(imgs.reshape(41,16,144,144,1).copy()), batch_size=16)
for c, p in zip(imgs_class,prediction):
        print (c,p)
#%%
img_batch, class_img_x = generator.__next__()
print(class_img_x.shape)
print(class_img_x)
for i,x in enumerate(class_img_x):
        print(i,x)
#%%
[conv_outputs, predictions] = get_output([Normalize(imgs[case_id].reshape(1,16,144,144,1).copy())])
print(conv_outputs.shape)

print(len(model.layers))
class_weights = model.layers[-3].get_weights()[0]
print(class_weights.shape)
print(conv_outputs.shape)

#Create the class activation map.
target_class = print(confusion_matrix)
cam = np.zeros(dtype = np.float32, shape = (4,1296))
class_weights = model.layers[-3].get_weights()[0]
class_weights_selected = class_weights[...,target_class]
for i in range(128):
    for j in range(4):
        data = conv_outputs[0,j,:, i].flatten()
        cam[j] += data * class_weights_selected[i,j]
cam = cam.reshape(4,36,36)
#cam /= np.max(cam)
#cam = (cam +1)/2
#print(np.min(cam))
#print(np.max(cam))
import cv2
counter=0

import matplotlib.pyplot as plt
import numpy as np
from skimage import transform

def add(image, heat_map, alpha=0.3, display=False, save=None, cmap='jet', axis='on', verbose=False):

    height = image.shape[0]
    width = image.shape[1]

    # resize heat map
    heat_map_resized = transform.resize(heat_map, (height, width))

    # normalize heat map
    max_value = np.max(heat_map_resized)
    min_value = np.min(heat_map_resized)
    normalized_heat_map = (heat_map_resized - min_value) / (max_value - min_value)

    # display
    plt.imshow(image)
    plt.imshow(255 * normalized_heat_map, alpha=alpha, cmap=cmap, interpolation='bilinear', vmin=0, vmax=255)
    plt.axis(axis)

    if display:
        plt.show()

    if save is not None:
        if verbose:
            print('save image: ' + save)
        plt.savefig(save, bbox_inches='tight', pad_inches=0)


print('Class', imgs_class[case_id])
print('Predicted',np.argmax(predictions))

for i in range1):
    #img_f = cv2.resize(cam[i], (144, 144))
    #heatmap = cv2.applyColorMap(np.uint8(255*img_f), cv2.COLORMAP_JET)
    
    for j in range(4):
        img_org = imgs[case_id,counter,:,:,0]
        counter+=1
        org_img = np.zeros((144,144,3))
        org_img[...,0] = img_org
        org_img[...,1] = img_org
        org_img[...,2] = img_org
        org_img = org_img/np.max(org_img)
        #img = heatmap*0.5+(np.uint8(org_img*255))
        #print(np.max(img))
        #print(np.min(img))
        output_path= "%s.png"% counter
        add(np.uint8(org_img*255),cam[i], save=output_path)
        plt.imshow(org_img)
        plt.show()
        #plt.imshow(img_org)
        #plt.show()
        #cv2.imwrite(output_path,img)

#%%
print(conv_outputs.shape)
cam = np.zeros(dtype = np.float32, shape = conv_outputs.shape[1:4])
print(cam.shape)
target_class = 1
class_w = class_weights[...,target_class]
print(class_w.shape)
cam = class_w * conv_outputs[0]
print(np.min(cam))
print(np.max(cam))
print(np.median(cam))
plt.plot(cam)
plt.show()
#%%
for i, w in enumerate():
    print(w.shape)
    print(conv_outputs.shape)
    cam += conv_outputs[0,i, :] * w
#%%
plt.plot(cam)
plt.show()
#%%
print(get_output)
#%%
for i in range(19):
    [conv_outputs, predictions] = get_output([Normalize(imgs[i].reshape(1,16,144,144,1).copy())])
    print(i, predictions)
    print(i, 'mean', np.mean(conv_outputs[0]))
    print(i, 'median', np.median(conv_outputs[0]))
    #conv_outputs = conv_outputs[0, :, :]
    
    for i in range(1):
        plt.title(predictions)
        plt.plot(conv_outputs[0])
        plt.show()
#%%
for i in range(8):
    img_to_show = conv_outputs[0,i,...,60]
    plt.imshow(img_to_show)
    plt.show()
#%%
from vis.utils import utils
penultimate_layer = utils.find_layer_idx(model, 'sine_re_lu_2')
layer_idx = utils.find_layer_idx(model, 'sine_re_lu_2')
print(layer_idx)

#%%
print(imgs.shape)
#%%

imgs = np.load('img_valid_data_3d_t2_tse_tra.npy', mmap_mode='r')
#%%
img = visualize_activation(model, layer_idx=1, filter_indices=[10,1,5], input_range=(0., 1.))
print(img)
#%%
#for modifier in [None, 'guided', 'relu']:
grads = visualize_saliency(model, 
									-1, filter_indices=None, 
									seed_input=imgs[0], 
									grad_modifier="absolute",
									backprop_modifier=None)
#%%


plt.figure()
f, ax = plt.subplots(1, 2)
for i, img in enumerate([imgs[0], imgs[1]]):
	ax[i].imshow(grads, cmap='jet')
	print(grads)
#%%	
def convolve(image, kernel):
	# grab the spatial dimensions of the image, along with
	# the spatial dimensions of the kernel
	(iH, iW) = image.shape[:2]
	(kH, kW) = kernel.shape[:2]
 
	# allocate memory for the output image, taking care to
	# "pad" the borders of the input image so the spatial
	# size (i.e., width and height) are not reduced
	pad = (kW - 1) // 2
	image = cv2.copyMakeBorder(image, pad, pad, pad, pad,
		cv2.BORDER_REPLICATE)
	output = np.zeros((iH, iW), dtype="float32")

'''
laplacian = np.array((
	[-1, -1, 0],
	[0, 1,0],
	[0, 1, 1]), dtype="float")
'''
laplacian = np.array((
	[-1, -0.2, 0],
	[0, 2,0],
	[0, 0.2, 1]), dtype="float")

#convoleOutput = convolve(output, laplacian)
opencvOutput = cv2.filter2D(h, -1, laplacian)
print(np.mean(opencvOutput))
print(np.max(opencvOutput))
print(np.min(opencvOutput))
plt.imshow(opencvOutput)
plt.show()
#%%

edges = cv2.Canny(im, th/2, th)
h= (h*255)+128
h[h<0] = 0
h[h>255] = 255 
h = h.astype(np.uint8)
print(np.min(h))
print(np.max(h))
print(type(h))
edges = cv2.Canny(h,20,128,apertureSize = 3)
lines = cv2.Eged(edges,2,np.pi/180,200)
for rho,theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(h,(x1,y1),(x2,y2),255,2)
print(h.shape)
plt.imshow(h[...,0]/255)
plt.show()
cv2.imwrite('houghlines3.jpg',h)
#%%
laplacian = cv2.Laplacian(h,cv2.CV_64F)
print(np.mean(laplacian))
print(np.median(laplacian))
print(np.max(laplacian))
print(np.min(laplacian))
plt.imshow(laplacian)
plt.show()
#%%
mag = np.abs(fft(laplacian))
response = 5 * np.log2(mag)
print(response.shape)
print(np.min(response))
print(np.max(response))
print(np.median(response))
print(np.mean(response))

response = np.clip(response, -10, 10)
#response = np.nan_to_num(response)
plt.imshow(h[...,0])
plt.show()
plt.imshow(response)
plt.show()

#%%
tf.spectral.fft3d
