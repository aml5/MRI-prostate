#%%
import h5py
import keras
import numpy as np
import matplotlib.pyplot as plt
#%%
from keras.models import load_model
from keras import backend as K
import utils
#%%
hdf5_path="./Data/T2_Ax.h5"
img_hdf5 = keras.utils.HDF5Matrix(hdf5_path, 'img')

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_true_f = K.round(y_true_f)
    y_pred_f = K.round(y_pred_f)
    intersection = K.sum(K.abs(y_true_f * y_pred_f), axis=-1)
    return (2. * intersection + smooth) / (K.sum(y_true_f,-1) + K.sum(y_pred_f,-1) + smooth)

model=load_model("./T2_Ax_result_ProstateCancerClassification_NormalvsCancer_01/weights-14.h5", custom_objects={'JunctionWeightLayer': utils.JunctionWeightLayer, 'dice_coef': dice_coef})
#%%
for img in img_hdf5:
    predict_y=model.predict(img.reshape(1,16,144,144,1))
    print(predict_y)
    stop()
#%%
img = img_hdf5[0]
predict_y=model.predict(img.reshape(1,16,144,144,1))
print(predict_y)
#%%

#%%
error_d  =[]
has_NAN =[]
import utils
#%%
import math
for i in range(len(img_hdf5)):
    mean_vle =np.mean(img_hdf5[i])

    if mean_vle>158:
        img_3D =img_hdf5[i]/np.max(img_hdf5[i])#utils.D3_equalize_adapthist(img_hdf5[i].astype(np.uint8))
        vle =np.mean(img_3D)
        if math.isnan(vle):
            has_NAN.append(i)
        else:
            error_d.append(i)
    
#%%
len(has_NAN)
print(has_NAN)
#%%
print(np.median(img_hdf5[200]))
print(np.mean(img_hdf5[200]))
print(np.mean(img_hdf5[0]))
#%%
#i=error_d[0]
print(np.mean(img_hdf5[603][5,:,:,0]))
img_3D =img_hdf5[603]#utils.D3_equalize_adapthist(img_hdf5[0].astype(np.uint8))
print(np.max(img_3D))
print(np.min(img_3D))
plt.imshow(img_3D[5,:,:,0])
plt.show()
plt.imshow(img_hdf5[603][5,:,:,0])
#%%

#%%

#%%
