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
label_hdf5 =keras.utils.HDF5Matrix(hdf5_path, 'label')
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_true_f = K.round(y_true_f)
    y_pred_f = K.round(y_pred_f)
    intersection = K.sum(K.abs(y_true_f * y_pred_f), axis=-1)
    return (2. * intersection + smooth) / (K.sum(y_true_f,-1) + K.sum(y_pred_f,-1) + smooth)

model=load_model("./T2_Ax_result_ProstateCancerClassification_NormalvsCancer_01/weights-14.h5", custom_objects={'JunctionWeightLayer': utils.JunctionWeightLayer, 'dice_coef': dice_coef})
#%%
img_list = img_hdf5[1233+100:]
y_true = label_hdf5[1233+100:]
y_true=np.array(y_true>=1, dtype=np.uint8)

img_norm=[]
for img in img_list:
    img_norm.append(utils.Normalize(img))

y_predict=model.predict(np.array(img_norm))
y_predict = y_predict.flatten()

#for i in range(len(y_true)):
#    print(y_true[i], y_predict[i])
from sklearn.metrics import auc,average_precision_score, precision_recall_curve,classification_report, f1_score, confusion_matrix,brier_score_loss
from sklearn.metrics import roc_auc_score,roc_curve ,fowlkes_mallows_score
title="ROC Curve for case detection with prostate cancer"
utils.plotROCCurveMultiCall(plt,y_true, y_predict, title)
plt.savefig("./PCA_MRI_DETECTION_roc_curve.eps", transparent=True)
plt.savefig("./PCA_MRI_DETECTION_roc_curve.pdf", transparent=True)
plt.savefig("./PCA_MRI_DETECTION_roc_curve.png", transparent=True)
plt.show()
plt.close()
fpr, tpr, threshold = roc_curve(y_true, y_predict)
roc_auc = auc(fpr, tpr)
print('roc_auc',roc_auc)
threshold =utils.cutoff_youdens(fpr, tpr, threshold)
print('threshold',threshold)
print("Confusion matrix")
print(confusion_matrix(y_true, y_predict>threshold))
print("Classification report")
print(classification_report(y_true, y_predict>threshold))
print("fowlkes_mallows_score")
fms = fowlkes_mallows_score(y_true, y_predict>threshold)
print(fms)
brier_score =brier_score_loss(y_true, y_predict)
print("brier score")
print(brier_score)
