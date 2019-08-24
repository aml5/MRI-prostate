from skimage.exposure import equalize_adapthist
import numpy as np
def Normalize(data):
    X_train_pos_ = data
    for index in range(X_train_pos_.shape[0]):
        X_train_pos = X_train_pos_[index]
        img = equalize_adapthist(X_train_pos_[index,:,:,0].astype(np.uint16),clip_limit=0.0005, nbins=1000)
        X_train_pos_[index,:,:,0] = img
    return X_train_pos_
def NormalizeOnce(data):
    X_train_pos_ = data
    for index in range(X_train_pos_.shape[0]):
        X_train_pos_[index] = Normalize(X_train_pos_[index])
    return X_train_pos_

X_train = np.load('img_train_data_3d_t2_tse_tra.npy', mmap_mode='r+')
X_valid = np.load('img_valid_data_3d_t2_tse_tra.npy', mmap_mode='r+')

X_train = NormalizeOnce(X_train)
X_valid = NormalizeOnce(X_valid)