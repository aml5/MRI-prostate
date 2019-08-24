#%%
from keras.models import Sequential
from keras.layers import Dense
from keras import layers
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score, GridSearchCV
import numpy as np
import keras
import os
from keras.regularizers import l2
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import model_libraries
from capsule_layer import Capsule

#%% 
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# load pima indians dataset
X = np.load('train_x_dataset.npy')
Y = np.load('train_y_dataset.npy')
Y = np.array(Y==b'True').astype(np.float)
#Y = keras.utils.to_categorical(Y, num_classes=2, dtype='float32')
#%%
# Samplewise normalization

X = np.nan_to_num(X)
'''
for i in range(6):
    X[...,i] =  (X[...,i] - np.mean(X[...,i]))/ np.std(X[...,i])
'''
print(X.shape)
#Apply Augmentation
#Elastics Transfromation, Rotation, Flip and Flop.

#print(Y)
#%%
def GenerateModel(n_filter=32, number_of_class=1, input_shape=(48,48), number_of_channel=6,activation_last='sigmoid', metrics=['mse', 'acc'],loss='mse', optimizer='adam',dropout=0.5, init='glorot_uniform'):
    ########
    #
    #
    #######
    init_X = init #keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)
    filter_size =n_filter*4
    model = Sequential()
    shape_default = (input_shape[0], input_shape[1], number_of_channel)
    model.add(layers.SeparableConv2D(filters=filter_size, input_shape=shape_default, kernel_size=(1,1), strides=(1, 1), 
                                padding='same', activation='selu', use_bias=True, depthwise_initializer=init_X, pointwise_initializer=init_X))
    model.add(layers.Conv2D(filters=filter_size, 
                                kernel_size = (3,3),
                                strides=2,
                                padding='same'))
    model.add(layers.BatchNormalization(scale=True))
    model.add(layers.Activation('relu'))
    for kernel_size in ((1,3), (3,1),(1,1)):
        model.add(layers.Conv2D(filters=(filter_size//2), 
                                    kernel_size = kernel_size,
                                    strides=1,
                                    padding='same'))
        model.add(layers.BatchNormalization(scale=True))
        model.add(layers.Activation('relu'))
    #model.add(layers.Dropout(0.2))
    for i in [3,4]:
        model.add(layers.Conv2D(filters=(filter_size//i), 
                                    kernel_size = (3,3),
                                    strides=1,
                                    padding='same'))
        model.add(layers.BatchNormalization(scale=True))
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D((2, 2), padding='valid'))
        
    model.add(layers.Dropout(dropout))
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(number_of_class, activation=activation_last))
    model.compile(optimizer=optimizer,loss=loss, metrics=metrics)
    return model
#%%
# split into input (X) and output (Y) variables
# create model

cov_model = model_libraries.ConvModel_V2()
cov_model.activation_last='sigmoid'
cov_model.input_shape = (48,48)
cov_model.batch_size=10
cov_model.epoch=150
cov_model.lr=0.001
cov_model.lr_factor=2
cov_model.metrics=['acc', 'mse']
cov_model.patience = 5
cov_model.loss = 'mse'
cov_model.number_of_class=1
cov_model.number_of_channel=6
#%%
model = KerasClassifier(build_fn=GenerateModel, epochs=20, batch_size=32, verbose=1)
#%%
# evaluate using 10-fold cross validation
from sklearn.metrics import accuracy_score, precision_score, recall_score
def getScores(estimator, x, y):
    yPred = estimator.predict(x)
    yPred = yPred>0.5
    Y_true = Y>0.5
    return (accuracy_score(Y_true, yPred), 
            precision_score(Y_true, yPred, pos_label=1, average='macro'), 
            recall_score(Y_true, yPred, pos_label=1, average='macro'))

def my_scorer(estimator, x, y):
    a, p, r = getScores(estimator, x, y)
    print (a, p, r)
    return a+p+r
#%%
kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold, scoring='roc_auc')
print(results.mean())

#%%
print(results)
#%%3x2x4x4x4x3=
optimizers = ['sgd']
init = ['glorot_uniform']
loss = ['mse'] #'hinge', 'poisson'
epochs = [100]
activation_last=['linear'] #, 'softmax','relu',
batches = [32] #32
Dropout = [0.2]

param_grid = dict(optimizer=optimizers, 
                  epochs=epochs, 
                  batch_size=batches, 
                  activation_last=activation_last,
                  loss=loss,
                  dropout=Dropout,
                  init=init)
grid = GridSearchCV(estimator=model, param_grid=param_grid,scoring='roc_auc')
grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
	print("%f (%f) with: %r" % (mean, stdev, param))

#%%
