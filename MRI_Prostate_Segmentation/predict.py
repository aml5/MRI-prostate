import SimpleITK as sitk
import matplotlib.pyplot as plt
from pylab import rcParams
import numpy as np
import matplotlib.gridspec as gridspec
# import cv2
import sys, os
from os.path import isfile, join
from os import listdir
import argparse
import keras
from keras import optimizers, callbacks, models
from keras_preprocessing.image import ImageDataGenerator
from keras.utils.vis_utils import plot_model
import loss_functions
import model_storage
import cv2
from PIL import Image, ImageEnhance
import PIL
from skimage import measure
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral
from skimage import exposure
from sklearn.preprocessing import QuantileTransformer
from skimage.morphology import remove_small_objects
import configuration
import progressbar
import time
import D3_utils as utils
import tensorflow as tf
import model_storage
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]= configuration.initial_gpu
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

class

class volume:

    def __init__(self, filename, weightpath=r"results\weights-32.h5"):
        self._filename = filename
        self._weightpath = weightpath

    def run(self):
        self.preprocess()
        self.modelConfig()
        self.predict()
        self.output()

    def preprocess(self):
        self._data = utils.LoadFile(self._filename,normalize='CLAHE')
        self._data = self._data.reshape(1,*self._data.shape)
        self._zeros = np.zeros(self._data.shape)

    def modelConfig(self):
        h_pars = configuration.hyperparameters[configuration.select_model]

        loader = model_storage.Model_Storage(input_shape=h_pars['input_shape'],
                                             n_filter=h_pars['n_filter'],
                                             number_of_class=h_pars['number_of_class'],
                                             activation_last=h_pars['activation_last'],
                                             metrics=h_pars['metrics'],
                                             loss=h_pars['loss'],
                                             optimizer=h_pars['optimizer']
                                             )
        if (configuration.parallel):
            with tf.device('/cpu:0'):
                method_to_call = getattr(loader, configuration.select_model)  # 'CancerDetection_HarmonicSeries')
                self._model = method_to_call()
        else:
            method_to_call = getattr(loader, configuration.select_model)  # 'CancerDetection_HarmonicSeries')
            self._model = method_to_call()
        print(self._model.summary())
        if configuration.parallel:
            from keras.utils import multi_gpu_model
            self._model = multi_gpu_model(self._model, gpus=configuration.number_of_gpus, cpu_relocation=False, cpu_merge=True)
            self._model = loader.Compile(self._model)
        else:
            # Compile the model.
            self._model = loader.Compile(self._model)
        self._model.load_weights(self._weightpath)

    def predict(self):
        self._clip = [384,0,384,0,0,100]
        mask = np.array(configuration.standard_volume)
        predict_set = np.vstack((self._data,self._zeros))#np.load(self._filename)
        predict_set = predict_set.reshape(*predict_set.shape,1)
        prelim = self._model.predict(predict_set)[-1]
        results = prelim[0]
        print(results.shape)
        mask = utils.smooth_contours(utils.smooth_contours(results,type='CV2'))
        utils.multi_slice_viewer_legacy(mask.reshape(configuration.standard_volume),
                                 self._data.reshape(configuration.standard_volume))
        plt.show()
        self.findBorder(mask)

    def findBorder(self, mask):
        # finds z limits
        for i in range(mask.shape[0]):
            if self.isEmpty(mask[i]):
                if (i - self._clip[4] <= 1):
                    self._clip[4] = i
            if not self.isEmpty(mask[i]):
                self._clip[5] = i+1

        # finds x and y limits
        mask_xy = self.flatten(mask)
        for i in range(mask_xy.shape[0]):
            for j in range(mask_xy.shape[1]):
                if mask_xy[i][j]:
                    if self._clip[0] > j : self._clip[0] = j
                    if self._clip[1] < j : self._clip[1] = j
                    if self._clip[2] > i : self._clip[2] = i
                    if self._clip[3] < i : self._clip[3] = i

    def isEmpty(self, slice):
        if (np.count_nonzero(slice) == 0):
            return True
        return False

    def flatten(self, mask):
        flattened = np.zeros([384,384])
        for i in range(mask.shape[1]):
            for j in range(mask.shape[2]):
                for k in range(mask.shape[0]):
                    if mask[k][i][j] == 1:
                        flattened[i][j] = 1
        return flattened

    def output(self):
        import h5py
        self._data = self._data.reshape(configuration.standard_volume)
        dim_orig = self._data.shape
        self._clip[0] *= dim_orig[1] / configuration.standard_volume[1]
        self._clip[1] *= dim_orig[1] / configuration.standard_volume[1]
        self._clip[2] *= dim_orig[2] / configuration.standard_volume[2]
        self._clip[3] *= dim_orig[2] / configuration.standard_volume[2]
        self._clip[4] *= dim_orig[0] / configuration.standard_volume[0]
        self._clip[5] *= dim_orig[0] / configuration.standard_volume[0]
        self._clip = [int(i) for i in self._clip]
        self._data = self._data[self._clip[4]: self._clip[5], self._clip[0]:self._clip[1], self._clip[2]:self._clip[3]]
        print('Saving to .h5 file.')
        h5f = h5py.File('data.h5', 'w')
        h5f.create_dataset('dataset', data=self._data)
        h5f.close()
        # sitkimg = sitk.GetImageFromArray(self._data)
        # sitk.WriteImage(sitkimg, 'data.mhd')

    def h5open(self):
        pass

if __name__ == '__main__':
    # filename, weightpath
    img = volume(sys.argv[1],sys.argv[2])
    img.run()