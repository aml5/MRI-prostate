import SimpleITK as sitk
import matplotlib.pyplot as plt
from pylab import rcParams
import numpy as np
import matplotlib.gridspec as gridspec
import cv2
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
import json
import pydicom
import tarfile
import h5py
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]= configuration.initial_gpu
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

class patient:

    def __init__(self, patientID, jsonpath):
        self._patientID = patientID
        self._jsonpath = jsonpath
        self.run()

    def run(self):
        self.readJSON()
        self.createData()

    def readJSON(self):
        path = self._jsonpath
        with open(path, 'r') as f:
            self._jsondata = json.load(f)[self._patientID]
        # wrapper, filename + (data -> AccessionID, Folder, FileList)

    def createData(self):
        self._data = np.array([])
        tgzpath = self._jsondata["filename"]
        with tarfile.open(tgzpath) as f:
            for i, fileset in enumerate(self._jsondata["data"]["FileList"]):
                tmp = volume(os.path.splitext(os.path.basename(tgzpath))[0] + '-' + self._jsondata['data']['ImageType'][i])
                tmp.compile(fileset, f)
                tmp.run()
                # tmp = volume.getVolume(tmp)
                # tmp.run()
        # self._data = np.append(self._data, tmp)

    def outputData(self):
        pass

class volume:

    def __init__(self, volname, weightpath=r"C:/Users/Andrew Lu/Documents/Projects/MRI_Prostate_Segmentation/results/result_Prostate_D3_Segmentation_20190705-1805/weights-32.h5"):
        self._weightpath = weightpath
        self._volname = volname

    #
    # def __init__(self, filename, weightpath=r"results\weights-32.h5"):
    #     self._filename = filename
    #     self._weightpath = weightpath
    #     self._data = utils.LoadFile(self._filename,normalize='CLAHE')

    @classmethod
    def saveVolume(cls, data, volname, outputtype, filetype='h5'):
        if filetype=='h5':
            print('Saving to output/' + volname + '/' + volname + '-' + outputtype + '.h5...')
            if os.path.exists('output/' + volname):
                pass
            else:
                os.mkdir('output/' + volname)
            h5f = h5py.File('output/' + volname + '/' + volname + '-' + outputtype + '.h5', 'w')
            h5f.create_dataset('dataset', data=data)
            h5f.close()
            print('Volume saved. \n ----------')
            volume.saveVolume(data, volname, outputtype, 'mhd')
        if filetype=='mhd':
            print('Saving to output/' + volname + '/' + volname + '-' + outputtype + '.mhd...')
            if os.path.exists('output/' + volname):
                pass
            else:
                os.mkdir('output/' + volname)
            sitkimg = sitk.GetImageFromArray(data)
            sitk.WriteImage(sitkimg, 'output/' + volname + '/' + volname + '-' + outputtype + '.mhd')
            print('Volume saved. \n ----------')

    @classmethod
    def saveVolumemhd(cls, data, volname, outputtype):
        print('Saving to output/' + volname + '-' + outputtype + '.mhd...')
        if os.path.exists('output'):
            pass
        else:
            os.mkdir('output')
        sitkimg = sitk.GetImageFromArray(data)
        sitk.WriteImage(sitkimg, 'output/' + volname + '-' + outputtype + '.mhd')
        print('Volume saved. \n ----------')

    @classmethod
    def getVolume(cls, patient):
        return patient._data

    def run(self):
        self.preprocess()
        self.modelConfig()
        self.predict()
        self.output()

    def compile(self, fileset, f):
        print('Compiling images...')
        self._data = sorted([pydicom.read_file(f.extractfile(x)) for x in fileset], key=lambda x: x.InstanceNumber)
        self._data = np.array([self._data[i].pixel_array for i in range(len(fileset))])
        self._orig = self._data.copy()
        volume.saveVolume(self._data, self._volname, 'original', 'h5')

    def compile_mhd(self, filename):
        img = sitk.ReadImage(filename)
        self._data = sitk.GetArrayFromImage(img)
        self._orig = self._data.copy()
        volume.saveVolume(self._data,self._volname,'original', 'h5')

    def compile_folder(self, folderpath):
        files = [os.path.join(folderpath, x) for x in os.listdir(folderpath) if
                 x.endswith('.dcm')]
        self._data = sorted([pydicom.read_file(x) for x in files], key=lambda x: x.InstanceNumber)
        self._data = np.array([self._data[i].pixel_array for i in range(len(files))])
        self._orig = self._data.copy()
        volume.saveVolume(self._data, self._volname, 'original', 'h5')

    def preprocess(self, verbose=False, apply_curve_smoothing=False):
        # Normalize
        ct_scan = utils.D3_equalize_adapthist(self._data, clip_limit=configuration.clip_limit, nbins=configuration.nbins)
        ct_scan = ct_scan / np.max(ct_scan)
        # Sharpen
        ct_scan = utils.Sharp3DVolume(ct_scan, type_of_sharpness=configuration.type_of_sharpness)
        img = sitk.GetImageFromArray(ct_scan)

        if verbose:
            print('img', img.GetDimension(), img.GetDirection(), img.GetOrigin(), img.GetSpacing())

        # Resize Standard

        new_x_size = configuration.standard_volume[1]
        new_y_size = configuration.standard_volume[2]
        new_z_size = configuration.standard_volume[0]

        # Create the reference image with a zero origin, identity direction cosine matrix and dimension
        new_size = [new_x_size, new_y_size, new_z_size]
        new_spacing = [old_sz * old_spc / new_sz for old_sz, old_spc, new_sz in
                       zip(img.GetSize(), img.GetSpacing(), new_size)]

        interpolator_type = sitk.sitkLanczosWindowedSinc
        new_img = sitk.Resample(img, new_size, sitk.Transform(), interpolator_type, img.GetOrigin(), new_spacing,
                                img.GetDirection(), 0.0, img.GetPixelIDValue())

        if apply_curve_smoothing:
            ct_scan = utils.smooth_images(ct_scan)
            print('apply_curve_smoothing')

        ct_scan = sitk.GetArrayFromImage(new_img)
        ct_scan[ct_scan > 1] = 1.
        ct_scan[ct_scan < 0] = 0.

        self._data = ct_scan
        volume.saveVolume(self._data, self._volname, 'preprocessed')
        self.load()

    def load(self):
        self._data = self._data.reshape(1, *self._data.shape)
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
        # print(self._model.summary())
        if configuration.parallel:
            from keras.utils import multi_gpu_model
            self._model = multi_gpu_model(self._model, gpus=configuration.number_of_gpus, cpu_relocation=False,
                                          cpu_merge=True)
            self._model = loader.Compile(self._model)
        else:
            # Compile the model.
            self._model = loader.Compile(self._model)
        self._model.load_weights(self._weightpath)

    def predict(self):
        print('Running prediction...')
        self._clip = [384, 0, 384, 0, 0, 100]
        mask = np.array(configuration.standard_volume)
        predict_set = np.vstack((self._data, self._zeros))  # np.load(self._filename)
        predict_set = predict_set.reshape(*predict_set.shape, 1)
        prelim = self._model.predict(predict_set)[-1]
        results = prelim[0]
        print(results.shape)
        mask = utils.smooth_contours(utils.smooth_contours(results, type='CV2'))
        volume.saveVolume(mask, self._volname, 'mask')
        # utils.multi_slice_viewer_legacy(mask.reshape(configuration.standard_volume),
        #                                 self._data.reshape(configuration.standard_volume))
        # plt.show()
        self.findBorder(mask)

    def findBorder(self, mask):
        # finds z limits
        for i in range(mask.shape[0]):
            if self.isEmpty(mask[i]):
                if (i - self._clip[4] <= 1):
                    self._clip[4] = i
            if not self.isEmpty(mask[i]):
                self._clip[5] = i + 1

        # finds x and y limits
        mask_xy = self.flatten(mask)
        for i in range(mask_xy.shape[0]):
            for j in range(mask_xy.shape[1]):
                if mask_xy[i][j]:
                    if self._clip[0] > j: self._clip[0] = j
                    if self._clip[1] < j: self._clip[1] = j
                    if self._clip[2] > i: self._clip[2] = i
                    if self._clip[3] < i: self._clip[3] = i

    def isEmpty(self, slice):
        if (np.count_nonzero(slice) == 0):
            return True
        return False

    def flatten(self, mask):
        flattened = np.zeros([384, 384])
        for i in range(mask.shape[1]):
            for j in range(mask.shape[2]):
                for k in range(mask.shape[0]):
                    if mask[k][i][j] == 1:
                        flattened[i][j] = 1
        return flattened

    def output(self):
        import h5py
        self._data = self._data.reshape(configuration.standard_volume)
        dim_orig = self._orig.shape
        self._clip[0] *= dim_orig[1] / configuration.standard_volume[1]
        self._clip[1] *= dim_orig[1] / configuration.standard_volume[1]
        self._clip[2] *= dim_orig[2] / configuration.standard_volume[2]
        self._clip[3] *= dim_orig[2] / configuration.standard_volume[2]
        self._clip[4] *= dim_orig[0] / configuration.standard_volume[0]
        self._clip[5] *= dim_orig[0] / configuration.standard_volume[0]
        self._clip = [int(i) for i in self._clip]
        clipped = self._orig[self._clip[4]: self._clip[5], self._clip[0]:self._clip[1], self._clip[2]:self._clip[3]]
        print('Saving to .h5 file.')
        volume.saveVolume(clipped, self._volname, 'output', 'h5')
        # sitkimg = sitk.GetImageFromArray(self._data)
        # sitk.WriteImage(sitkimg, 'data.mhd')

    def h5open(self):
        pass

if __name__ == '__main__':
    path = 'data/StudyCohort2.json'
    with open(path, 'r') as f:
        for i in range(len(json.load(f))):
            data = patient(i, path)
    print('Script complete. Exiting program now.')
