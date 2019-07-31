import SimpleITK as sitk
import matplotlib.pyplot as plt
from pylab import rcParams
import numpy as np
import matplotlib.gridspec as gridspec
import cv2
import sys, os
import shutil
sys.path.insert(0, 'MRI_Prostate_Segmentation')
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
        self.readJSON()
        self.createData()

    def run(self):
        for volume in self._volumes:
            volume.preprocess()
            volume.imgstats()

    def readJSON(self):
        path = self._jsonpath
        with open(path, 'r') as f:
            self._jsondata = json.load(f)[self._patientID]
        # wrapper, filename + (data -> AccessionID, Folder, FileList)

    def createData(self):
        self._volumes = np.array([])
        path = 'MRI_DataPreparation/MRI_cases_test'
        tgzpath = os.path.join(path, os.path.basename(self._jsondata["filename"]))
        print('Entering tgz directory: ' + tgzpath)
        if os.path.exists(tgzpath):
            with tarfile.open(tgzpath) as f:
                for i, fileset in enumerate(self._jsondata["data"]["FileList"]):
                    if (self._jsondata['data']['ImageType'][i] == 'PRIMARY_OTHER'):
                        tmp = volume(os.path.splitext(os.path.basename(tgzpath))[0] + '-' + self._jsondata['data']['ImageType'][i], os.path.splitext(os.path.basename(tgzpath))[0])
                        tmp.compile(fileset, f)
                        self._volumes = np.append(self._volumes, tmp)
                        # except NotImplementedError:
                        #     print('Skipping compressed image, directory: ' + self._jsondata["data"]["Folder"][i])
                    else:
                        print('No PRIMARY_OTHER type images. Datatype: ' + self._jsondata['data']['ImageType'][i])
                    # tmp = volume.getVolume(tmp)
                    # tmp.run()

    def outputData(self):
        pass

class volume:

    def __init__(self, volname, patient='', weightpath=r"C:/Users/Andrew Lu/Documents/Projects/MRI_Prostate_Segmentation/results/result_Prostate_D3_Segmentation_20190705-1805/weights-32.h5"):
        self._weightpath = weightpath
        self._volname = volname
        self._patient = volname if patient == '' else patient

    #
    # def __init__(self, filename, weightpath=r"results\weights-32.h5"):
    #     self._filename = filename
    #     self._weightpath = weightpath
    #     self._data = utils.LoadFile(self._filename,normalize='CLAHE')

    def saveVolume(self, outputtype, filetype='h5'):
        dir = 'MRI_DataPreparation/output/' + self._patient
        filename = self._volname + '-' + outputtype
        if os.path.exists(dir):
            pass
        else:
            os.mkdir(dir)
        if filetype=='h5':
            print('Saving to ' + dir + '/' + filename + '.h5...')
            h5f = h5py.File(dir + '/' + filename + '.h5', 'w')
            h5f.create_dataset('dataset', data=data)
            h5f.close()
            print('Volume saved. \n ----------')
        if filetype=='mhd':
            print('Saving to ' + dir + '/' + filename + '.mhd...')
            sitkimg = sitk.GetImageFromArray(data)
            sitkimg.SetSpacing(self._attr["spacing"])
            sitkimg.SetOrigin(self._attr["origin"])
            sitk.WriteImage(sitkimg, dir + '/' + filename + '.mhd')
            print('Volume saved. \n ----------')
        if filetype=='txt':
            text = open(dir + '/' + filename + '.txt', 'w')
            text.write('Image is empty.')
            text.close()

    @classmethod
    def saveVolumemhd(cls, data, volname, outputtype, spacing=(1,1,1), origin=(0,0,0)):
        print('Saving to output/' + volname + '-' + outputtype + '.mhd...')
        if os.path.exists('output'):
            pass
        else:
            os.mkdir('output')
        sitkimg = sitk.GetImageFromArray(data)
        sitkimg.SetSpacing(spacing)
        sitkimg.SetOrigin(origin)
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
        self._orig = []
        self._attr = {}
        for x in fileset:
            f.extract(x, path='temp-unzip')
            self._orig.append(pydicom.dcmread(os.path.join('temp-unzip', x)))
        self._orig.sort(key=lambda x: x.InstanceNumber)
        self._attr["spacing"] = (*self._orig[0].PixelSpacing, self._orig[0].SpacingBetweenSlices)
        self._attr["origin"]  = self._orig[0].ImagePosition
        self._data = np.array([self._orig[i].pixel_array for i in range(len(fileset))])

        self.saveVolume('original', 'mhd')

    def compile_mhd(self, filename):
        img = sitk.ReadImage(filename)
        self._data = sitk.GetArrayFromImage(img)
        self._spacing = img.GetSpacing()
        self._origin  = img.GetOrigin()
        self._orig = self._data.copy()
        volume.saveVolume(self._data,self._volname,'original', self._patient, 'h5')

    def compile_folder(self, folderpath):
        files = [os.path.join(folderpath, x) for x in os.listdir(folderpath) if
                 x.endswith('.dcm')]
        self._data = sorted([pydicom.read_file(x) for x in files], key=lambda x: x.InstanceNumber)
        self._data = np.array([self._data[i].pixel_array for i in range(len(files))])
        self._orig = self._data.copy()
        volume.saveVolume(self._data, self._volname, 'original', self._patient, 'h5')

    def preprocess(self, normalize='CLAHE', verbose=False, apply_curve_smoothing=False):
        # Normalize
        ct_scan = self._data.astype(np.float)
        if normalize=='CLAHE':
            ct_scan = utils.D3_equalize_adapthist(self._data, clip_limit=configuration.clip_limit, nbins=configuration.nbins)
        elif normalize == 'ZSCORE':
            ct_scan = (ct_scan - np.mean(ct_scan)) / (np.std(ct_scan))
        elif normalize == 'MAX':
            ct_scan = ct_scan / np.max(ct_scan)
        elif normalize == 'MAX_SPECIAL':
            for i in range(ct_scan.shape[0]):
                ct_scan[i] = ct_scan[i] / np.percentile(ct_scan[i], 95)
                ct_scan[i][ct_scan[i] > 1] = 1.0
        else:
            ct_scan = ct_scan
        ct_scan = ct_scan / np.max(ct_scan) # does not change result because max and min are 1.0 and 0.0 respectively
        # Sharpen
        # ct_scan = utils.Sharp3DVolume(ct_scan, type_of_sharpness=configuration.type_of_sharpness)
        ct_scan *= 0.9
        ct_scan = self.resize(ct_scan)
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
        self.saveVolume('preprocessed', 'mhd')

    def imgstats(self):
        print(self._patient)
        print(np.mean(self._data))
        print(np.median(self._data))
        print(np.std(self._data))
        print('-------------')

    def resize(self, ct_scan):
        if (ct_scan.shape[1] > 384):
            new_orig = int((ct_scan.shape[1] - configuration.standard_volume[1])/2)
            new_coord = (new_orig, new_orig+384, new_orig, new_orig+384)
            ct_scan = ct_scan[:, new_coord[0]: new_coord[1], new_coord[2]: new_coord[3]]
        return ct_scan

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
        self.load()
        print('Running prediction...')
        self._clip = [384, 0, 384, 0, 0, 300]
        mask = np.array(configuration.standard_volume)
        predict_set = np.vstack((self._data, self._zeros))  # np.load(self._filename)
        predict_set = predict_set.reshape(*predict_set.shape, 1)
        prelim = self._model.predict(predict_set)[-1]
        results = prelim[0]
        print(results.shape)
        mask = utils.smooth_contours(utils.smooth_contours(results, type='CV2'))
        volume.saveVolume(mask, self._volname, 'mask', self._patient)
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
        if clipped.size:
            print('Saving to .h5 file.')
            volume.saveVolume(clipped, self._volname, 'output', self._patient, 'h5')
        else:
            print('No output file.')
            volume.saveVolume(clipped, self._volname, 'output', self._patient, 'txt')
        # sitkimg = sitk.GetImageFromArray(self._data)
        # sitk.WriteImage(sitkimg, 'data.mhd')

    def h5open(self):
        pass

if __name__ == '__main__':
    path = 'MRI_DataPreparation/data/StudyCohort_cut.json'
    with open(path, 'r') as f:
        for i in range(len(json.load(f))):
            data = patient(i, path)
            data.run()
    print('Script complete. Exiting program now.')
