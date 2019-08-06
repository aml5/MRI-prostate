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

class dataset:

    def __init__(self,
                 jsonpath='MRI_DataPreparation/data/StudyCohort_cut.json',
                 datapath='MRI_DataPreparation/MRI_cases_test'):
        self._datapath = datapath
        self._jsonpath = jsonpath

    def run(self):
        h5 = h5py.File('data.h5','w')
        with open(self._jsonpath, 'r') as f:
            metafile = json.load(f)
            for i in range(len(metafile)):
                data = patient(i, self._jsonpath, self._datapath, stepoutput=False, verbose=True)
                vol, attrs = data.run()
                dataset.outputData(h5, vol, attrs)
        h5.close()

    @classmethod
    def outputData(cls, h5, vol, attrs):
        group = h5.create_group(attrs['Patient Path'])
        pixeldata = group.create_dataset(attrs['Patient'], data=vol)
        pixeldata.attrs.create('Age', attrs['Age'])
        pixeldata.attrs.create('Weight', attrs['Weight'])
        pixeldata.attrs.create('BMI', attrs['BMI'])
        pixeldata.attrs.create('Patient_Height', attrs['Patient_Height'])
        pixeldata.attrs.create('SeriesNr', attrs['SeriesNr'])
        pixeldata.attrs.create('Size', attrs['Size'])
        pixeldata.attrs.create('Origin', attrs['Origin'])
        pixeldata.attrs.create('Spacing', attrs['Spacing'])
        pixeldata.attrs.create('Direction', attrs['Direction'])
        pixeldata.attrs.create('NumberOfComponentsPerPixel', attrs['NumberOfComponentsPerPixel'])
        pixeldata.attrs.create('Width', attrs['Width'])
        pixeldata.attrs.create('Height', attrs['Height'])
        pixeldata.attrs.create('Depth', attrs['Depth'])

class patient:

    def __init__(self, patientID, jsonpath, datapath=None, stepoutput=False, verbose=False):
        self._patientID = patientID
        self._jsonpath = jsonpath
        self._datapath = datapath
        self._stepoutput = stepoutput
        self._verbose = verbose
        self.readJSON()
        self.createData()

    def run(self):
        for volume in self._volumes:
            # volume.preprocess()
            # volume.imgstats()
            return volume.run() # for now
            # volume.stats_hist()

    def readJSON(self):
        path = self._jsonpath
        with open(path, 'r') as f:
            self._jsondata = json.load(f)[self._patientID]
        # wrapper, filename + (data -> AccessionID, Folder, FileList)

    def createData(self):
        self._volumes = np.array([])
        tgzpath = os.path.join(self._datapath, os.path.basename(self._jsondata["filename"]))
        if self._verbose: print('Entering tgz directory: ' + tgzpath)
        if os.path.exists(tgzpath):
            with tarfile.open(tgzpath) as f:
                for i, fileset in enumerate(self._jsondata["data"]["FileList"]):
                    if (self._jsondata['data']['ImageType'][i] == 'PRIMARY_OTHER'):
                        if self._verbose: print('PRIMARY_OTHER type images found.')
                        tmp = volume(os.path.splitext(os.path.basename(tgzpath))[0] + '-' + self._jsondata['data']['ImageType'][i], os.path.splitext(os.path.basename(tgzpath))[0], verbose=self._verbose)
                        tmp.compile(fileset, self._jsondata['data']['Folder'][i], f)
                        # if tmp.imgstats() > 0.47:
                        self._volumes = np.append(self._volumes, tmp)
                        # else:
                        #     if self._verbose: print(f'Non-T2 image removed.')
                        # except NotImplementedError:
                        #     print('Skipping compressed image, directory: ' + self._jsondata["data"]["Folder"][i])
                    else:
                        if self._verbose: print('No PRIMARY_OTHER type images. Datatype: ' + self._jsondata['data']['ImageType'][i])
                    # tmp = volume.getVolume(tmp)
                    # tmp.run()

class volume:

    def __init__(self, volname, patient='', stepoutput=False, verbose=False, weightpath=r"C:\Users\Andrew Lu\Documents\Projects\MRI_Prostate_Segmentation\results\result_Prostate_D3_Segmentation_20190705-1805\weights-32.h5"):
        self._weightpath = weightpath
        self._volname = volname
        self._stepoutput = stepoutput
        self._verbose = verbose
        self._patient = volname if patient == '' else patient

    #
    # def __init__(self, filename, weightpath=r"results\weights-32.h5"):
    #     self._filename = filename
    #     self._weightpath = weightpath
    #     self._data = utils.LoadFile(self._filename,normalize='CLAHE')

    def saveVolume(self, data, outputtype, filetype='h5'):
        if self._stepoutput:
            dir = 'MRI_DataPreparation/output/' + self._patient
            filename = self._volname + '-' + outputtype
            if os.path.exists(dir):
                pass
            else:
                os.mkdir(dir)
            if filetype=='h5':
                if self._verbose: print('Saving to ' + dir + '/' + filename + '.h5...')
                h5f = h5py.File(dir + '/' + filename + '.h5', 'w')
                data = h5f.create_dataset('dataset', data=data)
                data.attrs.create('spacing', self._attr['spacing'])
                data.attrs.create('origin', self._attr['origin'])
                h5f.close()
                if self._verbose: print('Volume saved. \n ----------')
            if filetype=='mhd':
                if self._verbose: print('Saving to ' + dir + '/' + filename + '.mhd...')
                sitkimg = sitk.GetImageFromArray(data)
                sitkimg.SetSpacing(self._attr["spacing"])
                sitkimg.SetOrigin(self._attr["origin"])
                sitk.WriteImage(sitkimg, dir + '/' + filename + '.mhd')
                if self._verbose: print('Volume saved. \n ----------')
            if filetype=='txt':
                text = open(dir + '/' + filename + '.txt', 'w')
                text.write('Image is empty.')
                text.close()

    # @classmethod
    # def saveVolumemhd(cls, data, volname, outputtype, spacing=(1,1,1), origin=(0,0,0)):
    #     print('Saving to output/' + volname + '-' + outputtype + '.mhd...')
    #     if os.path.exists('output'):
    #         pass
    #     else:
    #         os.mkdir('output')
    #     sitkimg = sitk.GetImageFromArray(data)
    #     sitkimg.SetSpacing(spacing)
    #     sitkimg.SetOrigin(origin)
    #     sitk.WriteImage(sitkimg, 'output/' + volname + '-' + outputtype + '.mhd')
    #     print('Volume saved. \n ----------')

    def run(self):
        self.preprocess()
        self.modelConfig()
        self.predict()
        return self.output()

    def compile(self, fileset, folderpath, f):
        if self._verbose: print('Compiling images...')
        for x in fileset:
            f.extractall(x, path='temp-unzip')
        self._orig, reader = self.loadVolume(sys.path.join('temp-unzip', folderpath))
        size_array = self._orig.GetSize()
        origin_array = self._orig.GetOrigin()
        spacing_array = self._orig.GetSpacing()
        direction_array = self._orig.GetDirection()
        ComponentsPerPixel_array = self._orig.GetNumberOfComponentsPerPixel()
        width_array = self._orig.GetWidth()
        height_array = self._orig.GetHeight()
        depth_array = self._orig.GetDepth()
        self._attr = {}
        self._attr['Patient'] = self._patient
        self._attr['Age'] = reader.GetMetaData(1, '0010|1010').strip() if '0010|1010' in reader.GetMetaDataKeys(1) else -1
        self._attr['Weight'] = reader.GetMetaData(1, '0010|1030').strip() if '0010|1030' in reader.GetMetaDataKeys(1) else -1
        self._attr['Patient_Height'] = reader.GetMetaData(1, '0010|1020').strip() if '0010|1020' in reader.GetMetaDataKeys(1) else -1
        self._attr['BMI'] = self._attr['Weight'] / (self._attr['Patient_Height']/100)**2
        self._attr['SeriesNr'] = reader.GetMetaData(1, '0020|0011').strip() if '0020|1011' in reader.GetMetaDataKeys(1) else -1
        self._attr['Size'] = size_array
        self._attr['Spacing'] = spacing_array
        self._attr['Origin']  = origin_array
        self._attr['Direction'] = direction_array
        self._attr['NumberOfComponentsPerPixel'] = ComponentsPerPixel_array
        self._attr['Width'] = width_array
        self._attr['Height'] = height_array
        self._attr['Depth'] = depth_array
        self._attr['Patient Description'] = 'test'
        self._attr['Patient Path'] = self._attr['Patient'] + '/' + self._attr['Patient Description']
        self._data = self._orig.GetArrayFromImage()
        self.saveVolume(self._data, 'original', 'h5')
        self.saveVolume(self._data, 'original', 'mhd')

    def compile_mhd(self, filename):
        img = sitk.ReadImage(filename)
        self._orig = sitk.GetArrayFromImage(img)
        self._attr = {}
        self._attr['Spacing'] = img.GetSpacing()
        self._attr['Origin']  = img.GetOrigin()
        self._data = self._orig.copy()
        self.saveVolume(self._data, 'original', 'h5')
        self.saveVolume(self._data, 'original', 'mhd')

    def compile_folder(self, folderpath):
        files = [os.path.join(folderpath, x) for x in os.listdir(folderpath) if
                 x.endswith('.dcm')]
        self._orig = sorted([pydicom.read_file(x) for x in files], key=lambda x: x.ImagePositionPatient[2])
        self._data = np.array([self._orig[i].pixel_array for i in range(len(files))])
        self.saveVolume(self._data, 'original', 'h5')
        self.saveVolume(self._data, 'original', 'mhd')

    def loadVolume(self, dir):
        """Reads an entire DICOM series of slices from 'input_dir' and returns its pixel data as an array."""
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(dir)
        # Sort the dicom files
        # dicom_names = SortDicomFiles(dicom_names)
        reader.SetFileNames(dicom_names)
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()
        dicom_series = reader.Execute()
        return dicom_series, reader  # SimpleITK.GetArrayFromImage(dicom_series)

    def preprocess(self, normalize='CLAHE', verbose=False, apply_curve_smoothing=False):
        # Normalize
        ct_scan = self._data
        # ct_scan = self.resize(ct_scan)
        if normalize=='CLAHE':
            ct_scan = utils.D3_equalize_adapthist(ct_scan, clip_limit=configuration.clip_limit, nbins=configuration.nbins)
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
        # print(f'{self._patient}: mean: {np.mean(ct_scan):.5f}, median: {np.median(ct_scan):.5f}, std: {np.std(ct_scan):.5f}')
        ct_scan = utils.Sharp3DVolume(ct_scan, type_of_sharpness=configuration.type_of_sharpness)
        # print(f'{self._patient}: mean: {np.mean(ct_scan):.5f}, median: {np.median(ct_scan):.5f}, std: {np.std(ct_scan):.5f}')
        # print(f'{self._patient}: mean: {np.mean(ct_scan):.5f}, median: {np.median(ct_scan):.5f}, std: {np.std(ct_scan):.5f}')
        img = sitk.GetImageFromArray(ct_scan)
        self.getAttributes(img)

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

        self.setAttributes(new_img)
        ct_scan = sitk.GetArrayFromImage(new_img)

        # if np.mean(ct_scan) > 0.47:
        # # print(f'{self._patient}: mean: {np.mean(ct_scan)}, median: {np.median(ct_scan):.5f}, std: {np.std(ct_scan):.5f}')
        #     med_training = 0.5091
        #     # ct_scan *= med_training / np.median(ct_scan)
        #     ct_scan *+ 1.15
        # print(f'{self._patient}: mean: {np.mean(ct_scan)}, median: {np.median(ct_scan):.5f}, std: {np.std(ct_scan):.5f}')
        ct_scan[ct_scan > 1] = 1.
        ct_scan[ct_scan < 0] = 0.
        # print(f'{self._patient}: mean: {np.mean(ct_scan):.5f}, median: {np.median(ct_scan):.5f}, std: {np.std(ct_scan):.5f}')

        self._data = ct_scan
        self.saveVolume(self._data, 'preprocessed', 'h5')
        self.saveVolume(self._data, 'preprocessed', 'mhd')

    def setAttributes(self, img):
        self._attr['Spacing'] = img.GetSpacing()
        self._attr['Origin'] = img.GetOrigin()

    def getAttributes(self, img):
        img.SetSpacing(self._attr['Spacing'])
        img.SetOrigin(self._attr['Origin'])

    def imgstats(self):
        if self._verbose: print(f'{self._patient}: mean: {np.mean(self._data):.5f}, median: {np.median(self._data):.5f}, std: {np.std(self._data):.5f}')
        return np.mean(self._data)

    def stats_hist(self):
        plt.subplot(131), plt.hist(self._data.flatten()/1000, range=(0,1)), plt.ylim(0,2000000), plt.title('Original')
        plt.subplot(132), plt.hist(self.resize(self._data).flatten()/1000, range=(0,1)), plt.ylim(0,2000000), plt.title('Cropped')
        self.preprocess()
        plt.subplot(133), plt.hist(self._data.flatten()), plt.ylim(0,2000000), plt.title('Preprocessed')
        plt.show()

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
        if self._verbose: print('Running prediction...')
        self._clip = [384, 0, 384, 0, 0, 300]
        predict_set = np.vstack((self._data, self._zeros))  # np.load(self._filename)
        predict_set = predict_set.reshape(*predict_set.shape, 1)
        prelim = self._model.predict(predict_set)[-1]
        results = prelim[0]
        if self._verbose: print(results.shape)
        mask = utils.smooth_contours(utils.smooth_contours(results, type='CV2'))
        self.saveVolume(mask, 'mask', 'h5')
        self.saveVolume(mask, 'mask', 'mhd')
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
        self._data = self._data.reshape(configuration.standard_volume)
        clipped = self._orig.GetArrayFromImage()
        dim_orig = clipped.shape
        comp_factor = 0.05
        x_len = self._clip[1] - self._clip[0]
        y_len = self._clip[3] - self._clip[2]
        z_len = self._clip[5] - self._clip[4]
        x_adj = np.ceil(x_len * comp_factor)
        y_adj = np.ceil(y_len * comp_factor)
        z_adj = 1 # np.ceil(z_len * comp_factor)
        self._clip[0] -= x_adj
        self._clip[0] *= dim_orig[1] / configuration.standard_volume[1]
        self._clip[1] += x_adj
        self._clip[1] *= dim_orig[1] / configuration.standard_volume[1]
        self._clip[2] -= y_adj
        self._clip[2] *= dim_orig[2] / configuration.standard_volume[2]
        self._clip[3] += y_adj
        self._clip[3] *= dim_orig[2] / configuration.standard_volume[2]
        self._clip[4] -= z_adj
        self._clip[4] *= dim_orig[0] / configuration.standard_volume[0]
        self._clip[5] += z_adj
        self._clip[5] *= dim_orig[0] / configuration.standard_volume[0]
        self._clip = [int(i) for i in self._clip]
        clipped = clipped[self._clip[4]: self._clip[5], self._clip[0]:self._clip[1], self._clip[2]:self._clip[3]]
        if clipped.size:
            if self._verbose: print('Saving to .h5 file.')
            self.saveVolume(clipped, 'output', 'h5')
            self.saveVolume(clipped, 'output', 'mhd')
        else:
            if self._verbose: print('No output file.')
            self.saveVolume(clipped, 'output', 'txt')
        # sitkimg = sitk.GetImageFromArray(self._data)
        # sitk.WriteImage(sitkimg, 'data.mhd')
        return clipped, self._attr

    def h5open(self):
        pass

if __name__ == '__main__':
    data = dataset()
    data.run()
    print('Script complete. Exiting program now.')
    # data = volume('case12', '', True)
    # data.compile_mhd(r'MRI_Prostate_Segmentation/train/Case12.mhd')
    # data.preprocess()
    # # data.stats_hist()
    # data.run()