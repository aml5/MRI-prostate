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
import pandas as pd
import random
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]= configuration.initial_gpu
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))



output_dir = 'MRI_DataPreparation/output'
json_filepath = 'MRI_DataPreparation/data/StudyCohort_cut.json'
data_dir = 'MRI_DataPreparation/MRI_cases_test'
weight_path = r"C:\Users\Andrew Lu\Documents\Projects\MRI_Prostate_Segmentation\results\result_Prostate_D3_Segmentation_20190705-1805\weights-32.h5"
output_size = [144,144,16]
output_size_zyx = (output_size[2],output_size[1],output_size[0])
MEAN_THRESHOLD = 0.44
data_split = [.7, .2, .1]

block_id = []
# with open("AS_blocklist.csv") as f:
#     block_id = f.read().lower().splitlines()

class dataset:

    def __init__(self,
                 jsonpath=json_filepath,
                 datapath=data_dir):
        self._datapath = datapath
        self._jsonpath = jsonpath

    def run(self):
        h5 = {}
        with open(self._jsonpath, 'r') as f:
            metafile = json.load(f)
            for i in range(len(metafile)):
                data = patient(i, self._jsonpath, self._datapath, stepoutput=True, verbose=True)
                vols = data.run()
                if vols is not None:
                    for vol in vols:
                        imagetype = vol.getImageType()
                        if imagetype not in h5:
                            h5[imagetype] = h5py.File(output_dir + '/' + imagetype + '.h5', 'w')
                            h5[imagetype].create_group('images')
                            h5[imagetype].create_group('labels')
                        dataset.outputData(h5[imagetype], vol)
        for key, value in h5.items():
            value.close()

    def runPath(self):
        h5 = {}
        with open(self._jsonpath, 'r') as f:
            metafile = json.load(f)
            for i in range(len(metafile)):
                data = patient(i, self._jsonpath, self._datapath, stepoutput=True, verbose=True)
                vols = data.run()
                if vols is not None:
                    for vol in vols:
                        imagetype = vol.getImageType()
                        if imagetype not in h5:
                            h5[imagetype] = []
                            print(f'New h5[{imagetype}] ')
                        if vol._attr['Image_Type'] == 'PRIMARY_OTHER':
                            if vol._data.shape == output_size_zyx:
                                h5[imagetype].append((vol._data, vol.getLabel(), vol._attr))
                                print(f'Add volume to h5[{imagetype}], shape:{vol._data.shape}')
                            else:
                                print(f'Skip adding volume to h5[{imagetype}],wrong shape {vol._data.shape}')
                        else:
                            h5[imagetype].append((vol._data, vol.getLabel(), vol._attr))
                            print(f'Add volume to h5[{imagetype}], shape:{vol._data.shape}')
                        # print('shape:',vol._data.shape)
                        #     h5[imagetype].append((vol._data, vol.getLabel(), vol._attr))
        for key, value in h5.items():
            dataset.outputDataPath(h5py.File(output_dir + '/' + key + '.h5','w'), [i[0] for i in h5[key]], [i[1] for i in h5[key]], [i[2] for i in h5[key]])

    def stats(self):
        means = []
        with open(self._jsonpath, 'r') as f:
            metafile = json.load(f)
            for i in range(len(metafile)):
                data = patient(i, self._jsonpath, self._datapath, stepoutput=True, verbose=True)
                mean = data.run_stats()
                if mean is not None:
                    means += mean
        csv = pd.DataFrame(means, columns=['patient (+ img num)', 'mean', 'median', 'max',
                                           'mean/max', 'median/max', 'std/max', 'echo time'])
        csv.to_csv('stats.csv')

    @classmethod
    def outputData(cls, h5, vol):
        attrs = vol._attr
        data_array = vol._data
        img_group = h5.get('images')
        label_group = h5.get('labels')
        pixeldata = img_group.create_dataset(attrs['Patient'], data=data_array)
        pixeldata.attrs.create('Patient', attrs['Patient'].encode('ascii'))
        pixeldata.attrs.create('StudyUID', attrs['StudyUID'].encode('ascii'))
        pixeldata.attrs.create('SeriesUID', attrs['SeriesUID'].encode('ascii'))
        pixeldata.attrs.create('Image_Type', attrs['Image_Type'].encode('ascii'))
        pixeldata.attrs.create('SeriesNr', attrs['SeriesNr'])
        pixeldata.attrs.create('Age', attrs['Age'])
        pixeldata.attrs.create('Weight', attrs['Weight'])
        pixeldata.attrs.create('BMI', attrs['BMI'])
        pixeldata.attrs.create('Patient_Height', attrs['Patient_Height'])
        pixeldata.attrs.create('Size', attrs['Size'])
        pixeldata.attrs.create('Origin', attrs['Origin'])
        pixeldata.attrs.create('Spacing', attrs['Spacing'])
        pixeldata.attrs.create('Direction', attrs['Direction'])
        pixeldata.attrs.create('NumberOfComponentsPerPixel', attrs['NumberOfComponentsPerPixel'])
        pixeldata.attrs.create('Width', attrs['Width'])
        pixeldata.attrs.create('Height', attrs['Height'])
        pixeldata.attrs.create('Depth', attrs['Depth'])
        pixeldata.attrs.create('Clipped_Pixel_Boundary', attrs['Clipped_Pixel_Boundary'])
        pixeldata.attrs.create('Echo_Time', attrs['Echo_Time'])
        label_group.create_dataset(attrs['Patient'], data=vol.getLabel())

    @classmethod
    def outputDataPath(cls, h5, imgs, labels, attrs):
        train_imgs, train_labels, train_attrs = [], [], []
        val_imgs, val_labels, val_attrs = [], [], []
        test_imgs, test_labels, test_attrs = [], [], []

        from random import shuffle
        x = [i for i in range(len(imgs))]
        shuffle(x)

        n_train = int(round(data_split[0] * len(imgs)))
        n_val = int(round(data_split[1] * len(imgs)))
        n_test = int(round(data_split[2] * len(imgs)))

        for i in x[:n_train]:
            train_imgs.append(imgs[i])
            train_labels.append(labels[i])
            train_attrs.append(attrs[i])

        for i in x[n_train: n_train+n_val]:
            val_imgs.append(imgs[i])
            val_labels.append(labels[i])
            val_attrs.append(attrs[i])

        for i in x[len(x)-n_test:]:
            test_imgs.append(imgs[i])
            test_labels.append(labels[i])
            test_attrs.append(attrs[i])

        # for i in range(len(imgs)):
        #     rand = random.random()
        #     if rand < data_split[0]:
        #         train_imgs.append(imgs[i])
        #         train_labels.append(labels[i])
        #         train_attrs.append(attrs[i])
        #     elif rand < data_split[0] + data_split[1] and rand > data_split[0]:
        #         val_imgs.append(imgs[i])
        #         val_labels.append(labels[i])
        #         val_attrs.append(attrs[i])
        #     elif rand > 1 - data_split[2]:
        #         test_imgs.append(imgs[i])
        #         test_labels.append(labels[i])
        #         test_attrs.append(attrs[i])

        if len(train_imgs) > 0:
            train = h5.create_dataset('train_img', data=train_imgs)
            dataset.saveAttrs(train, train_attrs)
            h5.create_dataset('train_labels', data=train_labels)
            # h5.create_dataset('train_attrs', data=train_attrs)
        if len(val_imgs) > 0:
            val = h5.create_dataset('val_img', data=val_imgs)
            dataset.saveAttrs(val, val_attrs)
            h5.create_dataset('val_labels', data=val_labels)
            # h5.create_dataset('val_attrs', data=val_attrs)
        if len(test_imgs) > 0:
            test = h5.create_dataset('test_img', data=test_imgs)
            dataset.saveAttrs(test, test_attrs)
            h5.create_dataset('test_labels', data=test_labels)

    @classmethod
    def saveAttrs(cls, dataset, attrs):
        dataset.attrs.create('Patient', [attr['Patient'].encode('ascii') for attr in attrs])
        dataset.attrs.create('StudyUID', [attr['StudyUID'].encode('ascii') for attr in attrs])
        dataset.attrs.create('SeriesUID', [attr['SeriesUID'].encode('ascii') for attr in attrs])
        dataset.attrs.create('Image_Type', [attr['Image_Type'].encode('ascii') for attr in attrs])
        dataset.attrs.create('SeriesNr', [attr['SeriesNr'] for attr in attrs])
        dataset.attrs.create('Age', [attr['Age'] for attr in attrs])
        dataset.attrs.create('Weight', [attr['Weight'] for attr in attrs])
        dataset.attrs.create('BMI', [attr['BMI'] for attr in attrs])
        dataset.attrs.create('Patient_Height', [attr['Patient_Height'] for attr in attrs])
        dataset.attrs.create('Size', [attr['Size'] for attr in attrs])
        dataset.attrs.create('Origin', [attr['Origin'] for attr in attrs])
        dataset.attrs.create('Spacing', [attr['Spacing'] for attr in attrs])
        dataset.attrs.create('Direction', [attr['Direction'] for attr in attrs])
        dataset.attrs.create('NumberOfComponentsPerPixel', [attr['NumberOfComponentsPerPixel'] for attr in attrs])
        dataset.attrs.create('Width', [attr['Width'] for attr in attrs])
        dataset.attrs.create('Height', [attr['Height'] for attr in attrs])
        dataset.attrs.create('Depth', [attr['Depth'] for attr in attrs])
        dataset.attrs.create('Clipped_Pixel_Boundary', [attr['Clipped_Pixel_Boundary'] for attr in attrs])
        dataset.attrs.create('Echo_Time', [attr['Echo_Time'] for attr in attrs])

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
        volumes = []
        for volume in self._volumes:
            volume.run()
            # volume.stats_hist()
            volumes.append(volume)
        return volumes

    def run_stats(self):
        stats = []
        for volume in self._volumes:
            stats += volume.run_stats()
        return stats

    def readJSON(self):
        path = self._jsonpath
        with open(path, 'r') as f:
            self._jsondata = json.load(f)[self._patientID]
        # wrapper, filename + (data -> AccessionID, Folder, FileList)

    def createData(self):
        #self._volumes = np.array([])
        self._volumes = []
        tgzpath = os.path.join(self._datapath, os.path.basename(self._jsondata["filename"]))
        if self._verbose: print('Entering tgz directory: ' + tgzpath)
        if os.path.exists(tgzpath):
            with tarfile.open(tgzpath) as f:
                for i, fileset in enumerate(self._jsondata["data"]["FileList"]):
                    # if (self._jsondata['data']['ImageType'][i] == 'PRIMARY_OTHER'):
                    #if self._verbose: print('PRIMARY_OTHER type images found.')
                    tmp = volume(os.path.splitext(os.path.basename(tgzpath))[0] + '-' + self._jsondata['data']['ImageType'][i], os.path.splitext(os.path.basename(tgzpath))[0], self._jsondata['data']['ImageType'][i],
                                 stepoutput=self._stepoutput, weightpath=weight_path, verbose=self._verbose)
                    tmp.compile(fileset, f)
                    # if tmp.imgstats() > MEAN_THRESHOLD:

                    #self._volumes = np.append(self._volumes, tmp)
                    self._volumes.append(tmp)
                    # else:
                    #     if self._verbose: print(f'Non-T2 image removed.')
                    # except NotImplementedError:
                    #     print('Skipping compressed image, directory: ' + self._jsondata["data"]["Folder"][i])
                # else:
                #     if self._verbose: print('No PRIMARY_OTHER type images. Datatype: ' + self._jsondata['data']['ImageType'][i])
                    # tmp = volume.getVolume(tmp)
                    # tmp.run()

class volume:

    def __init__(self, volname, patient='', imagetype='', stepoutput=False, verbose=False, weightpath=weight_path):
        self._weightpath = weightpath
        self._volname = volname
        self._imagetype = imagetype
        self._stepoutput = stepoutput
        self._verbose = verbose
        self._patient = volname if patient == '' else patient

    #
    # def __init__(self, filename, weightpath=r"results\weights-32.h5"):
    #     self._filename = filename
    #     self._weightpath = weightpath
    #     self._data = utils.LoadFile(self._filename,normalize='CLAHE')

    def run(self):
        self.preprocess()
        if self._imagetype == 'PRIMARY_OTHER':
            self.modelConfig()
            self.predict()
            self.clip()
            self.postprocess()
        elif self._data.dtype == 'float64':
            self._data = self._data.astype(np.float16)
        # return self.output()

    def run_stats(self):
        stats = []
        stats += [[self._volname + '-total', np.mean(self._data), np.median(self._data), np.max(self._data), np.mean(self._data)/np.max(self._data), np.median(self._data)/np.max(self._data), np.std(self._data)/np.max(self._data), self._attr['Echo_Time']]]
        for i, data in enumerate(self._data):
            stats += [[self._volname + '-' + str(i), np.mean(data), np.median(data), np.max(data), np.mean(data)/np.max(data), np.median(data)/np.max(data), np.std(data)/np.max(data), self._attr['Echo_Time']]]
        self._data = self._data.astype(float)
        self._data /= np.max(self._data)
        self._data = utils.D3_equalize_adapthist(self._data, clip_limit=configuration.clip_limit, nbins=configuration.nbins)
        stats += [[self._volname + '-total (preprocessed)', np.mean(self._data), np.median(self._data), np.max(self._data),
                   np.mean(self._data) / np.max(self._data), np.median(self._data) / np.max(self._data),
                   np.std(self._data) / np.max(self._data), self._attr['Echo_Time']]]
        for i, data in enumerate(self._data):
            stats += [[self._volname + '-' + str(i) + ' preprocessed', np.mean(data), np.median(data), np.max(data),
                       np.mean(data) / np.max(data), np.median(data) / np.max(data), np.std(data) / np.max(data), self._attr['Echo_Time']]]
        return stats

    def saveVolume(self, data, outputtype, filetype='h5'):
        if self._stepoutput:
            dir = os.path.join(output_dir, self._patient)
            filename = self._volname + '-' + outputtype
            if os.path.exists(dir):
                pass
            else:
                os.makedirs(dir)
            if filetype=='h5':
                if self._verbose: print('Saving to ' + dir + '/' + filename + '.h5...')
                h5f = h5py.File(dir + '/' + filename + '.h5', 'w')
                data = h5f.create_dataset('dataset', data=data)
                data.attrs.create('Spacing', self._attr['Spacing'])
                data.attrs.create('Origin', self._attr['Origin'])
                h5f.close()

            if filetype=='mhd':
                if self._verbose: print('Saving to ' + dir + '/' + filename + '.mhd...')
                sitkimg = sitk.GetImageFromArray(data)
                sitkimg.SetSpacing(self._attr["Spacing"])
                sitkimg.SetOrigin(self._attr["Origin"])
                sitk.WriteImage(sitkimg, dir + '/' + filename + '.mhd')

            if filetype=='txt':
                text = open(dir + '/' + filename + '.txt', 'w')
                text.write('Image is empty.')
                text.close()

    # @classmethod
    # def saveVolumemhd(cls, data, volname, outputtype, Spacing=(1,1,1), Origin=(0,0,0)):
    #     print('Saving to output/' + volname + '-' + outputtype + '.mhd...')
    #     if os.path.exists('output'):
    #         pass
    #     else:
    #         os.mkdir('output')
    #     sitkimg = sitk.GetImageFromArray(data)
    #     sitkimg.SetSpacing(Spacing)
    #     sitkimg.SetOrigin(Origin)
    #     sitk.WriteImage(sitkimg, 'output/' + volname + '-' + outputtype + '.mhd')
    #     print('Volume saved. \n ----------')

    def compile(self, fileset, f):
        if self._verbose: print('Compiling images...')
        for x in fileset:
            f.extract(x, path='temp-unzip')
        folderpath, _ = os.path.split(fileset[0])
        self._orig, reader = self.loadVolume(os.path.join('temp-unzip', folderpath))
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
        self._attr['Image_Type'] = self._imagetype
        self._attr['Age'] = int(reader.GetMetaData(1, '0010|1010').strip()[:3]) if '0010|1010' in reader.GetMetaDataKeys(1) else -1
        self._attr['Weight'] = reader.GetMetaData(1, '0010|1030').strip() if '0010|1030' in reader.GetMetaDataKeys(1) else -1
        self._attr['Patient_Height'] = reader.GetMetaData(1, '0010|1020').strip() if '0010|1020' in reader.GetMetaDataKeys(1) else -1
        self._attr['BMI'] = self._attr['Weight'] / (self._attr['Patient_Height']/100)**2 if self._attr['Weight'] > 0 and self._attr['Patient_Height'] > 0 else -1
        self._attr['StudyUID'] = reader.GetMetaData(1, '0020|000d')
        self._attr['SeriesUID'] = reader.GetMetaData(1, '0020|000e')
        self._attr['SeriesNr'] = reader.GetMetaData(1, '0020|0011').strip() if '0020|1011' in reader.GetMetaDataKeys(1) else -1
        self._attr['Size'] = size_array
        self._attr['Spacing'] = spacing_array
        self._attr['Origin']  = origin_array
        self._attr['Direction'] = direction_array
        self._attr['NumberOfComponentsPerPixel'] = ComponentsPerPixel_array
        self._attr['Width'] = width_array
        self._attr['Height'] = height_array
        self._attr['Depth'] = depth_array
        self._attr['Echo_Time'] = float(reader.GetMetaData(1, '0018|0081').strip())
        self._attr['Patient Description'] = reader.GetMetaData(1, '0008|103e') if '0008|103e' in reader.GetMetaDataKeys(1) else 'No Description'
        self._attr['Patient Path'] = os.path.join(self._patient, self._imagetype)
        self._attr['Clipped_Pixel_Boundary'] = (0, self._attr['Width'], 0, self._attr['Height'], 0, self._attr['Depth'])
        self._data = sitk.GetArrayFromImage(self._orig)
        # self.saveVolume(self._data, 'original', 'h5')
        self.saveVolume(self._data, 'original', 'mhd')

    def compile_mhd(self, filename):
        img = sitk.ReadImage(filename)
        self._orig = img.copy()
        self._attr = {}
        self._attr['Spacing'] = img.GetSpacing()
        self._attr['Origin']  = img.GetOrigin()
        self._data = sitk.GetArrayFromImage(self._orig)
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
        dicom_names = self.filterRepeats(dicom_names)
        reader.SetFileNames(dicom_names)
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()
        dicom_series = reader.Execute()
        return dicom_series, reader  # SimpleITK.GetArrayFromImage(dicom_series)

    def filterRepeats(self, files):
        files_filtered = []
        for i, filename in enumerate(files):
            mri = sitk.ReadImage(filename)
            space_z = float(mri.GetMetaData('0020|0032').split('\\')[2])
            if i == 0:
                prev_z = space_z
            if abs(space_z - prev_z > 10.0):
                break
            else:
                files_filtered.append((filename, space_z))
                prev_z = space_z
        files_filtered = sorted(files_filtered, key=lambda x:x[1])
        files_filtered = [x[0] for x in files_filtered]
        return files_filtered

    def preprocess(self, normalize='CLAHE', verbose=False, apply_curve_smoothing=False):
        # Normalize
        ct_scan = self._data.astype(float)
        ct_scan /= np.max(ct_scan)
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
        self.setAttributes(img)

        if verbose:
            print('img', img.GetDimension(), img.GetDirection(), img.GetOrigin(), img.GetSpacing())

        # Resize Standard
        new_x_size = configuration.standard_volume[1]
        new_y_size = configuration.standard_volume[2]
        new_z_size = configuration.standard_volume[0]

        # Create the reference image with a zero Origin, identity direction cosine matrix and dimension
        new_size = [new_x_size, new_y_size, new_z_size]
        new_Spacing = [old_sz * old_spc / new_sz for old_sz, old_spc, new_sz in
                       zip(img.GetSize(), img.GetSpacing(), new_size)]

        interpolator_type = sitk.sitkLanczosWindowedSinc
        new_img = sitk.Resample(img, new_size, sitk.Transform(), interpolator_type, img.GetOrigin(), new_Spacing,
                                img.GetDirection(), 0.0, img.GetPixelIDValue())

        if apply_curve_smoothing:
            new_img = utils.smooth_images(new_img)
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
        # self.saveVolume(self._data, 'preprocessed', 'h5')
        self.saveVolume(self._data, 'preprocessed', 'mhd')

    def setAttributes(self, img):
        self._attr['Spacing'] = img.GetSpacing()
        self._attr['Origin'] = img.GetOrigin()

    def getAttributes(self, img):
        img.SetSpacing(self._attr['Spacing'])
        img.SetOrigin(self._attr['Origin'])

    def getImageType(self):
        return self._imagetype

    def getLabel(self):
        AccessionID = self._patient.split('-')[1].lower()
        if "n" in AccessionID:
            label = 0
        elif "c" in AccessionID:
            label = 2
        elif "wm" in AccessionID:
            label = 2
        elif "b" in AccessionID:
            if AccessionID not in block_id:
                label = 1
            else:
                label = 2
        return label

    def imgStats(self):
        if self._verbose: print(f'{self._patient}-{self._imagetype}: mean: {np.mean(self._data):.5f}, median: {np.median(self._data):.5f}, std: {np.std(self._data):.5f}, max: {np.max(self._data):.5f}')
        return np.mean(self._data)

    def statsHist(self):
        plt.subplot(131), plt.hist(self._data.flatten()/1000, range=(0,1)), plt.ylim(0,2000000), plt.title('original')
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
        predict_set = np.vstack((self._data, self._zeros))  # np.load(self._filename)
        self._data = self._data.reshape(configuration.standard_volume)
        predict_set = predict_set.reshape(*predict_set.shape, 1)
        prelim = self._model.predict(predict_set)[-1]
        results = prelim[0]
        if self._verbose: print(results.shape)
        self._mask = utils.smooth_contours(utils.smooth_contours(results, type='CV2'))
        # self.saveVolume(mask, 'mask', 'h5')
        self.saveVolume(self._mask, 'mask', 'mhd')
        # utils.multi_slice_viewer_legacy(mask.reshape(configuration.standard_volume),
        #                                 self._data.reshape(configuration.standard_volume))
        # plt.show()
        if self.isEmpty(self._mask):
            self._clip = None
        else:
            self.findBorder(self._mask)

    def findBorder(self, mask):
        clip = [384, 0, 384, 0, 0, 300]
        # finds z limits
        for i in range(mask.shape[0]):
            if self.isEmpty(mask[i]):
                if (i - clip[4] <= 1):
                    clip[4] = i
            if not self.isEmpty(mask[i]):
                clip[5] = i + 1

        # finds x and y limits
        mask_xy = self.flatten(mask)
        for i in range(mask_xy.shape[0]):
            for j in range(mask_xy.shape[1]):
                if mask_xy[i][j]:
                    if clip[0] > j: clip[0] = j
                    if clip[1] < j: clip[1] = j
                    if clip[2] > i: clip[2] = i
                    if clip[3] < i: clip[3] = i
        self._clip = clip

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

    def clip(self):
        if self._clip is not None:
            dim_orig = self._orig.GetSize()
            comp_factor = 0.05
            x_len = self._clip[1] - self._clip[0]
            y_len = self._clip[3] - self._clip[2]
            z_len = self._clip[5] - self._clip[4]
            x_adj = np.ceil(x_len * comp_factor)
            y_adj = np.ceil(y_len * comp_factor)
            z_adj = 1 # np.ceil(z_len * comp_factor)

            self._clip[0] -= min(x_adj, self._clip[0])
            self._clip[0] *= dim_orig[0] / configuration.standard_volume[1]
            self._clip[1] += min(x_adj, dim_orig[0] - self._clip[1])
            self._clip[1] *= dim_orig[0] / configuration.standard_volume[1]
            self._clip[2] -= min(y_adj, self._clip[2])
            self._clip[2] *= dim_orig[1] / configuration.standard_volume[2]
            self._clip[3] += min(y_adj, dim_orig[1] - self._clip[3])
            self._clip[3] *= dim_orig[1] / configuration.standard_volume[2]
            self._clip[4] -= min(z_adj, self._clip[4])
            self._clip[4] *= dim_orig[2] / configuration.standard_volume[0]
            self._clip[5] += min(z_adj, dim_orig[2] - self._clip[5])
            self._clip[5] *= dim_orig[2] / configuration.standard_volume[0]
            self._clip = [int(i) for i in self._clip]

            clipped = self._orig[self._clip[0] : self._clip[1], self._clip[2] : self._clip[3], self._clip[4] : self._clip[5]]
            self.setAttributes(clipped)

            data = sitk.GetArrayFromImage(clipped)
            if data.size:
                if self._verbose: print('Saving to .mhd file.')
                # self.saveVolume(data, 'clipped', 'h5')
                self.saveVolume(data, 'clipped', 'mhd')
            else:
                if self._verbose: print('No output file.')
                self.saveVolume(data, 'clipped', 'txt')
            # sitkimg = sitk.GetImageFromArray(self._data)
            # sitk.WriteImage(sitkimg, 'data.mhd')
            self._sitk = clipped
            self._attr['Clipped_Pixel_Boundary'] = tuple(self._clip)

    def postprocess(self):
        if self._clip:
            new_Spacing = [old_sz * old_spc / new_sz for old_sz, old_spc, new_sz in
                           zip(self._sitk.GetSize(), self._sitk.GetSpacing(), output_size)]
            output_sitk = sitk.Resample(self._sitk, output_size, sitk.Transform(),
                                    sitk.sitkLanczosWindowedSinc, self._sitk.GetOrigin(), new_Spacing,
                                    self._sitk.GetDirection(), 0.0, self._sitk.GetPixelIDValue())
            self.setAttributes(output_sitk)
            self._data = sitk.GetArrayFromImage(output_sitk)
            # self.saveVolume(self._data, 'output', 'h5')
            self.saveVolume(self._data, 'output', 'mhd')

    def output(self):
        return self

    def h5open(self):
        pass

def argsparser():
    global output_dir
    global json_filepath
    global data_dir
    global weight_path
    global data_split

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-o', '--output_dir',
        default = output_dir,
        help = 'directory where h5 files are to be stored'
    )
    parser.add_argument(
        '-j', '--json_path',
        default = json_filepath,
        help = 'path of Study Cohort json file'
    )
    parser.add_argument(
        '-d', '--data_dir',
        default = data_dir,
        help = 'directory where tgz files are stored'
    )
    parser.add_argument(
        '-w', '--weight_path',
        default = weight_path,
        help = 'path of model weight',
    )
    parser.add_argument(
        '-s', '--data_split',
        default = data_split,
        nargs = 3,
        help = 'split for training, validation, and test sets in decimal form'
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    json_filepath = args.json_path
    data_dir = args.data_dir
    weight_path = args.weight_path
    data_split = args.data_split

if __name__ == '__main__':
    argsparser()
    data = dataset(jsonpath=json_filepath, datapath=data_dir)
    data.runPath()
    # data = volume('test')
    # data.compile_folder('temp-unzip/1.2.840.4267.32.316936701248529032407369277386144371677/1.2.840.4267.32.226789496748388476211964232698380117696')
    # data.run()
    print('Script complete. Exiting program now.')
    # data = volume('case12', '', True)
    # data.compile_mhd(r'MRI_Prostate_Segmentation/train/Case12.mhd')
    # data.preprocess()
    # # data.stats_hist()
    # data.run()
    exit(0)