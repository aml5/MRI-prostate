import sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import SimpleITK as sitk
import matplotlib.pyplot as plt
from pylab import rcParams
import numpy as np
import matplotlib.gridspec as gridspec
import cv2
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
import math
import shutil
from h5Store import HDF5Store

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]= configuration.initial_gpu
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

output_dir = 'Z:\Projects\MRI_Database'#'MRI_DataPreparation/' #'MRI_DataPreparation/output'
json_filepath = './data/StudyCohort_prostate_mri_v4.json'
data_dir = "Z:\MRI_PRAD\MRI_PRAD" #'MRI_DataPreparation/MRI_cases_test'
weight_path = r"C:\Users\Andrew Lu\Documents\Projects\MRI_Prostate_Segmentation\results\result_Prostate_D3_Segmentation_20190705-1805\weights-32.h5"
step_output = False
keep_unzip = False
output_size = (144,144,16)
output_size_zyx = (output_size[2],output_size[1],output_size[0])
to_store = (output_size[2],output_size[1],output_size[0],1)
MEAN_THRESHOLD = 0.44
CLIP_MINIMUM=(50,50,7)  #zyx min
data_split = (.7, .2, .1)
image_types = {'T2': {'T2_Ax', 'DIXON_INPHASE'}, 'Secondary': {'ADC'}}
database_name = {'T2_Ax': 0, 'ADC': 0}
no_mask = []
UNZIP_TEMP_DIR = 'temp-unzip'

class dataset:

    def __init__(self,
                 jsonpath=json_filepath,
                 datapath=data_dir,
                 stepoutput=step_output):
        self._datapath = datapath
        self._jsonpath = jsonpath
        self._stepoutput = stepoutput
        

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
                            print(output_dir + '/' + imagetype + '.h5')
                            h5[imagetype] = h5py.File(output_dir + '/' + imagetype + '.h5', 'w')
                            h5[imagetype].create_group('images')
                            h5[imagetype].create_group('labels')
                        dataset.outputData(h5[imagetype], vol)
        for key, value in h5.items():
            value.close()
      
    def runPath(self):
        with open(self._jsonpath, 'r') as f:
            metafile = json.load(f)
            #Calculate the existing image number for each image type
            unique_list ={}
            for key in metafile:
                list_c = metafile[key]
                for v in list_c:
                    if 'ImageType' in v:
                        if  v['ImageType'] not in unique_list:
                            unique_list[v['ImageType']] = 0
                        unique_list[v['ImageType']] +=1
            print(unique_list)
            #Generate DataAdapter for each image type 
            h5_files = {}
            for key in database_name.keys():
                h5_files[key] = HDF5Store(output_dir + '/' + key + '.h5', shape=to_store)
            #Go through each patient and get the images that contain prostate
            for patientID, patient_data in metafile.items():
                data = patient(patientID, patient_data, self._datapath, stepoutput=self._stepoutput, verbose=True)
                vols = data.run()
                if vols is not None:

                    for vol in vols:
                        imagetype = vol.getImageType()
                        key=vol._attr['Image_Type']
                        if key in image_types['T2']:
                            if vol._data.shape == output_size_zyx:
                                number_of_images =h5_files["T2_Ax"].append(vol.getCategory(),vol._data, json.dumps(vol._attr, sort_keys=True))
                                print(f'Add volume to h5[T2_Ax], volume:{imagetype}-{number_of_images-1:04d}')
                            else:
                                print(f'Skip adding volume to h5[T2_Ax], volume:{imagetype}- shape {vol._data.shape}')
                        elif key in image_types['Secondary']:
                            number_of_images =h5_files[key].append(vol.getCategory(),vol._data, json.dumps(vol._attr, sort_keys=True))
                            print(f'Add volume to h5[{imagetype}], volume:{imagetype}-{number_of_images-1:04d}')
        dataset.reportNoMask(no_mask)

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
    def outputDataPath(cls, h5, imgs, labels, attrs,max_length):
        if len(imgs) > 0:
            if 'img' not in h5:
                data = h5.create_dataset('img', data=imgs, maxshape=(max_length, output_size_zyx,1))
            else:
                h5['img']
                data.append()
            #dataset.saveAttrs(imgs, attrs)
            h5.create_dataset('labels', data=labels)

    @classmethod
    def outputDataPath(cls, h5, imgs, labels, attrs):
        train_imgs, train_labels, train_attrs = [], [], []
        val_imgs, val_labels, val_attrs = [], [], []
        test_imgs, test_labels, test_attrs = [], [], []

        x = [i for i in range(len(imgs))]
        random.Random(19).shuffle(x)

        n_train = int(round(data_split[0] * len(imgs)))
        n_val = math.ceil((data_split[1] + data_split[0]) * len(imgs)) - n_train
        n_test = math.ceil(data_split[2] * len(imgs))

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

    @classmethod
    def reportNoMask(cls, list):
        csv = pd.DataFrame(list, columns=['patient', 'Echo_Time'])
        csv.to_csv('no_mask.csv')


class patient:
    def __init__(self, patientID, patient_data, datapath=None, stepoutput=False, verbose=False):
        self._patientID = patientID
        self._patientdata = patient_data
        self._datapath = datapath
        self._stepoutput = stepoutput
        self._verbose = verbose
        # self.readJSON()
        self.createData()

    def run(self):
        volumes = []
        for volume in self._volumes:
            try:
                volume.run()
                # volume.stats_hist()
                volumes.append(volume)
            except:
                print("can not generate volume file...")
        return volumes

    def run_stats(self):
        stats = []
        for volume in self._volumes:
            stats += volume.run_stats()
        return stats

    def readJSON_legacy(self):
        path = self._jsonpath
        with open(path, 'r') as f:
            self._jsondata = json.load(f)[self._patientID]
        # wrapper, filename + (data -> AccessionID, Folder, FileList)

    def createData(self):
        #self._volumes = np.array([])
        self._volumes = []
        zips = sorted(list(filter(lambda x: '.tgz' in x, os.listdir(data_dir))))
        tgzname = self._patientID.strip()
        # tgzpath = os.path.join(self._datapath, os.path.basename(self._jsondata["filename"]))
        # if os.path.exists(tgzpath):
        for file in zips:
            if tgzname + '.tgz' in file:
                if self._verbose: print('Entering tgz directory: ' + file)
                tgzpath = os.path.join(data_dir, file)
                patient_name = os.path.splitext(os.path.basename(tgzpath))[0]
                try:
                    with tarfile.open(tgzpath) as f:
                        # patient_data = self._jsondata[tgzname]
                        for i, series in enumerate(self._patientdata):
                            fileset = [x['fullpath'] for x in series['Files']]
                            # if (self._jsondata['data']['ImageType'][i] == 'PRIMARY_OTHER'):
                            #if self._verbose: print('PRIMARY_OTHER type images found.')
                            imgtype = series['ImageType']
                            if imgtype in image_types['T2'] | image_types['Secondary']:
                                tmp = volume(patient_name + '-' + imgtype, patient_name, imgtype, series['category'],
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
                except:
                    print("Cannot open the zipfile {tgzpath}, Escaping this file...")

class volume:

    model = None

    def __init__(self, volname, patient='', imagetype='', category='', stepoutput=False, verbose=False, weightpath=weight_path):
        self._weightpath = weightpath
        self._volname = volname
        self._imagetype = imagetype
        self._category = category
        self._stepoutput = stepoutput
        self._verbose = verbose
        self._patient = volname if patient == '' else patient

        if imagetype in image_types['T2']:
            self._id = database_name['T2_Ax']
            database_name['T2_Ax'] += 1
        elif imagetype in image_types['Secondary']:
            self._id = database_name['ADC']
            database_name['ADC'] += 1

    def run(self):
        if self._imagetype in image_types['T2']:
            self.preprocess()
            self.modelConfig()
            self.predict()
            if self.clip():
                self.resample(clip=True)
            elif self._attr['Echo_Time'] > 100:
                self.resample(clip=False)
        if self._imagetype in image_types['Secondary']:
            self.resample(clip=False)
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
            filename = f'{self._volname}-{self._id:04d}-{outputtype}'
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


    def compile(self, fileset, f):
        if self._verbose: print('Compiling images... ' + self._patient + '-' + self._imagetype)
        for x in fileset:
            f.extract(x, path=UNZIP_TEMP_DIR)
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
        self._attr['Mean'] = float(np.mean(self._data))
        self._attr['Max'] = float(np.max(self._data))
        self._attr['Min'] = float(np.min(self._data))
        self._attr['Stdev'] = float(np.std(self._data))
        self._attr['Proc_Mean'] = float(np.mean(self._data))
        # self.saveVolume(self._data, 'original', 'h5')
        self.saveVolume(self._data, 'original', 'mhd')

        if not keep_unzip:
            shutil.rmtree(UNZIP_TEMP_DIR)


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

    def trim(self, ct_scan, threshold):
        # trim black edges
        img = ct_scan[2]
        flat = np.sum(img, axis=0)
        left = 0
        right = len(flat)
        for i in range(len(flat)):
            if flat[i] <= threshold:
                left += 1
            else:
                break
        for i in range(len(flat)):
            if flat[len(flat)-i -1] <= threshold:
                right -= 1
            else:
                break
        if left != 0 or right != len(flat):
            ct_scan = ct_scan[:,:,left:right]
        return ct_scan



    def preprocess(self, normalize='CLAHE', verbose=False, apply_curve_smoothing=False):
        # Normalize
        ct_scan = self._data.astype(float)
        ct_scan /= np.max(ct_scan)
        ct_scan *= 1000
        ct_scan = ct_scan.astype(np.uint16)

        #trim image black edges
        ct_scan = self.trim(ct_scan, threshold=0.00001)
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
        ct_scan = utils.Sharp3DVolume(ct_scan, False,type_of_sharpness=configuration.type_of_sharpness)
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
        self._attr['Proc_Mean'] = float(np.mean(self._data))
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

    def getCategory(self):
        return self._category

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
        if volume.model is None:
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
                    volume.model = method_to_call()
            else:
                method_to_call = getattr(loader, configuration.select_model)  # 'CancerDetection_HarmonicSeries')
                volume.model = method_to_call()
            # print(volume.model.summary())
            if configuration.parallel:
                from keras.utils import multi_gpu_model
                volume.model = multi_gpu_model(volume.model, gpus=configuration.number_of_gpus, cpu_relocation=False,
                                              cpu_merge=True)
                volume.model = loader.Compile(volume.model)
            else:
                # Compile the model.
                volume.model = loader.Compile(volume.model)
            volume.model.load_weights(self._weightpath)

    def predict(self):
        self.load()
        if self._verbose: print('Running prediction...')
        predict_set = np.vstack((self._data, self._zeros))  # np.load(self._filename)
        self._data = self._data.reshape(configuration.standard_volume)
        predict_set = predict_set.reshape(*predict_set.shape, 1)
        prelim = volume.model.predict(predict_set)[-1]
        results = prelim[0]
        if self._verbose: print(results.shape)
        self._mask = utils.smooth_contours(utils.smooth_contours(results, type='CV2'))
        self.saveVolume(self._mask, 'mask', 'mhd')
        # utils.multi_slice_viewer_legacy(mask.reshape(configuration.standard_volume),
        #                                 self._data.reshape(configuration.standard_volume))
        # plt.show()
        if self.isEmpty(self._mask):
            self._clip = None
            no_mask.append((self._patient, self._attr['Echo_Time']))
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
        if self._clip is None:
            return False

        x_len = self._clip[1] - self._clip[0]
        y_len = self._clip[3] - self._clip[2]
        z_len = self._clip[5] - self._clip[4]
        if x_len >= CLIP_MINIMUM[0] and y_len >= CLIP_MINIMUM[1] and z_len >= CLIP_MINIMUM[2]:
            dim_orig = self._orig.GetSize()
            comp_factor = 0.05
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
                self.saveVolume(data, 'clipped', 'mhd')
            else:
                if self._verbose: print('No output file.')
                self.saveVolume(data, 'clipped', 'txt')

            self._sitk = clipped
            self._attr['Clipped_Pixel_Boundary'] = tuple(self._clip)

            if self._verbose: print(f'clipping...{dim_orig} to ({x_len},{y_len},{z_len})')
            return True
        else:
            if self._verbose:
                print(f'clipping...skipped. Too small ({x_len},{y_len},{z_len}) to clip.')
                no_mask.append(self._patient, self._attr['EchoTime'])
            return False

    def resample(self, clip):
        if clip:
            img = self._sitk
        else:
            img = self._orig
        new_Spacing = [old_sz * old_spc / new_sz for old_sz, old_spc, new_sz in
                       zip(img.GetSize(), img.GetSpacing(), output_size)]
        output_sitk = sitk.Resample(img, output_size, sitk.Transform(),
                                sitk.sitkLanczosWindowedSinc, img.GetOrigin(), new_Spacing,
                                img.GetDirection(), 0.0, img.GetPixelIDValue())
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
    global step_output
    global keep_unzip

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
    parser.add_argument(
        '--step_output',
        default = False,
        action = 'store_true',
        help = 'boolean to store intermediate image output in mhd format (e.g., original, preprocessed, mask)'
    )
    parser.add_argument(
        '--keep_unzip',
        default = False,
        action = 'store_true',
        help = 'boolean to keep intermediate unzipped dicom files'
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    json_filepath = args.json_path
    data_dir = args.data_dir
    weight_path = args.weight_path
    data_split = args.data_split
    step_output = args.step_output
    keep_unzip = args.keep_unzip

if __name__ == '__main__':
    argsparser()
    data = dataset(jsonpath=json_filepath, datapath=data_dir, stepoutput=step_output)
    data.runPath()

    print('Script complete. Exiting program now.')

    exit(0)