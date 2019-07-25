# %%
import SimpleITK
import SimpleITK as sitk
import cv2
import numpy as np
import glob
import os
import h5py
import csv
import argparse
from os import path, listdir, walk
import matplotlib.gridspec as gridspec
from tempfile import NamedTemporaryFile
import pandas as pd
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from skimage import exposure
import matplotlib.patches as patches
import PIL
from PIL import Image


# %%
#######
# General Functions for data storage
#######
def difference_btn_twolist(lst1, lst2):
    lst3 = [value for value in lst1 if value not in lst2]
    return lst3


def multi_slice_viewer(volume, reference, depth=1):
    remove_keymap_conflicts({'j', 'k'})
    '''
    img_mask = np.zeros((volume.shape[0],volume.shape[1],volume.shape[2],3))
    for i, x in enumerate(reference):
        if depth==1:
            for j in range(3):
                img_mask[i,:,:,j] = x
            mask = volume[i] == 1
        else:
            img_mask[i,:,:,j] = x
            if volume.shape[3]==3:
                mask = np.logical_or(volume[i,:,:,2]>=0.43,volume[i,:,:,1]>0.43)

        mask_emp = np.zeros((x.shape[0],x.shape[0]), dtype=np.uint8)
        mask_emp[mask] = 255
        im2, contours, hierarchy = cv2.findContours(mask_emp,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        if len(contours):
            cv2.drawContours(img_mask[i], contours, -1, (1,0,0), 1)
    '''
    fig, ax = plt.subplots(nrows=1, ncols=2)

    ax[0].volume = volume
    ax[0].index = volume.shape[0] - 1  # // 2
    ax[0].imshow(volume[ax[0].index])
    ax[1].volume = reference
    ax[1].index = reference.shape[0] - 1  # // 2
    ax[1].imshow(reference[ax[1].index])
    fig.canvas.mpl_connect('key_press_event', process_key)


def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)


def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    ab = fig.axes[1]
    if event.key == 'j':
        previous_slice(ax)
        previous_slice(ab)
    elif event.key == 'k':
        next_slice(ax)
        next_slice(ab)
    fig.canvas.draw()


def previous_slice(ax):
    volume = ax.volume
    # print('previous slice',ax.index - 1)
    ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
    ax.images[0].set_array(volume[ax.index])


def next_slice(ax):
    volume = ax.volume
    # print('next slice',ax.index + 1)
    ax.index = (ax.index + 1) % volume.shape[0]
    ax.images[0].set_array(volume[ax.index])


def SortDicomFiles(files):
    files_lst = {}
    for filename in files:
        mri = sitk.ReadImage(filename)
        files_lst[int(mri.GetMetaData("0020|0013"))] = filename
    files_sorted = []
    for key in sorted(files_lst):
        files_sorted.append(files_lst[key])
    return files_sorted


def csv_to_h5(csv_file, h5):
    df = pd.read_csv(csv_file)  # csv.reader(f, delimiter=',')
    for _, row in df.iterrows():
        patient_id = row['ProxID']
        name = row['Name']
        finding_id = row['fid']
        pos = row['pos']
        world_matrix = row['WorldMatrix']
        ijk = row['ijk']
        top_level = row['TopLevel']
        spacing_between = row['SpacingBetweenSlices']
        voxel_spacing = row['VoxelSpacing']
        dim = row['Dim']
        dcm_descr = row['DCMSerDescr']
        # dcm_sernum = row['DCMSerNum']
        zone = row['zone']
        clin_sig = row['ClinSig']

        # train csv file contains redundant information. For example row 4 and 5 add no more information when
        # row 3 has already been seen.
        pathname = patient_id + '/' + dcm_descr + '/lesions/' + str(finding_id)
        if pathname in h5:
            print('Skipping duplicate {}'.format(pathname))
            continue
        else:
            dt_str = 'S255'
            group = h5.create_group(pathname)
            group.attrs.create('Name', name, dtype=dt_str)
            group.attrs.create('Pos', pos, dtype=dt_str)
            group.attrs.create('WorldMatrix', world_matrix, dtype=dt_str)
            group.attrs.create('ijk', ijk, dtype=dt_str)
            group.attrs.create('TopLevel', top_level, dtype=dt_str)
            group.attrs.create('SpacingBetween', spacing_between, dtype=dt_str)
            group.attrs.create('VoxelSpacing', voxel_spacing, dtype=dt_str)
            group.attrs.create('Dim', dim, dtype=dt_str)
            group.attrs.create('Zone', zone, dtype=dt_str)
            group.attrs.create('ClinSig', clin_sig, dtype=dt_str)


def dicom_to_h5(root_dir, h5):
    sub_dirs = [x[0] for x in os.walk(root_dir)]  # Gather all subdirectories in 'root_dir'
    ages = {}
    for directory in sub_dirs:
        file_list = glob.glob(directory + '/*.dcm')  # Look for .dcm files
        if not file_list:  # If we find a dir with a .dcm series, process it
            continue

        dcm_filename = file_list[0]  # Checking just one .dcm file is sufficient

        img = SimpleITK.ReadImage(dcm_filename)  # Read single .dcm file to obtain metadata

        # Extract some metadata that we want to keep
        patient_id = img.GetMetaData('0010|0020').strip()
        patient_age = img.GetMetaData('0010|1010').strip()
        if '0010|1030' in img.GetMetaDataKeys():
            patient_weight = img.GetMetaData('0010|1030').strip()
            patient_weight = int(patient_weight)
        else:
            patient_weight = -1

        if '0010|1020' in img.GetMetaDataKeys():
            patient_size = img.GetMetaData('0010|1020').strip()
            if len(patient_size.split('.')[0]) == 1:
                patient_size = int(round(float(patient_size) * 100))
            else:
                patient_size = int(round(float(patient_size)))
        else:
            patient_size = -1

        series_number = int(img.GetMetaData('0020|0011').strip())
        series_description = img.GetMetaData('0008|103e').strip()
        patient_age = int(patient_age[:3])
        # Add the age info in the dictionary for Ktrans
        ages[patient_id] = patient_age
        if patient_size > 0 and patient_weight > 0:
            patient_size_df = (patient_size / 100)
            patient_bmi = patient_weight / (patient_size_df * patient_size_df)
        else:
            patient_bmi = -1
        print(patient_id, series_number, series_description)

        data_path = patient_id + '/' + series_description
        print(data_path)
        print('Converting: {}'.format(data_path))

        # If we find a DICOM series that already exists, we check if the series number is higher. If so, remove
        # series that is already present and add this one.
        create = False

        if data_path in h5:
            if h5[data_path]['pixel_array'].attrs.get('SeriesNr') < series_number:
                del h5[data_path]
                print('New series has higher series number, so adding.')
                create = True
            else:
                print('New series has lower series number, so not adding.')
        else:
            create = True

        if create:
            dicom_matrix = load_dicom_series(directory)
            size_array = dicom_matrix.GetSize()
            origin_array = dicom_matrix.GetOrigin()
            spacing_array = dicom_matrix.GetSpacing()
            direction_array = dicom_matrix.GetDirection()
            ComponentsPerPixel_array = dicom_matrix.GetNumberOfComponentsPerPixel()
            width_array = dicom_matrix.GetWidth()
            height_array = dicom_matrix.GetHeight()
            depth_array = dicom_matrix.GetDepth()
            pixel_array = SimpleITK.GetArrayFromImage(dicom_matrix)
            group = h5.create_group(data_path)
            pixeldata = group.create_dataset('pixel_array', data=pixel_array)
            pixeldata.attrs.create('Age', int(patient_age))
            pixeldata.attrs.create('Weight', int(patient_weight))
            pixeldata.attrs.create('BMI', int(patient_bmi))
            pixeldata.attrs.create('Patient_Height', int(patient_size))
            pixeldata.attrs.create('SeriesNr', int(series_number))
            pixeldata.attrs.create('Size', size_array)
            pixeldata.attrs.create('Origin', origin_array)
            pixeldata.attrs.create('Spacing', spacing_array)
            pixeldata.attrs.create('Direction', direction_array)
            pixeldata.attrs.create('NumberOfComponentsPerPixel', ComponentsPerPixel_array)
            pixeldata.attrs.create('Width', width_array)
            pixeldata.attrs.create('Height', height_array)
            pixeldata.attrs.create('Depth', depth_array)
            dt_str = 'S255'
            pixeldata.attrs.create('StoreDirectory', directory, dtype=dt_str)
            pixeldata.attrs.create('StoreDirectoryType', 'Dir', dtype=dt_str)
    return ages


def Combine_two_csv_by_lesion_finding_id(csv_file_images, csv_finding_classification, csv_ktrans=None, filename=None):
    image_finding = pd.read_csv(csv_file_images)
    image_finding['ProxID_fid'] = image_finding['ProxID'] + '_' + (image_finding['fid']).astype(str)
    image_finding.set_index('ProxID_fid')
    # 1.Add the trans if applicable
    if csv_ktrans is not None:
        ktrans_finding = pd.read_csv(csv_ktrans)
        ktrans_finding['ProxID_fid'] = ktrans_finding['ProxID'] + '_' + (ktrans_finding['fid']).astype(str)
        ktrans_finding.set_index('ProxID_fid')
        ktrans_finding['Name'] = 'ktrans'
        ktrans_finding['DCMSerDescr'] = 'ktrans'
        ktrans_finding['DCMSerNum'] = 0
        result = image_finding.append(ktrans_finding)
        image_finding = result
    # 2. Merge between Table with ktrans and finding_classification
    finding_classification = pd.read_csv(csv_finding_classification)
    finding_classification['ProxID_fid'] = finding_classification['ProxID'] + '_' + (
    finding_classification['fid']).astype(str)
    finding_classification.set_index('ProxID_fid')
    result = pd.merge(image_finding, finding_classification, on='ProxID_fid', how='outer')
    result['ProxID'] = result['ProxID_x']
    result['pos'] = result['pos_x']
    result['fid'] = result['fid_x']
    result.set_index('ProxID')
    result = result.drop(columns=['ProxID_fid', 'ProxID_y', 'fid_y', 'pos_y', 'ProxID_x', 'fid_x', 'pos_x'])

    if filename is not None:
        result.to_csv(filename, sep=',', )
    return result


def ktrans_to_h5(root_dir, h5, ages):
    sub_dirs = [x[0] for x in os.walk(root_dir)]
    length_key = len('ProstateX-0022')
    for directory in sub_dirs:
        file_list = glob.glob(directory + '/*.mhd')

        if not file_list:  # If we find a dir with a .dcm or a .mhd series, process it
            continue
        series_filename = file_list[0]  # Checking just one .dcm file is sufficient
        img = sitk.ReadImage(series_filename)  # Read single .dcm file to obtain metadata

        filename = os.path.basename(series_filename)
        patient_id = filename[:length_key]
        print(patient_id)
        patient_age = ages[patient_id]
        series_number = 0
        series_description = 'ktrans'

        # Combine series description and series number to create a unique identifier for this DICOM series.
        # Should be unique for each patient. Only exception is ProstateX-0025, hence the try: except: approach.
        data_path = patient_id + '/' + series_description  # + '_' + series_number
        print('Converting: {}'.format(data_path))

        # If we find a DICOM series that already exists, we check if the series number is higher. If so, remove
        # series that is already present and add this one.
        create = False

        if data_path in h5:
            if h5[data_path]['pixel_array'].attrs.get('SeriesNr') < series_number:
                del h5[data_path]
                print('New series has higher series number, so adding.')
                create = True
            else:
                print('New series has lower series number, so not adding.')
        else:
            create = True

        if create:
            group = h5.create_group(data_path)
            data = sitk.GetArrayFromImage(img)
            size_array = img.GetSize()
            origin_array = img.GetOrigin()
            spacing_array = img.GetSpacing()
            direction_array = img.GetDirection()
            ComponentsPerPixel_array = img.GetNumberOfComponentsPerPixel()
            width_array = img.GetWidth()
            height_array = img.GetHeight()
            depth_array = img.GetDepth()
            pixeldata = group.create_dataset('pixel_array', data=data)
            pixeldata.attrs.create('Age', patient_age)
            pixeldata.attrs.create('SeriesNr', series_number)
            pixeldata.attrs.create('Size', size_array)
            pixeldata.attrs.create('Origin', origin_array)
            pixeldata.attrs.create('Spacing', spacing_array)
            pixeldata.attrs.create('Direction', direction_array)
            pixeldata.attrs.create('NumberOfComponentsPerPixel', ComponentsPerPixel_array)
            pixeldata.attrs.create('Width', width_array)
            pixeldata.attrs.create('Height', height_array)
            pixeldata.attrs.create('Depth', depth_array)
            dt_str = 'S255'
            pixeldata.attrs.create('StoreDirectory', series_filename, dtype=dt_str)
            pixeldata.attrs.create('StoreDirectoryType', 'file', dtype=dt_str)


# Reader for dicoms
def check_scan_metadata(scan_directory, expected_metadata):
    scan_files = glob.glob(scan_directory + '/*.dcm')

    # Read single .dcm file to obtain metadata
    img = SimpleITK.ReadImage(scan_files[0])

    for (key, value) in expected_metadata.items():
        # Check whether metadata key contains the right value
        if value not in img.GetMetaData(key):
            return False
    return True


def find_dicom_series_paths(root_dir, expected_metadata):
    series_paths = []

    case_dirs = [os.path.join(root_dir, x) for x in os.listdir(root_dir)]

    print("Found", len(case_dirs), "case(s).")

    for case_dir in case_dirs:
        # Expecting that each case folder (e.g. "ProstateX-0000") contains
        # one folder with a 'numbers' name (e.g. "1.3.6.1.4.1.14519.5.2.1.7311.5101.158323547117540061132729905711")
        subdirs = os.listdir(case_dir)
        if len(subdirs) != 1:
            print("Unexpectedly found multiple or no folders for case", case_dir)
            print("Skipping this case for now.")
            continue

        case_dir = os.path.join(case_dir, subdirs[0])
        scan_dirs = [os.path.join(case_dir, x) for x in os.listdir(case_dir)]

        scan_found = False

        for scan_dir in scan_dirs:
            if check_scan_metadata(scan_dir, expected_metadata):
                series_paths.append(scan_dir)

                if scan_found:
                    print("Found another scan with matching metadata.", scan_dir)
                scan_found = True

        if not scan_found:
            print("Could not find a scan for case", case_dir)

    return series_paths


def load_dicom_series(input_dir):
    """Reads an entire DICOM series of slices from 'input_dir' and returns its pixel data as an array."""
    reader = SimpleITK.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(input_dir)
    # Sort the dicom files
    dicom_names = SortDicomFiles(dicom_names)
    reader.SetFileNames(dicom_names)
    dicom_series = reader.Execute()
    return dicom_series  # SimpleITK.GetArrayFromImage(dicom_series)


def load_all_ADC_dicom_series(dir):
    metadata = {'0008|103e': 'ADC'}
    return load_all_dicom_series(dir, metadata)


def load_all_dicom_series(root_dir, expected_metadata):
    series_paths = find_dicom_series_paths(root_dir, expected_metadata)
    dicom_series = [load_dicom_series(path) for path in series_paths]
    return dicom_series


# %%
#####
# 1. Store all images and ktrans into h5 and store all finding location and their classification as well (Done)
# 2. Co-Register the images corresponding to Axial T2-weighted MRI prostate (Done)
# 3. Considere the coordination given by T2-weighted MRI prostate as reference (Done)
# 4. Extract the lesions by Croping the region by 48x48 (Based on Centroid) --> (Done)
#####
# 1.
########
#
#   Data preparation section include the whole information in one single h5 file.
#
#######
def PrepareData(to_store=None, folder_with_images=None, folder_with_ktrans=None, csv_for_finding_classification=None,
                csv_for_finding_images=None, csv_ktrans=None):
    # 1. Combine the classification information to the finding position..
    if csv_for_finding_classification is None or csv_for_finding_images is None or to_store is None or folder_with_ktrans is None:
        raise Exception('All parameters should be provided...')

    print('Warn: Any file with the given name will be removed...')
    # Generate a tmp file
    f = NamedTemporaryFile(delete=False)
    print('the temporary file is created: ', f.name)
    # Merge two csv files
    Combine_two_csv_by_lesion_finding_id(csv_for_finding_images, csv_for_finding_classification, csv_ktrans=csv_ktrans,
                                         filename=f.name)
    # Generate the data storage...
    # Remove the file name that shares the same file name....

    if os.path.exists(to_store):
        output = raw_input('A file was found. This file will be removed. Continue (Y/n)?')
        if output.lower() in ('y'):
            os.remove(to_store)
        else:
            raise Exception('Canceled by the user...')
    print(to_store)
    # Store all images in one file h5
    h5file = h5py.File(to_store, 'a')
    print('Proc: Storing all images into a single file....')
    print(folder_with_images)
    age = dicom_to_h5(folder_with_images, h5file)

    ktrans_to_h5(folder_with_ktrans, h5file, age)
    print('Proc: Adding textual information about the location and classification of ROI...')
    csv_to_h5(f.name, h5file)
    f.close()


# %%
####
# 2.
########
#
#   Co-Registration
#
########
#####
#
# General Function
#
#####
########
# 1. Get the images in each patients DONE
# 2. Generate a Group with Co-Registration DONE
# 3. Apply Elastix for Co-Registration  DONE
# Note: Ktrans has an inverse order of the slices. This should be fixed first.
# 4. Apply Rigid (Affine is not required) DONE
# 5. Store into CoRegistration path for later use (All stored...) DONE
########
def GenerateSitkImage(pixel_array, Normalize=True, ktrans_mode=False, verbose=False):
    pixel_array_ = np.array(pixel_array)
    Spacing = np.array(pixel_array.attrs.get('Spacing'))
    Origin = np.array(pixel_array.attrs.get('Origin'))
    Direction = np.array(pixel_array.attrs.get('Direction'))

    if ktrans_mode:
        pixel_array_ = (pixel_array_ - np.mean(pixel_array_)) / np.std(pixel_array_)
        pixel_array_d = pixel_array_.copy()
        pixel_array_[pixel_array_d > 3.] = 3.
        pixel_array_ = np.log2(np.log2(pixel_array_ + 1) + 1)
        '''
        This code fix the slice order of ktrans. MRI usually looks from Upside to downside
        '''
        simg = np.zeros(pixel_array_.shape)
        counter = 0
        for i in range(simg.shape[0] - 1, -1, -1):
            # print(i)
            simg[counter] = pixel_array_[i]
            counter = counter + 1
        pixel_array_ = simg

        if verbose:
            for i in range(pixel_array_.shape[0]):
                plt.imshow(pixel_array_[i])
                plt.show()

    img = sitk.GetImageFromArray(pixel_array_.astype(float))
    img.SetDirection(Direction)
    img.SetOrigin(Origin)
    img.SetSpacing(Spacing)
    if Normalize:
        img = sitk.AdaptiveHistogramEqualization(img)
        rescale = sitk.RescaleIntensityImageFilter()
        img = rescale.Execute(img)
    elif ktrans_mode:
        rescale = sitk.RescaleIntensityImageFilter()
        img = rescale.Execute(img)
    return img


def GetPatientList(h5_file):
    # Selecting all patients
    patients = [h5_file[patient_id] for patient_id in h5_file.keys()]
    print('number of patients:', len(patients))
    return patients


def WriteTheResult(data_path, co_registered, h5):
    print('CoRegistration: {}'.format(data_path))
    if data_path in h5:
        del h5[data_path]
    group = h5.create_group(data_path)
    data = sitk.GetArrayFromImage(co_registered)
    size_array = co_registered.GetSize()
    origin_array = co_registered.GetOrigin()
    spacing_array = co_registered.GetSpacing()
    direction_array = co_registered.GetDirection()
    ComponentsPerPixel_array = co_registered.GetNumberOfComponentsPerPixel()
    width_array = co_registered.GetWidth()
    height_array = co_registered.GetHeight()
    depth_array = co_registered.GetDepth()

    pixeldata = group.create_dataset('pixel_array', data=data)
    pixeldata.attrs.create('Size', size_array)
    pixeldata.attrs.create('Origin', origin_array)
    pixeldata.attrs.create('Spacing', spacing_array)
    pixeldata.attrs.create('Direction', direction_array)
    pixeldata.attrs.create('NumberOfComponentsPerPixel', ComponentsPerPixel_array)
    pixeldata.attrs.create('Width', width_array)
    pixeldata.attrs.create('Height', height_array)
    pixeldata.attrs.create('Depth', depth_array)


def command_iteration(method):
    print("{0:3} = {1:10.5f} : {2}".format(method.GetOptimizerIteration(),
                                           method.GetMetricValue(),
                                           method.GetOptimizerPosition()))


def KtranCoRegistration(ktrans, mri, verbose=False):
    # read the images
    fixed_image = sitk.Cast(mri, sitk.sitkFloat32)
    moving_image = sitk.Cast(ktrans, sitk.sitkFloat32)
    transform = sitk.CenteredTransformInitializer(fixed_image,
                                                  moving_image,
                                                  sitk.AffineTransform(fixed_image.GetDimension()),
                                                  # Euler3DTransform(),
                                                  sitk.CenteredTransformInitializerFilter.GEOMETRY)

    # multi-resolution rigid registration using Mutual Information
    registration_method = sitk.ImageRegistrationMethod()
    # registration_method.SetMetricAsMeanSquares()
    registration_method.SetMetricAsJointHistogramMutualInformation()
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.05)
    registration_method.SetInterpolator(sitk.sitkLinear)

    registration_method.SetOptimizerAsGradientDescentLineSearch(learningRate=1,
                                                                numberOfIterations=200,
                                                                convergenceMinimumValue=1e-4,
                                                                convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromJacobian()
    # registration_method.SetOptimizerScalesFromPhysicalShift
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration_method.SetInitialTransform(transform)
    final_transform = registration_method.Execute(fixed_image, moving_image)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(final_transform)

    out = resampler.Execute(moving_image)

    if verbose:
        simg2 = sitk.GetArrayViewFromImage(out)
        simg1 = sitk.GetArrayFromImage(mri)
        print(simg1.shape)
        print(simg2.shape)
        sitk.WriteImage(out, 'test_ktran.mhd', True)
        sitk.WriteImage(mri, 'test_mri.mhd', True)
        multi_slice_viewer(simg2, simg1)
        plt.show()
    return out


# This is a root function of the co-registration segment.
def ApplyCoRegistration(index, pts_list, h5,
                        considered_img=[
                            ['ktrans', 't2_tse_tra'],
                            ['DYNDIST', 't2_tse_tra'],
                            ['BVAL', 't2_tse_tra'],
                            ['ADC', 't2_tse_tra'],
                            ['PD', 't2_tse_tra']]
                        , data_storage_identifier='pixel_array', verbose=False):
    '''
    index : the index of the patient
    pts_list : the mri data from h5 with patient proxy accesss
    h5 : the h5 object that has the access to h5 file to store the result
    considered_img: Define the moving and fixed images e.g.
                            [['DYNDIST','t2_tse_tra'],
                            ['BVAL','t2_tse_tra']]]
    data_storage_identifier : the access name for h5
    verbose : Show the image results
    return : None
    '''
    patient_id = pts_list[index].name
    # Call the requested images to co-register
    for considered_mri_p, reference in considered_img:
        found = False

        # Search for the required images
        for name in pts_list[index].keys():
            if considered_mri_p in ['ADC', 'BVAL']:
                key_to_look = '_' + considered_mri_p
            else:
                key_to_look = considered_mri_p
            if key_to_look in name:
                if considered_mri_p == 'DYNDIST' and ('BVAL' in name or 'ADC' in name or 'PD' in name):
                    continue
                considered_mri_p_ = name
                found = True
                break

        if found == True:
            data_path = patient_id + '/CoRegistration/' + considered_mri_p
            if considered_mri_p == 'ktrans':
                # ktrans
                reference_ = None
                for key in pts_list[index].keys():
                    if reference in key:
                        reference_ = key
                        break
                if reference_ is None:
                    print('No Reference was found...Skipping...')
                    continue
                mri = GenerateSitkImage(pts_list[index][reference_][data_storage_identifier])

                if verbose:
                    d = sitk.GetArrayFromImage(mri)
                    multi_slice_viewer(d, d)
                    plt.show()
                ktrans_mri = GenerateSitkImage(pts_list[index]['ktrans'][data_storage_identifier], ktrans_mode=True,
                                               Normalize=False, verbose=verbose)

                if verbose:
                    d = sitk.GetArrayFromImage(ktrans_mri)
                    for i in range(d.shape[0]):
                        plt.imshow(d[i])
                        plt.show()
                    multi_slice_viewer(d, d)
                    plt.show()
                co_registered = KtranCoRegistration(ktrans_mri, mri)
                # Store Ktrans
                data_path = patient_id + '/CoRegistration/ktrans'
                WriteTheResult(data_path, co_registered, h5)
            else:
                reference_ = None
                for key in pts_list[index].keys():
                    if reference in key:
                        reference_ = key
                        break
                if reference_ is None:
                    print('No Reference was found...Skipping...')
                    continue

                img_fixed = GenerateSitkImage(pts_list[index][reference_][data_storage_identifier])

                if verbose:
                    d = sitk.GetArrayFromImage(img_fixed)
                    multi_slice_viewer(d, d)
                    plt.show()
                img_move = GenerateSitkImage(pts_list[index][considered_mri_p_][data_storage_identifier])
                if verbose:
                    d = sitk.GetArrayFromImage(img_move)
                    multi_slice_viewer(d, d)
                    plt.show()
                co_registered = CoRegistration(img_fixed, img_move)
                WriteTheResult(data_path, co_registered, h5)

            print('Done: CoRegistration: {}'.format(data_path))


def CoRegistration(fixedImage, movingImage, verbose=False):
    parameterMapVector = sitk.VectorOfParameterMap()
    parameterMapVector.append(sitk.GetDefaultParameterMap("rigid"))
    # parameterMapVector.append(sitk.GetDefaultParameterMap("affine"))
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(fixedImage)
    elastixImageFilter.SetMovingImage(movingImage)
    elastixImageFilter.LogToConsoleOff()
    elastixImageFilter.SetParameterMap(parameterMapVector)
    elastixImageFilter.Execute()
    resultImage = elastixImageFilter.GetResultImage()
    if verbose:
        simg2 = sitk.GetArrayFromImage(resultImage)
        simg1 = sitk.GetArrayFromImage(fixedImage)
        multi_slice_viewer(simg2, simg1)
        plt.show()
    return resultImage


# Use this function to run the co-registration
def GenerateCoRegistration(filename, debug_mode=False):
    print('Proc: Apply Co-Registration...')
    with h5py.File(filename, 'a') as h5_file:
        # Generate a list with all patients
        patients = [h5_file[patient_id] for patient_id in h5_file.keys()]
        number_of_patients = len(patients)
        counter = 0
        for index in range(number_of_patients):
            ApplyCoRegistration(index, patients, h5_file)
            if debug_mode:
                if counter == 2:
                    break
                counter = counter + 1
    print('Proc: Done Applying Co-Registration...')


# %%
# 3.
#############
#
#   Extract the lesions from h5 dataset Co-Registered images
#
#############
####
#
# General Function for Lesion Extraction
#
####
# %%
# %%
# %%
def dicom_series_query(h5_file, query_words, include, CoRegistration=None):
    """Returns a list of HDF5 groups of DICOM series that match words in query_words."""
    include_ = include
    if include_ is None:
        include_ = h5_file.keys()
    elif len(include_) == 0:
        include_ = h5_file.keys()

    if CoRegistration is not None:
        query_result = [
            h5_file[patient_id][dcm_series]  # We want patients with DICOM series such that:
            for patient_id in include_  # For all patients
            for dcm_series in h5_file[patient_id][CoRegistration].keys()  # For all DICOM series
            for word in query_words  # For every word in query words
            if word in dcm_series  # The word is present in DICOM series name
        ]
    else:
        query_result = [
            h5_file[patient_id][dcm_series]  # We want patients with DICOM series such that:
            for patient_id in include_  # For all patients
            for dcm_series in h5_file[patient_id].keys()  # For all DICOM series
            for word in query_words  # For every word in query words
            if word in dcm_series  # The word is present in DICOM series name
        ]
    return query_result


def get_lesion_info(h5_file, query_words, include=[], CoRegistration=None):
    query = dicom_series_query(h5_file, [query_words], include, CoRegistration=CoRegistration)

    # list of attributes to include in the lesion info
    include_attrs = ['ijk', 'VoxelSpacing', 'Zone', 'ClinSig']

    lesions_info = []
    for h5_group in query:
        if 'pixel_array' not in h5_group or 'lesions' not in h5_group:
            print('Warning in {}: No pixel array or lesions found for {}. Skipping...'
                  .format(get_lesion_info, h5_group))
            continue

        pixel_array = h5_group['pixel_array'][:]  # The actual DICOM pixel data
        patient_age = h5_group['pixel_array'].attrs.get('Age')

        lesion_info = []
        for finding_id in h5_group['lesions'].keys():
            lesion_dict = {
                'name': h5_group.name,
                'patient_id': filename_to_patient_id(h5_group.name)
            }
            for attr in include_attrs:
                # Per lesion finding, gather the attributes necessary for actual lesion extraction from DICOM image
                lesion_dict[attr] = h5_group['lesions'][finding_id].attrs.get(attr)
            lesion_dict['fid'] = finding_id
            lesion_dict['Age'] = patient_age
            lesion_info.append(lesion_dict)

        lesions_info.append([lesion_info, pixel_array])

    return lesions_info


def filename_to_patient_id(name):
    return name[11:15]


# %%
########
# 1. Extract the lesion information from t2_tse_tra
# 2. Generate the clipped area from Co_registered section
# (Apply 6 Channels for each lesion)
# 20 mm Window size
# #According to an article, the average length of the tumor is 13.4 mm on 526 pts.
# Tumor Length in Prostate Cancer Robin T. Vollmer, MD;
# Am J Clin Pathol 2008;130:77-82 77 77 DOI: 10.1309/PJNRHT63TP6FVC8B
########
def GenerateImageWithMultiChannels(img,
                                   channels,
                                   ijk_par,
                                   voxel_spacing_par,
                                   window_size_for_ML=(32, 32),
                                   Window_size_mm=(16, 16),
                                   Normalize=True,
                                   verbose=False):
    # Extract voxel spacing and ijk coordination
    VoxelSpacing = voxel_spacing_par.split(b',')
    i, j, k = ijk_par.split(b' ')
    i = int(i)
    j = int(j)
    k = int(k)

    # Nbr of channels
    channels_nbr = len(channels)

    # Create 6 channels image
    img_xv = np.zeros((window_size_for_ML[0], window_size_for_ML[1], channels_nbr))
    for index, channel in enumerate(channels):
        print('index', 'channel', index, channel)
        if channel not in img:
            img_xv[:, :, index] = np.zeros(window_size_for_ML)
        else:
            x = img[channel]
            if k >= x.shape[0]:
                k = x.shape[0] - 1
            x = x[k]
            if Normalize:
                x = exposure.equalize_adapthist(x / np.max(x), clip_limit=0.05, nbins=1000)
            # TODO Run 3D Region extraction
            window_size = int(round(Window_size_mm[0] / float(VoxelSpacing[0]))), int(
                round(Window_size_mm[1] / float(VoxelSpacing[1])))
            window_size_half_i, window_size_half_j = window_size[0] // 2, window_size[1] // 2
            start_x, end_x = (i - window_size_half_i), (i + window_size_half_i)
            start_y, end_y = (j - window_size_half_j), (j + window_size_half_j)
            if verbose:
                fig, ax = plt.subplots(1)
                # Display the image
                ax.imshow(x)
                # Create a Rectangle patch
                rect = patches.Rectangle((start_x, start_y), window_size[0], window_size[1], linewidth=1, edgecolor='r',
                                         facecolor='none')

                # Add the patch to the Axes
                ax.add_patch(rect)
                plt.show()

            clipped_area = x[start_y:end_y, start_x: end_x]
            # Convert to the standard window after getting 16 mm x 16 mm window
            if window_size_for_ML is not window_size:
                crop_arrea_ = Image.fromarray(clipped_area)
                crop_arrea_ = crop_arrea_.resize(window_size_for_ML)
                clipped_area = np.array(crop_arrea_)
            img_xv[:, :, index] = clipped_area.copy()
    return img_xv


def GenerateLesions(h5_file, reference='t2_tse_tra',
                    stored_in='CoRegistration',
                    from_mri=['DYNDIST', 'BVAL', 'ADC', 'PD', 'ktrans'],
                    store_directory='./',
                    img_stored='pixel_array',
                    Window_size_for_ML=(48, 48),
                    Window_size_mm=(20, 20),
                    verbose=False,
                    test_mode=False,
                    prefix='test',
                    select_cases=[]
                    ):
    '''
    '''
    counter = 0
    if verbose:
        print('select_cases', select_cases)
    lesions_info = get_lesion_info(h5_file, reference, select_cases, CoRegistration=None)
    plt_x = []
    plt_y = []
    if verbose:
        print('lesions_info', lesions_info)
    # Generate the images from the lesions with the given window size
    for lesion_info, pixel_array in lesions_info:
        print('Number of lesions:', len(lesion_info))
        # Run to each lesion
        for data in lesion_info:
            name = data['name'].split('/')[1]
            print(name, data['ijk'], data['VoxelSpacing'], data['ClinSig'], data['Zone'])
            # shape_0 = len(from_mri) + 1
            img_x = {}
            img_x[reference] = pixel_array

            for img_type in from_mri:
                data_path = '/' + name + '/' + stored_in + '/' + img_type
                if data_path in h5_file:
                    data_path = '/' + name
                    group = h5_file[data_path][stored_in][img_type]
                    img = group[img_stored][:]
                    img_x[img_type] = img.copy()

            # Function
            chn = from_mri + [reference]
            print(chn)
            img = GenerateImageWithMultiChannels(img_x,
                                                 chn,
                                                 data['ijk'],
                                                 data['VoxelSpacing'],
                                                 Window_size_for_ML,
                                                 Window_size_mm,
                                                 verbose=verbose)
            plt_x.append(img)
            plt_y.append(data['ClinSig'])
            if verbose:
                print(img.shape)
                for i in range(len(chn)):
                    print(chn[i])
                    plt.show()
                    plt.imshow(img[:, :, i])
                    plt.show()
        if test_mode:
            counter = counter + 1
            if counter == 3:
                print('Stopped....')
                stop()
    # Save the results
    print('Proc: Saving the lesions in numpy matrix format...')
    filename = store_directory + '%s_x_dataset.npy' % (prefix)
    print(np.array(plt_x).shape)
    print(np.array(plt_y).shape)
    np.save(filename, np.array(plt_x))
    filename = store_directory + '%s_y_dataset.npy' % (prefix)
    np.save(filename, np.array(plt_y))
    print('Done: Saving the lesions... Ready for DL or ML...')


######
# END LESION EXTRACTION SECTION
######
# %%
######
#
#   Splitting the data.
#
######
def SplitData(h5_file,
              store_directory='./',
              reference='t2_tse_tra',
              stored_in='CoRegistration',
              from_mri=['DYNDIST', 'BVAL', 'ADC', 'PD', 'ktrans'],
              img_stored='pixel_array',
              Window_size_for_ML=(48, 48),
              Window_size_mm=(20, 20),
              verbose=False,
              test_mode=False,
              train_portion=0.9):
    import random
    ptl_list = list(h5_file.keys())
    select_nr = int(round(len(ptl_list) * train_portion))
    print('#' * 30)
    print('Training set:')
    print('n=', select_nr)
    # Random selection...
    selected_plts = random.sample(ptl_list, k=select_nr)
    print(selected_plts)
    print('#' * 30)
    print('Valid set:')
    print('n=', len(ptl_list) - select_nr)
    remaining = difference_btn_twolist(ptl_list, selected_plts)
    print(remaining)
    print('#' * 30)

    # Generating training and test sets
    print('Proc: Generating the data sets for training set...')
    GenerateLesions(h5_file, prefix='train',
                    store_directory=store_directory,
                    reference=reference,
                    from_mri=from_mri,
                    img_stored=img_stored,
                    Window_size_for_ML=Window_size_for_ML,
                    Window_size_mm=Window_size_mm,
                    verbose=verbose,
                    test_mode=test_mode,
                    select_cases=selected_plts)
    print('Proc: Generating the data sets for test set...')
    GenerateLesions(h5_file, prefix='test',
                    store_directory=store_directory,
                    reference=reference,
                    from_mri=from_mri,
                    img_stored=img_stored,
                    Window_size_for_ML=Window_size_for_ML,
                    Window_size_mm=Window_size_mm,
                    verbose=verbose,
                    test_mode=test_mode,
                    select_cases=remaining)
    print('Proc: Done')  # %%


####
def Run(args):
    if args.register is not True and args.lesion is not True and args.split is not True:
        print('Proc: Storing the whole data in one single file...')
        PrepareData(to_store=args.h5_path,
                    folder_with_images=args.mpmri,
                    folder_with_ktrans=args.ktrans,
                    csv_for_finding_classification=args.csvfinding,
                    csv_for_finding_images=args.csvimg,
                    csv_ktrans=args.csvktrans)
    if args.lesion is not True and args.split is not True:
        GenerateCoRegistration(args.h5_path, debug_mode=args.debug)
    print('Proc: Splitting the data into train and test sets with the given ratio...')
    with h5py.File(args.h5_path, 'a') as f:
        SplitData(f, store_directory=args.store, test_mode=args.debug, verbose=args.verbose)
    print('Proc: Done splitting the data...')
    print('Proc: Done')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", help="the path where the train and test data will be stored",
                        default="./")
    parser.add_argument('-r', "--register", help="Start with co-registration...",
                        action="store_true")
    parser.add_argument('-l', "--lesion", help="Start with lesion generation...",
                        action="store_true")
    parser.add_argument('-s', "--split", help="Start with splitting...",
                        action="store_true")
    parser.add_argument("--h5_path", help="the path where the h5 data will be stored",
                        default="./database.h5")
    parser.add_argument("--mpmri", help="the path where the mpMRI images are stored",
                        default="./train/PROSTATEx/")
    parser.add_argument("--ktrans", help="the path where the ktrans images are stored",
                        default="./ProstateXKtrains-train-fixed")
    parser.add_argument("--csvfinding", help="load the csv file with the lesion information",
                        default='./ProstateX-TrainingLesionInformationv2/ProstateX-Findings-Train.csv')
    parser.add_argument("--csvktrans", help="load the csv file for ktrans images",
                        default='./ProstateX-TrainingLesionInformationv2/ProstateX-Images-KTrans-Train.csv')
    parser.add_argument("--csvimg", help="load the csv file for mri images",
                        default='./ProstateX-TrainingLesionInformationv2/ProstateX-Images-Train.csv')
    parser.add_argument("--ratio", help="ratio of train set",
                        default=0.9, type=float)
    parser.add_argument("--verbose", help="increase output verbosity",
                        action="store_true")
    parser.add_argument('--debug', action='store_true',
                        help="Run debug mode for Co-Registration")
    args = parser.parse_args()
    print(args)
    Run(args)

# %%
