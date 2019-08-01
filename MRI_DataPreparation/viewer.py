import sys, os
sys.path.insert(0, 'MRI_Prostate_Segmentation')
import D3_utils as utils
import matplotlib.pyplot as plt
import SimpleITK as sitk

class multipatient:

    def __init__(self, parentpath, imagetype, filetype):
        self._parentpath = parentpath
        self._imagetype = imagetype
        self._filetype = filetype

class slices:

    def __init__(self, parentpath, patient, volname, filetype):
        self._patient = patient
        self._imgpath = f'{parentpath}/{patient}/{volname}-preprocessed.{filetype}'
        self._maskpath = f'{parentpath}/{patient}/{volname}-mask.{filetype}'

    def readFiles(self):
        self._img = sitk.GetArrayFromImage(sitk.ReadImage(self._imgpath))
        self._mask = sitk.GetArrayFromImage(sitk.ReadImage(self._maskpath))

    def viewSlices(self):
        utils.multi_slice_viewer_legacy(self._mask, self._img)
        plt.suptitle(self._patient)
        plt.gcf().canvas.set_window_title(self._patient)
        plt.show()

if __name__ == '__main__':
    display = slices(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    display.readFiles()
    display.viewSlices()