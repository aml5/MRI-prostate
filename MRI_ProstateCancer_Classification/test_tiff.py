#%%
from PIL import Image
from PIL.TiffTags import TAGS
import imageio
import numpy as np
import tifffile
from openslide import OpenSlide
#%%
with tifffile.TiffFile('Y:\Histo_Images\McNealx40\Prostate_Cancer_TIFF_files\OE1__20190125_114509.tiff') as tif:
    data =tif.pages
#%%
print((data))
#%%
plt.imshow(data[4].asarray())
#%%
import matplotlib.pyplot as plt
from imagecodecs import jpegsof3_decode
#%%
print(data[10].asarray())
#%%
img = imageio.mimread('Y:\Histo_Images\McNealx40\Prostate_Cancer_TIFF_files\OE1__20190125_114509.tiff')
#%%
with Image.open('Y:\Histo_Images\McNealx40\Prostate_Cancer_TIFF_files\OE1__20190125_114509.tiff') as img:
    meta_dict = {TAGS[key] : img.tag[key] for key in img.tag.iterkeys()}