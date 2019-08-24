#%%
from augment import augment
import numpy as np
import matplotlib.pyplot as plt
#%%
db =np.load('./img_valid_data_3d_t2_tse_tra.npy')

#%%
image = db[1]
image = image.reshape((16,144,144))
transformation = augment.create_identity_transformation(image.shape)
# jitter in 3D
transformation += augment.create_elastic_transformation(
        image.shape,
        control_point_spacing=95,
        jitter_sigma=3)
# apply transformation
image = augment.apply_transformation(image, transformation)
for i in range(image.shape[0]):
    plt.imshow(image[i])
    plt.show()
print(image.shape)