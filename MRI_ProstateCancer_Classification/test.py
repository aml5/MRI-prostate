#%%
import keras
import numpy as np
import matplotlib.pyplot as plt
#%%
X = np.load('./CoRegistration/img_valid_data_3d_t2_tse_tra.npy')
#%%
print(X.shape)
for i in range(16):
    plt.imshow(X[0,i,:,:,0])
    plt.show()
#%%
print(X.shape)
plt.imshow(X[4500,3,:,:,0])
plt.show()