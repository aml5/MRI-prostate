import numpy as np
import h5py

class HDF5Store(object):
    """
    Simple class to append value to a hdf5 file on disc (usefull for building keras datasets)
    
    Params:
        datapath: filepath of h5 file
        dataset: dataset name within the file
        shape: dataset shape (not counting main/batch axis)
        dtype: numpy dtype
    
    Usage:
        hdf5_store = HDF5Store('/tmp/hdf5_store.h5','X', shape=(20,20,3))
        x = np.random.random(hdf5_store.shape)
        hdf5_store.append(x)
        hdf5_store.append(x)
        
    From https://gist.github.com/wassname/a0a75f133831eed1113d052c67cf8633
    """
    def __init__(self, datapath, shape, dtype=np.float32, compression=None, chunk_len=1, label_definition={'NORM':0,'iPCA':1,'sPCA':2}):
        self.datapath = datapath
        self.shape = shape
        self.i = 0
        self.label_definition = label_definition
        
        with h5py.File(self.datapath, mode='w') as h5f:
            self.img = h5f.create_dataset(
                'img',
                shape=(0, ) + self.shape,
                maxshape=(None, ) + self.shape ,
                dtype=dtype,
                compression=None,
                chunks=(chunk_len, ) + self.shape)
            self.label = h5f.create_dataset(
                'label',
                shape=(0, ),
                maxshape=(None, ),
                dtype=dtype,
                compression=None,
                chunks=(chunk_len, ))

            dt = h5py.special_dtype(vlen=str)
            self.attr = h5f.create_dataset(
                'attr',
                shape=(0,),
                maxshape=(None,),
                dtype=dt,
                compression=None,
                chunks=(chunk_len,))

    
    def append(self, label_vle, img_vle, attr_vle):
        with h5py.File(self.datapath, mode='a') as h5f:
            img = h5f['img']
            img.resize((self.i + 1, ) + self.shape)
            img[self.i] = [img_vle.reshape(self.shape)]
            label = h5f['label']
            label.resize((self.i + 1, ))
            label[self.i] = [self.label_definition[label_vle]]
            attr = h5f['attr']
            attr.resize((self.i + 1,))
            attr[self.i] = [attr_vle]
            self.i += 1
            h5f.flush()
            return len(img)