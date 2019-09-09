import sys
import numpy as np
import h5py

H5DATASET=['img','label','attr']
SHAPE = (16,144,144,1)
DTYPE = np.float16

class h5subset():
    def __init__(self, infilepath, outfilepath ):
        self._infile = infilepath
        self._outfile = outfilepath
        self._shape = SHAPE
        self._out_index = 0

        h5fin = h5py.File(self._infile, 'r')
        self._img = h5fin[H5DATASET[0]][()]
        self._labels = h5fin[H5DATASET[1]][()]
        try:
            self._attrs = h5fin[H5DATASET[2]][()]
        except:
            self._attrs = None

        dtype = DTYPE
        chunk_len = 1
        shape = self._shape


        with h5py.File(self._outfile, mode='x') as h5fout:
            self.img = h5fout.create_dataset(
                'img',
                shape=(0, ) + shape,
                maxshape=(None, ) + shape ,
                dtype=dtype,
                compression=None,
                chunks=(chunk_len, ) + shape)
            self.label = h5fout.create_dataset(
                'label',
                shape=(0, ),
                maxshape=(None, ),
                dtype=dtype,
                compression=None,
                chunks=(chunk_len, ))

            if self._attrs is not None:
                dt = h5py.special_dtype(vlen=str)
                self.attr = h5fout.create_dataset(
                    'attr',
                    shape=(0,),
                    maxshape=(None,),
                    dtype=dt,
                    compression=None,
                    chunks=(chunk_len,))

    def subset(self, starting, ending):
        for idx in range(starting, ending):
            img = self._img[idx]
            label = self._labels[idx]
            attr = self._attrs[idx] if self._attrs is not None else None
            self.append(img, label, attr)


    def append(self,  img_vle, label_vle, attr_vle):
        with h5py.File(self._outfile, mode='a') as h5fout:
            img = h5fout['img']
            img.resize((self._out_index + 1, ) + self._shape)
            img[self._out_index] = [img_vle.reshape(self._shape)]
            label = h5fout['label']
            label.resize((self._out_index + 1, ))
            label[self._out_index] = [label_vle]
            if attr_vle is not None:
                attr = h5fout['attr']
                attr.resize((self._out_index + 1,))
                attr[self._out_index] = [attr_vle]
            self._out_index += 1
            h5fout.flush()

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print('Usage: python h5copy.ph <h5 source file> <h5 target file> <starting index> <ending index>')
        exit()

    h5cp = h5subset(sys.argv[1], sys.argv[2])
    h5cp.subset(int(sys.argv[3]),int(sys.argv[4]))
