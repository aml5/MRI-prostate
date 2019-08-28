import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py

LABELS=['NORM','iPCA','sPCA']

class h5viewer():
    def __init__(self, filepath, h5dataset='img',h5labelset='label',h5attrset='attr'):
        self._filepath = filepath
        h5f = h5py.File(filepath, 'r')
        img5d = h5f[h5dataset][()]
        newshape = tuple(list(img5d.shape)[0:4])
        self._img4d = np.reshape(img5d, newshape) # reshape from (8,16,144,144,1) to (8,16,144,144)
        self._labels = h5f[h5labelset][()]
        self._attrs = h5f[h5attrset][()]
        self._volnum = 0

    def view(self):
        self._volnum = 0
        self.multivolume_view()

    def multivolume_view(self):
        self._remove_keymap_conflicts({'j', 'k', 'left', 'right', 'u', 'n', 'up', 'down'})

        img3d = self._img4d[self._volnum]
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.volume = img3d
        ax.index = 0
        ax.imshow(img3d[ax.index], cmap='gray') #, vmin=0.0, vmax=1.0)
        fig.suptitle(f'Volume {self._volnum}, Slice {ax.index},  Label {LABELS[int(self._labels[self._volnum])]}')
        fig.canvas.mpl_connect('key_press_event', self._process_key)
        plt.show()

    def stats(self, img3d):
        return (np.mean(img3d), np.std(img3d))

    def _remove_keymap_conflicts(self, new_keys_set):
        for prop in plt.rcParams:
            if prop.startswith('keymap.'):
                keys = plt.rcParams[prop]
                remove_list = set(keys) & new_keys_set
                for key in remove_list:
                    keys.remove(key)

    def _process_key(self, event):
        fig = event.canvas.figure
        for ax in fig.axes:
            if hasattr(ax, 'index'):
                if event.key in ['j', 'left']:
                    self._previous_slice(ax)
                elif event.key in ['k', 'right']:
                    self.__next_slice(ax)
                elif event.key in ['i', 'down']:
                    self.__previous_volume(ax)
                elif event.key in ['u', 'up']:
                    self.__next_volume(ax)
        fig.suptitle(f'Volume {self._volnum}, Slice {ax.index},  Label {LABELS[int(self._labels[self._volnum])]}')
        img = self._img4d[self._volnum]
        fig.gca().set_xlabel(f'Mean:{np.mean(img):.2f}, Std:{np.std(img):.2f}, Max:{np.max(img)}')
        fig.canvas.draw()

    def _previous_slice(self, ax):
        volume = ax.volume
        ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
        ax.images[0].set_array(volume[ax.index])

    def __next_slice(self, ax):
        volume = ax.volume
        ax.index = (ax.index + 1) % volume.shape[0]
        ax.images[0].set_array(volume[ax.index])

    def __previous_volume(self, ax):
        self._volnum = (self._volnum - 1) % len(self._img4d)
        volume = self._img4d[self._volnum]
        ax.volume = volume
        ax.images[0].set_array(volume[ax.index])

    def __next_volume(self, ax):
        self._volnum = (self._volnum + 1) % len(self._img4d)
        volume = self._img4d[self._volnum]
        ax.volume = volume
        ax.images[0].set_array(volume[ax.index])

if __name__ == '__main__':
    if len(sys.argv) != 2 :
        print('Usage: python h5viewer.py <h5 filepath> ')
    else:
        viewer = h5viewer(sys.argv[1],h5dataset='img',h5labelset='label',h5attrset='attr')
        viewer.view()
