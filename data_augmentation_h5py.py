import numpy as np
import nibabel as nib
import matplotlib.pylab as plt
from tensorflow import keras
from skimage.transform import AffineTransform, SimilarityTransform, warp
from tensorflow.keras import backend as K
import h5py

class DataGeneratorH5PY2D(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, imgs_list, masks_list,  patch_size, crop, batch_size,
                 rot = None, shift_x = None, shift_y = None,
                 scale = None, shear = None, flip_x = None, flip_y = None, augment_flag = False, shuffle = True):


        self.imgs_list = imgs_list
        self.masks_list = masks_list
        self.patch_size = patch_size
        self.crop = crop
        self.batch_size = batch_size
        self.rot = rot
        self.shift_x = shift_x
        self.shift_y = shift_y
        self.scale = scale
        self.shear = shear
        self.flip_x = flip_x
        self.flip_y = flip_y
        self.shuffle = shuffle
        self.augment_flag = augment_flag
        self.nsamples,self.samples_cumsum = self.__computer_number_of_samples__()
        # self.file_ids, self.slice_ids = self.__compute_file_id_and_slice_id__()
        self.on_epoch_end()

    def __computer_number_of_samples__(self):
        """Counts the number of slices in the dataset and provides a cumsum array for mapping indexes to slices i
        specific files"""

        slice_count = []
        for ii in self.imgs_list:
            with h5py.File(ii, 'r') as f:
                aux = f['data'].shape[0]
                slice_count.append(aux - (self.crop[1] + self.crop[0]))

        slice_count = np.array(slice_count)
        return slice_count.sum(),slice_count.cumsum()

    def __compute_file_id_and_slice_id__(self):
        
        indexes = np.arange(self.nsamples)

        file_ids = [0]*self.samples_cumsum[0]
        for i in range(1,self.samples_cumsum.size):
            file_ids += [i]*(self.samples_cumsum[i] - self.samples_cumsum[i-1])
        file_ids = np.array(file_ids, dtype=int)

        file_slices = np.zeros(self.nsamples, dtype=int)
        file_slices[:self.samples_cumsum[0]] = np.arange(self.samples_cumsum[0])
        file_slices[self.samples_cumsum[0]:] = (self.crop[0] + indexes[self.samples_cumsum[0]:] 
            - self.samples_cumsum[file_ids[indexes[self.samples_cumsum[0]:]]-1])
        
        return file_ids, file_slices

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.nsamples / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'

        # Generate indexes of the batch
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        X, Y = self.__data_generation(batch_indexes)

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.nsamples)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_indexes):
        'Generates data containing batch_size samples'

        # Initialization
        X = np.empty((self.batch_size, self.patch_size[0], self.patch_size[1], 1))
        Y = np.empty((self.batch_size, self.patch_size[0], self.patch_size[1], 1))
       
        # Generate data
        for ii in range(batch_indexes.shape[0]):
            try:
                file_id = np.where(batch_indexes[ii] >= self.samples_cumsum)[0][-1]
            except IndexError:
                file_id = 0
            if file_id != 0 or batch_indexes[ii] > self.samples_cumsum[file_id]:
                file_slice = self.crop[0] + batch_indexes[ii] - self.samples_cumsum[file_id]
                file_id += 1
            else:
                file_slice = self.crop[0] + batch_indexes[ii]
            
            with h5py.File(self.imgs_list[file_id], 'r') as f:
                aux_img = f['data'][file_slice]
            with h5py.File(self.masks_list[file_id], 'r') as f:
                aux_mask = (f['data'][file_slice] > 0.5)
                
            # mask before aug
            if self.augment_flag:
                aux_img_aug, aux_mask_aug = self.augment(aux_img,aux_mask) 
                aux_img_crop, aux_mask_crop = self.extract_patch(aux_img_aug, aux_mask_aug)
            else:
                aux_img_crop, aux_mask_crop = self.extract_patch(aux_img, aux_mask)
            # print(self.imgs_list[file_id], self.masks_list[file_id])
            X[ii,:,:,0] = aux_img_crop
            Y[ii,:,:,0] = aux_mask_crop
        return X,Y


    def augment(self,img,mask):

        rotation = np.random.random_integers(0, self.rot)
        translation = (np.random.random_integers(-self.shift_x, self.shift_x),\
                           np.random.random_integers(-self.shift_y, self.shift_y))
        scale = np.random.random_sample()*self.scale + 0.7
        shear = np.random.random_integers(-self.shear, self.shear)
        flip_x = np.random.random_sample()
        flip_y = np.random.random_sample()

        if flip_x > 0.5:
            img = img[::-1,:]
            mask = mask[::-1, :]
        if flip_y > 0.5:
            img = img[:,::-1]
            mask = mask[:,::-1]

        tf_augment = AffineTransform(scale=scale, rotation=np.deg2rad(rotation), translation=translation,
                                         shear=np.deg2rad(shear))

        img_aug = warp(img, tf_augment, order=1, preserve_range=True, mode='symmetric')
        mask_aug = warp(mask, tf_augment, order=0, preserve_range=True, mode='symmetric')

        return img_aug, mask_aug

    def extract_patch(self, img, mask):
        crop_idx = [None]*2
        crop_idx[0] = np.random.randint(0, img.shape[0] - self.patch_size[0])
        crop_idx[1] = np.random.randint(0, img.shape[1] - self.patch_size[1])
        img_cropped =  img[crop_idx[0]:crop_idx[0] + self.patch_size[0],\
                              crop_idx[1]:crop_idx[1] + self.patch_size[1]]
        mask_cropped = mask[ crop_idx[0]:crop_idx[0] + self.patch_size[0], \
                          crop_idx[1]:crop_idx[1] + self.patch_size[1]]
        return img_cropped,mask_cropped