#%matplotlib inline
import matplotlib.pylab as plt
import os
import glob
import numpy as np
import math
from data_augmentation_h5py import *
import cnn_utils
import ipt_utils
import natsort
import time
import random
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

'''

# -----------------------------------------------------
# IF COMBINING HEALTHY AND UNHEALTHY BRAIN 
# DATASETS INTO ONE DIRECTORY WITHOUT CREATING
# A VALIDATION DIRECTORY
# -----------------------------------------------------
imgs_list = glob.glob("c:/users/fungs/bmen501/Data/Training_Orig/*.nii.gz")
masks_list = glob.glob("c:/users/fungs/bmen501/Data/Training_STAPLE/*.nii.gz")

# shuffle the images and masks
combined_list = list(zip(imgs_list, masks_list))
random.shuffle(combined_list)
imgs_list_shuf, masks_list_shuf = zip(*combined_list)

# percentage of data used for validation
val_perc = 0.25

# number of brains in validation set
val = math.ceil(val_perc*len(imgs_list))

# train and validation split
img_train = imgs_list_shuf[:-val]
mask_train = masks_list_shuf[:-val]
img_val = imgs_list_shuf[-val:]
mask_val = masks_list_shuf[-val:]

'''

# -----------------------------------------------------
# IF SPLITTING BOTH HEALTHY AND UNHEALTHY BRAINS
# INTO TRAINING AND VALIDATION DIRECTORIES
# -----------------------------------------------------
# loading all the filenames
# h5py images
img_train = glob.glob("c:/users/fungs/bmen501/Data/Train-hdf5/Image/*.h5")
mask_train = glob.glob("c:/users/fungs/bmen501/Data/Train-hdf5/Mask/*.h5")
img_val = glob.glob("c:/users/fungs/bmen501/Data/Val-hdf5/Image/*.h5")
mask_val = glob.glob("c:/users/fungs/bmen501/Data/Val-hdf5/Mask/*.h5")

# nifti images
# img_train = glob.glob("c:/users/fungs/bmen501/Data/Train/Image/*.nii.gz")
# mask_train = glob.glob("c:/users/fungs/bmen501/Data/Train/Mask/*.nii.gz")
# img_val = glob.glob("c:/users/fungs/bmen501/Data/Val/Image/*.nii.gz")
# mask_val = glob.glob("c:/users/fungs/bmen501/Data/Val/Mask/*.nii.gz")

# -----------------------------------------------------

# sort the sets
img_train = natsort.natsorted(img_train)
mask_train = natsort.natsorted(mask_train)
img_val = natsort.natsorted(img_val)
mask_val = natsort.natsorted(mask_val)

print(img_train[268])
print(mask_train[268])

# training set data generator
train_dg = DataGeneratorH5PY2D(img_train, mask_train, (96,96), (30,30), 36,
                 rot = 30, shift_x = 15, shift_y = 10,
                 scale = 0.6, shear = 15, flip_x = True, flip_y = True, shuffle = True)

# validation set data generator
val_dg = DataGeneratorH5PY2D(img_val, mask_val, (96,96), (30,30), 9,
                  rot = 30, shift_x = 15, shift_y = 10,
                  scale = 0.6, shear = 15, flip_x = True, flip_y = True, shuffle = True)

# t  =time.time()
# X,Y = train_dg.__getitem__(10)
# t = time.time() -t
# print(t)

# string used to save models after each epoch
model_name = "./Network/unet_data_best_15pat.hdf5"

# early stopping callback to shut down training after
# 10 epochs with no improvement
earlyStopping = EarlyStopping(monitor='val_dice_coef', patience=15, verbose=0, mode='max')

# checkpoint callback to save model  along the epochs
checkpoint = ModelCheckpoint(model_name, mode = 'max', monitor='val_dice_coef',verbose=0,
                             save_best_only=False, save_weights_only = True)
    
# create model
model = cnn_utils.get_unet_mod(patch_size = (96,96), nchannels = 1, learning_rate = 1e-4,
                                learning_decay = 5e-9)

hist = model.fit(train_dg, epochs=500, verbose = 1, validation_data = (val_dg),
                    callbacks = [checkpoint, earlyStopping], max_queue_size = 20, workers = 12)

# saving training history
np.save("./network/dice_15pat.npy",np.array(hist.history['dice_coef']))
np.save("./network/val_dice_15pat.npy",np.array(hist.history['val_dice_coef']))

# plotting the dice coefficient
dice = np.array(hist.history['dice_coef'])
val_dice = np.array(hist.history['val_dice_coef'])
loss = np.array(hist.history['loss'])
val_loss = np.array(hist.history['val_loss'])
print (val_loss.argmin())
plt.figure(figsize = (4,4/1.5),facecolor = "w",dpi = 450)
plt.plot(dice,'x-', markersize = 4,label = "Train")
plt.plot(val_dice,'+-',markersize = 4,label = "Validation")
plt.legend()
plt.grid()
plt.xlim(0,500)
plt.ylim(0.90,1.0)
plt.xlabel("Epoch")
plt.ylabel("Dice coefficient")
plt.show()

