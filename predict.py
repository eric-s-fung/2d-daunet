import matplotlib.pylab as plt
import numpy as np
import nibabel as nib
import tensorflow as tf
import os
import glob
import sys
import time
import ipt_utils
import cnn_utils
import metrics_utils
from math import ceil

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

img_test = glob.glob("C:/users/fungs/bmen501/data/Val/Image/*.nii.gz")
mask_test = glob.glob("C:/users/fungs/bmen501/data/Val/Mask/*.nii.gz")
# img_test_tum = glob.glob("C:/users/fungs/bmen501/data/test/testing_tumper/*.nii.gz")
# mask_test_tum = glob.glob("C:/users/fungs/bmen501/data/test/testing_robex/*.nii.gz")

# im = img_test[8]
# ma = mask_test[8]

# print(im)
# print(ma)

model_path = "./network/unet_data_aug_best_15pat.hdf5"
model = cnn_utils.get_unet_mod(nchannels = 1)
model.load_weights(model_path)

dice = np.zeros ((len(img_test),1))

for ii in range(len(img_test)):
    orig_img = nib.load(img_test[ii])

    # ensure the image is a multiple of 16 for unet prediction 
    # shape of the original image
    a,b,c = orig_img.shape
    # calculate how many zeros to pad
    bb = 16 - b%16
    cc = 16 - c&16 if c%16 != 0 else 0
    # initialize array with all multiples of 16
    aux_img = np.zeros((a,b+bb,c+cc))

    # load image data
    temp_img = orig_img.get_fdata()
    # normalize image data
    temp_min = temp_img.min(axis = (1,2), keepdims = True)
    temp_max = temp_img.max(axis = (1,2), keepdims = True)  
    temp_img = (temp_img - temp_min)/(temp_max - temp_min)
    # handle any NaN values
    temp_img = np.nan_to_num(temp_img)
    # insert image data into zero-padded array
    aux_img[:,:b,:c] = temp_img

    # add channel to numpy array to fit model
    x, y, z = aux_img.shape
    img = np.zeros((x,y,z,1))
    img[:,:,:,0] = aux_img[:,:,:]

    # model prediction
    predict = model.predict(img, batch_size = 4)
    predict = predict[:,:,:,0]
    predict2 = np.zeros((a,b,c))

    # remove zero padding
    predict2[:,:,:] = predict[:x,:y-bb,:z-cc]
    predict2 = (predict2 >0.5).astype(np.uint8)

    # # show image
    # H,W,Z = aux_img.shape
    # plt.figure()
    # plt.subplot(131)
    # plt.imshow(aux_img[ceil(H/2),:,:], cmap = 'gray')
    # plt.imshow(predict2[ceil(H/2),:,:], cmap = 'cool',alpha = 0.2)
    # plt.axis("off")
    # plt.subplot(132)
    # plt.imshow(aux_img[:,ceil(W/2),:], cmap = 'gray')
    # plt.imshow(predict2[:,ceil(W/2),:], cmap = 'cool',alpha = 0.2)
    # plt.axis("off")
    # plt.subplot(133)
    # plt.imshow(aux_img[:,:,ceil(Z/2)], cmap = 'gray')
    # plt.imshow(predict2[:,:,ceil(Z/2)], cmap = 'cool',alpha = 0.2)
    # plt.axis("off")
    # plt.show()

    # compare to manual/ROBEX masks
    mask_manual = nib.load(mask_test[ii]).get_fdata()>0.5
    dice[ii,0] = metrics_utils.dice(mask_manual,predict2)  

print ("Average dice and standard deviation") 
print (dice.mean(axis = 0)) 
print (dice.std(axis = 0))
# print(im)
# print(ma)   