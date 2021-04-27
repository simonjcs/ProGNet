"""
Author: Simon John Christoph Soerensen, MD, et al. (simonjcs@stanford.edu)
Link to paper: https://www.auajournals.org/doi/10.1097/JU.0000000000001783
Created: February 16, 2021
Latest Version: April 10, 2021
Use this code to run the ProGNet prostate clinical segmentation pipeline
Specify input (T2-DICOM folders), output (t2-SEG-DICOM folders), Prognet_t2.h5 (deep learning weights), std_hist_T2.ny at the bottom of the code
"""

from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
import datetime
import numpy as np
import pandas as pd
import math
import os
import glob
import pydicom
import re
import SimpleITK as sitk
import sys
from matplotlib import pyplot as plt
from itertools import chain
import tensorflow as tf
import matplotlib.pyplot as plt
import logging
import nibabel as nib
from glob import glob
from scipy.interpolate import interp1d
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
from sklearn import metrics
from sklearn.metrics import auc as skauc
from scipy import interp
from scipy.ndimage import gaussian_filter
from time import strftime
import pickle
import vtk
import pandas
import collections
from pydicom_seg import MultiClassWriter
from pydicom_seg.template import from_dcmqi_metainfo
from pyntcloud import PyntCloud
from skimage.morphology import convex_hull_object
import trimesh, struct
from random import randint

def Step1(inputDir, outputDir):
    if(inputDir is None):
        print("InputDir is not specified!")
        exit()

    if(outputDir is None):
        print("OutputDir is not specified!")
        exit()

    print('inputDir: ' + inputDir)
    print('outputDir: ' + outputDir)

    fusionimagesdir = inputDir
    writexls = True
    writemha = True

    xlspath = os.path.join(outputDir, 'fusionimagesdata.xls')
    OUTPUT_DIR = os.path.join(outputDir, 'AllFusionImages')

    if not os.path.isdir(outputDir):
        os.mkdir(outputDir)
    if not os.path.isdir(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)


    def labelseries(seriesname):
        T2strings = ['T2']

        isT2 = issmFOV = False

        seriesname = seriesname.upper()

        for T2string in T2strings:
            p = re.compile(T2string)
            if re.search(p, seriesname):
                isT2 = True

        if isT2 == True:
            p = re.compile('SMFOV')
            if re.search(p, seriesname):
                issmFOV = True

        if isT2:
            if issmFOV:
                return 'T2 smFOV'
            else:
                return 'T2'
        else:
            return 'error'

    def removeslash(string):
        string = string.replace('/', '')
        string = string.replace('\\', '')
        return string

    def removedashandspace(string):
        string = string.replace(' ', '')
        string = string.replace('-', '')
        string = string.replace('_', '')
        return string

    accession_list = []
    mrdate_list = []
    seriesname_list = []
    mrfiles_list = []
    dimensions_list = []
    numslices_list = []
    serieslabel_list = []
    mhanames_list = []
    pixelspacing_list = []

    outdata = []

    errormsgs = []


    datefolders = [f.path for f in os.scandir(fusionimagesdir) if f.is_dir()]

    imageReader = sitk.ImageSeriesReader()
    writer = sitk.ImageFileWriter()

    processedcounter = 0

    for i, datepath in enumerate(datefolders):
        mhacounter = 0

        seriesIDs = imageReader.GetGDCMSeriesIDs(datepath)

        for series in seriesIDs:
            try:
                seriesfilenames = imageReader.GetGDCMSeriesFileNames(datepath, series)

                imageReader.SetFileNames(seriesfilenames)

                image = imageReader.Execute()
                size = image.GetSize()

                ds = pydicom.read_file(seriesfilenames[0])
                seriesname = ds.SeriesDescription.encode('ascii', 'ignore').decode()
                serieslabel = labelseries(seriesname)

                if serieslabel != 'error':
                    pixelspacing = str(ds.PixelSpacing)
                    accessionnum = removedashandspace(str(ds.AccessionNumber))
                    slicethickness = ds.SliceThickness
                    mrdate = str(ds.StudyDate)
                    firstpos = ds.ImagePositionPatient
                    imgorientationpatient = ds.ImageOrientationPatient
                    institution = ds.InstitutionName
                    manufacturer = ds.Manufacturer
                    series_num = ds[0x20, 0x11]
                  

                    processedcounter += 1

                    mhaname = accessionnum + '_' + mrdate + '_' + str(mhacounter) + '_' + serieslabel
                    outdata.append(dict(zip(['accession', 'institution', 'manufacturer', 'mrdate', 'origfilename', 'seriesname', 'serieslabel', 'numslices', 'dimensions', 'pixelspacing', 'mhafilename', 'StudyDescription', 'ReceiveCoilName', 'MagneticFieldStrength'], 
                                            [accessionnum, institution, manufacturer, mrdate, datepath, seriesname, serieslabel, len(seriesfilenames), size, pixelspacing, mhaname, ds.get('StudyDescription'), ds.get('ReceiveCoilName'), ds.get('MagneticFieldStrength')])))

                    # read the images into mha file
                    reader = sitk.ImageSeriesReader()
                    reader.SetFileNames(seriesfilenames)
                    image = sitk.Cast(reader.Execute(), sitk.sitkFloat32)

                    if writemha == True:
                        writer.Execute(image, os.path.join(OUTPUT_DIR, mhaname + '.mha'), True)

                    if processedcounter % 50 == 0:
                        print('T2s processed: ' + str(processedcounter))
                        outDF = pd.DataFrame(outdata)
                        outDF = outDF[['accession', 'institution', 'manufacturer', 'mrdate', 'seriesname', 'serieslabel', 'numslices', 'dimensions', 'pixelspacing', 'mhafilename', 'ReceiveCoilName', 'MagneticFieldStrength']]
                        if writexls == True :
                            outDF.to_excel(xlspath, sheet_name='Sheet1')


                    mhacounter = mhacounter + 1

            except:
                print('Error in reading ' + str(datepath) + '/' + str(series))
                errormsgs.append(dict(zip(['series', 'datepath'], [str(series), str(datepath)])))

    outDF = pd.DataFrame(outdata)
    outDF = outDF[['accession', 'institution', 'manufacturer', 'mrdate', 'origfilename', 'seriesname', 'serieslabel', 'numslices', 'dimensions', 'pixelspacing', 'mhafilename', 'StudyDescription', 'ReceiveCoilName', 'MagneticFieldStrength']]

    if writexls == True:
        outDF.to_excel(xlspath, sheet_name="Sheet1")
        


def Step2(outputDir):
    def bbox_3D(img):

        z = np.any(img, axis=(1, 2))    #z
        c = np.any(img, axis=(0, 2))    #y
        r = np.any(img, axis=(0, 1))    #x

        rmin, rmax = np.where(r)[0][[0, -1]]
        cmin, cmax = np.where(c)[0][[0, -1]]
        zmin, zmax = np.where(z)[0][[0, -1]]

        #x min max, y min max, z min max
        return rmin, rmax, cmin, cmax, zmin, zmax


    def read_project_mask(accession, deidentified_name, fn_mri_T2, finalsize, finalspacing):
 

        im_mri_T2 = sitk.ReadImage(fn_mri_T2)


        T2_size = im_mri_T2.GetSize()
        T2_spacing = im_mri_T2.GetSpacing()
        T2_direction = im_mri_T2.GetDirection()
        T2_channels = im_mri_T2.GetNumberOfComponentsPerPixel() # number of channels in image

        im_T2_center = np.array(T2_size) / 2.0

         # find physical center of the histology image
        imgcenterphyspace = im_mri_T2.TransformContinuousIndexToPhysicalPoint(im_T2_center)
        #spacing is the spacing from RP cohort
        # f * T2_size[0]*T2_spacing[0] = xsize * spacing

        f1 = (finalsize *finalspacing)/(T2_size[0] *T2_spacing[0])
        f2 = (finalsize *finalspacing)/(T2_size[1]*T2_spacing[1])

        #crop around the center

        #mask_np = sitk.GetArrayFromImage(im_mask)
        #prostate = np.nonzero(mask_np)
        prostate = np.nonzero(sitk.GetArrayFromImage(im_mri_T2))
        x_coor = prostate[2]
        y_coor = prostate[1]
        z_coor = prostate[0]
        zmin = np.min(z_coor)
        zmax = np.max(z_coor)

        xmin = np.min(x_coor)
        xmax = np.max(x_coor)

        ymin = np.min(y_coor)
        ymax = np.max(y_coor)
        T2_npy = sitk.GetArrayFromImage(im_mri_T2)

        width = xmax - xmin
        height = ymax - ymin
        largerdim = np.max([width, height])
        newspacing = [finalspacing, finalspacing, T2_spacing[2]]
        percentprostate = float(largerdim/(f1 * T2_size[0]))

        if True:
            large_prostate = 0
            croppedwidth = croppedheight = int(largerdim / percentprostate)

            cropxmin = int((xmin+xmax - croppedwidth)/2)
            cropxmax = int((xmin+xmax + croppedwidth)/2)
            cropymin = int((ymin+ymax - croppedheight)/2)
            cropymax = int((ymin+ymax + croppedheight)/2)

            cropxmin = max([cropxmin, 0])
            cropxmax = min([cropxmax, T2_size[0]])
            cropymin = max([cropymin, 0])
            cropymax = min([cropymax, T2_size[0]])
            # CROP TO PROSTATE
            newnpy = T2_npy[zmin:zmax, cropymin:cropymax, cropxmin:cropxmax]
            croppedimage = sitk.GetImageFromArray(newnpy)
            # calculate new origin
            topleft = [int(cropxmin), int(cropymin), int(zmin)]
            neworigin = im_mri_T2.TransformIndexToPhysicalPoint(topleft)

            croppedimage.SetOrigin(neworigin)
            croppedimage.SetDirection(T2_direction)
            croppedimage.SetSpacing(T2_spacing)


            # RESAMPLE TO FINAL SIZE 
            finalnpy = np.zeros([croppedimage.GetSize()[2], finalsize, finalsize])
            reference = sitk.GetImageFromArray(finalnpy)
            reference.SetOrigin(neworigin)
            reference.SetDirection(T2_direction)


            newspacing = [finalspacing, finalspacing, T2_spacing[2]]
            reference.SetSpacing(newspacing)


            # MAKING RESAMPLING FILTERS
            resample = sitk.ResampleImageFilter()
            resample.SetReferenceImage(reference)
            resample.SetInterpolator(sitk.sitkLinear)

            # nearest neighbor interpolation for segmentation mask
            resampleNN = sitk.ResampleImageFilter()
            resampleNN.SetReferenceImage(reference)
            resampleNN.SetInterpolator(sitk.sitkNearestNeighbor)


            # RESAMPLE TO finalsize X finalsize x n
            resampledimage = resample.Execute(croppedimage)

            output_T2_name = target_dir_T2 + '/' + deidentified_name + '_res_T2.nii'


            sitk.WriteImage(resampledimage, output_T2_name)


            print("processed", accession, "as", deidentified_name)
        else:

            print ("prostate percent >1, did not process")
            large_prostate = 1
            """
            percentprostate2 = 0.75

            croppedwidth = croppedheight = int(largerdim / percentprostate2)
            cropxmin = int((xmin+xmax - croppedwidth)/2)
            cropxmax = int((xmin+xmax + croppedwidth)/2)
            cropymin = int((ymin+ymax - croppedheight)/2)
            cropymax = int((ymin+ymax + croppedheight)/2)

            cropxmin = max([cropxmin, 0])
            cropxmax = min([cropxmax, T2_size[0]])
            cropymin = max([cropymin, 0])
            cropymax = min([cropymax, T2_size[0]])
            """

        return large_prostate, percentprostate

    directory = os.path.join(outputDir, "AllFusionImages")


    T2_dir = sorted([o for o in os.listdir(directory) if '.mha' in o])
   
    target_directory =  os.path.join(outputDir, 'AllFusionImages_resample')


    if os.path.exists(target_directory) == False:
        os.makedirs(target_directory)

    target_dir_T2 = target_directory 
   

    if os.path.exists(target_dir_T2) == False:
        os.makedirs(target_dir_T2)

   
    finalsize = 256
   
    finalspacing = 0.39

   
    prostate_err_cases = []
    err_cases = []


    accession_percentprostate = {}
    for T2_file in T2_dir:
       
        fn_mri_T2 = os.path.join(directory, T2_file)
       
        accession = T2_file.split('_')[0] + ('_') + T2_file.split('_')[1]
       
        deidentified_name = accession
        

        errorcode = 0

        if os.path.exists(fn_mri_T2) == False:
            print ("T2 does not exist")
            errorcode = 1
       

        if errorcode == 0:
            prostate_err, percentprostate = read_project_mask(accession, deidentified_name, fn_mri_T2, finalsize, finalspacing)
            accession_percentprostate[accession] = percentprostate
        if prostate_err == 1:
            prostate_err_cases.append(accession)
        

def Step3(outputDir, standardHist):


def Step4(outputDir, modelPath):
    
    print(tf.__version__)

    devices = tf.config.experimental.list_physical_devices('GPU')
    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)
    ## Use the following code to specify which GPU to use: 0 Quadro, 1 RTX
    if(len(devices) > 1):
        tf.config.experimental.set_visible_devices(devices[0], 'GPU')

    def get_image_slice_range(image):
        min_slice = 0
        max_slice = image.GetDepth() - 1
        return [min_slice, max_slice]

    use_t2 = True
    stack_input_slices = True
    add_flipped = True
    input_width = 256
    input_height = 256

    def get_slice_file_name(stack_input_slices):
        slice_file_name = '' if stack_input_slices else '_nonstacking' 
        return slice_file_name

    slice_file_name = get_slice_file_name(stack_input_slices)

    def get_model_checkpoint_name():
        return modelPath

    pred_mask_label_path = os.path.join(outputDir, 'prognet_predictions')
    
    if not os.path.isdir(pred_mask_label_path):
        os.mkdir(pred_mask_label_path)

    def unet(pretrained_weights = None,input_size = (256,256,1)):
        inputs = Input(input_size)
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
        merge6 = concatenate([drop4,up6], axis = 3)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

        up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        merge7 = concatenate([conv3,up7], axis = 3)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

        up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
        merge8 = concatenate([conv2,up8], axis = 3)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

        up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
        merge9 = concatenate([conv1,up9], axis = 3)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

        model = Model(inputs = inputs, outputs = conv10)
        model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

        #model.summary()

        if(pretrained_weights):
            model.load_weights(pretrained_weights)

        return model

    
    def showimage(img):
        plt.figure()
        plt.imshow(img)
        #plt.imshow(img, cmap=plt.cm.binary, interpolation='nearest')
        plt.colorbar()
        plt.grid(False)
        plt.show()

    def showimagelist(img_list):
        col=len(img_list)
        fig = plt.figure(figsize=(15,15))
        for i in range(col):
            plt.subplot(1,col,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(img_list[i], cmap=plt.cm.binary)
            plt.xlabel(str(i))
        plt.show()

    def load_t2_file(t2_list, t2_file, t2_path, skip_empty_slices, add_flipped):
        t2_np = 0

        active_slice_range = None

        if use_t2:
            t2 = sitk.ReadImage(os.path.join(t2_path, t2_file))
            t2_np = sitk.GetArrayFromImage(t2)
            t2_slice_range = get_image_slice_range(t2)
            active_slice_range = t2_slice_range

        for index in range(active_slice_range[0], active_slice_range[1] + 1):
            def stack_slices(slices_np, index):
                actual_slice = slices_np[index,:,:,np.newaxis]
                before_slice = []
                after_slice = []
                if index == 0:
                    before_slice = actual_slice
                else:
                    before_slice = slices_np[index-1,:,:,np.newaxis]

                if index == len(slices_np) - 1:
                    after_slice = actual_slice
                else:
                    after_slice = slices_np[index+1,:,:,np.newaxis]

                stacked_slices = np.concatenate((before_slice, actual_slice, after_slice),-1)
                return stacked_slices

            # Add t2
            if stack_input_slices:
                stacked_slices = stack_slices(t2_np, index)

                t2_list.append(stacked_slices)

                if add_flipped:
                    # flip around y-axis (left <-> right)
                    t2_list.append(stacked_slices[:,::-1,:])
            else:
                slice = t2_np[index,:,:,np.newaxis]

                t2_list.append(slice)

                if add_flipped:
                    # flip around y-axis (left <-> right)
                    t2_list.append(slice[:,::-1,:])

    def load_t2(t2_dir, t2_path, skip_empty_slices, add_flipped, file_filter=None):
        t2_list = []
        t2_file_list = []
        counter = 1     

        for t2_file in t2_dir:

            if file_filter != None and t2_file.find(file_filter) < 0:
                continue

            t2_file_list.append(t2_file)

            load_t2_file(t2_list, t2_file, t2_path, skip_empty_slices, add_flipped)

            if file_filter == None:
                print('\rLoading MRI images: %d / %d' % (counter, len(t2_dir)), end='\r')
            else:
                print('\rLoading MRI images: %d' % (counter), end='\r')

            counter = counter + 1

        print("\n")

        if len(t2_file_list) == 0:
            print("Could not find any t2 using file filter: '%s'" % (file_filter))
            return None, None, [], []

        t2_list = np.asarray(t2_list)

        return t2_list, t2_file_list

    def dice_coef(y_true, y_pred, smooth=1):
        # Assume pixles outside the prostate are -1 in y_pred, remove these before calculating y_true
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.where(tf.equal(y_true, 2), tf.zeros_like(y_pred), y_pred)
        y_true = tf.where(tf.equal(y_true, 2), tf.zeros_like(y_true), y_true)

        intersection = K.sum(tf.multiply(y_true, y_pred), axis=[1,2,3])
        union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
        #union = K.sum(y_true) + K.sum(y_pred)
        return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

    def make_model():
        model = unet(input_size=(input_width,input_height, 3 if stack_input_slices else 1))

        opt = Adam(lr=1E-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=[dice_coef])

        #model.summary()
        return model

    def make_and_load_model():
        model = make_model()
        model_checkpoint_name = get_model_checkpoint_name()
        load_model(model, model_checkpoint_name)
        return model    

    def load_model(model, model_checkpoint_name):   
        print("Loading model: " + model_checkpoint_name)
        try:
            model.load_weights(model_checkpoint_name)
        except:
            print("Could not load model: " + model_checkpoint_name)

    def save_prediction(image_data, filename, original_image, active_slice_range=None):
        original_slice_range = get_image_slice_range(original_image)
        if active_slice_range != None:
            original_slice_range = active_slice_range
        print("original_slice_range: " + str(original_slice_range))

        image_data_all_slices = np.zeros((original_image.GetDepth(), original_image.GetWidth(), original_image.GetHeight()), dtype=image_data.dtype)
        image_data_all_slices[original_slice_range[0]:original_slice_range[1]+1,:,:] = np.squeeze(image_data[:,:,:])

        image_data = image_data_all_slices

        pred_vol = 255 * image_data
        pred_vol = np.squeeze(pred_vol)
        pred_vol = sitk.GetImageFromArray(pred_vol)
        pred_vol.SetSpacing(original_image.GetSpacing())
        pred_vol.SetDirection(original_image.GetDirection())
        pred_vol.SetOrigin(original_image.GetOrigin())
        origin_x = original_image.GetOrigin()[0] + (((original_image.GetSize()[0] - pred_vol.GetSize()[0]) * 0.5))*original_image.GetSpacing()[0]
        origin_y = original_image.GetOrigin()[1] + (((original_image.GetSize()[1] - pred_vol.GetSize()[1]) * 0.5))*original_image.GetSpacing()[1]
        pred_vol.SetOrigin([origin_x, origin_y, original_image.GetOrigin()[2]])
        sitk.WriteImage(pred_vol, filename)
        return pred_vol

    def evaluate_patient_partial(patient_number, name, model, t2_path, use_prediction=True):
        t2_dir = [file for file in sorted(os.listdir(t2_path))]
        return evaluate_patient_full(patient_number, name, model, t2_dir, t2_path, use_prediction)

    def get_used_width_height(image_np, threshold):
        max_width = 0
        max_height = 0
        count_y = 0
        for y in range(len(image_np)):
            count_x = 0
            for x in range(len(image_np[y])):
                if(image_np[y, x] >= 0.5):
                    count_x = count_x + 1
            if count_x > max_width:
                max_width = count_x
            if count_x > 0:
                count_y = count_y + 1
        if count_y > max_height:
            max_height = count_y
        return max_width, max_height

    def remove_bottom_small_slides(image_np, prediction_threshold, width_threashold, height_threashold):
        image_np = np.copy(image_np)

        for i in range(len(image_np)):
            max_width, max_height = get_used_width_height(image_np[i,:,:,0], prediction_threshold)

            if max_width <= width_threashold and max_height <= height_threashold:
                image_np[i,:,:,0] = 0
            else:
                break

        return image_np

    def remove_trailing_slides(image_np):
        image_np = np.copy(image_np)
        last_slice_empty = False
        found_non_empty_slice = 0
        for i in range(len(image_np)):
            nonzeroes = np.nonzero(image_np[i])
            if(len(nonzeroes) == 0 or len(nonzeroes[0]) == 0):
                last_slice_empty = True
            else:
                if found_non_empty_slice >= 5 and last_slice_empty:
                    image_np[i,:,:,0] = 0
                else:
                    found_non_empty_slice = found_non_empty_slice + 1
                    last_slice_empty = False

        return image_np

    def remove_front_slides(image_np):
        image_np = np.copy(image_np)
        last_slice_empty = False
        found_non_empty_slice = 0
        for i in reversed(range(len(image_np))):
            nonzeroes = np.nonzero(image_np[i])
            if(len(nonzeroes) == 0 or len(nonzeroes[0]) == 0):
                last_slice_empty = True
            else:
                if found_non_empty_slice >= 5 and last_slice_empty:
                    image_np[i,:,:,0] = 0
                else:
                    found_non_empty_slice = found_non_empty_slice + 1
                    last_slice_empty = False

        return image_np


    def remove_suddenly_increased_slides(image_np, prediction_threshold):
        image_np = np.copy(image_np)

        max_width = 0
        max_height = 0
        last_width = 0
        last_height = 0
        decrease_started = False
        remove_rest = False
        for i in range(len(image_np)):
            if remove_rest:
                image_np[i,:,:,0] = 0
                continue

            width, height = get_used_width_height(image_np[i,:,:,0], prediction_threshold)

            if width > last_width + 7 and height > last_height + 7 and decrease_started:
                remove_rest = True
                image_np[i,:,:,0] = 0
                continue

            if width > max_width and height > max_height:
                max_width = width
                max_height = height

            if width <= max_width - 7 and height <= max_height - 7:
                decrease_started = True

            last_width = width
            last_height = height

        return image_np


    def paint_island_map(image_np_orig, threshold, island_map_np, index, x_orig, y_orig):
        height = len(image_np_orig)
        width = len(image_np_orig[0])

        points_to_check = [[x_orig, y_orig]]

        while len(points_to_check) > 0:
            x, y = points_to_check.pop(0)

            if image_np_orig[y, x] >= threshold and island_map_np[y, x] == 0:
                island_map_np[y, x] = index
            else:
                continue

            if x + 1 < width:
                points_to_check.append([x+1,y])
            if x - 1 >= 0:
                points_to_check.append([x-1,y])

            if y + 1 < height:
                if x + 1 < width:
                    points_to_check.append([x+1,y+1])
                if x - 1 >= 0:
                    points_to_check.append([x-1,y+1])

            if y - 1 >= 0:
                if x + 1 < width:
                    points_to_check.append([x+1,y-1])
                if x - 1 >= 0:
                    points_to_check.append([x-1,y-1])


    def get_island_map(image_np_orig, threshold):
        height = len(image_np_orig)
        width = len(image_np_orig[0])
        island_map_np = np.zeros((height, width), dtype=float)
        island_index = 1
        island_sizes = {}

        for y in range(len(island_map_np)):
            for x in range(len(island_map_np[y])):
                if image_np_orig[y, x] >= threshold and island_map_np[y, x] == 0:
                    index = island_index
                    island_sizes[island_index] = 0
                    island_index = island_index + 1
                    paint_island_map(image_np_orig, threshold, island_map_np, index, x, y)

        for y in range(len(island_map_np)):
            for x in range(len(island_map_np[y])):
                if image_np_orig[y, x] >= threshold:
                    index = island_map_np[y, x]
                    assert(index != 0)
                    island_sizes[index] = island_sizes[index] + 1

        return island_map_np, island_sizes

    def compare_island_maps(island_map_np, last_island_map_np):
        island_count = {}
        for y in range(len(island_map_np)):
            for x in range(len(island_map_np[y])):
                index = island_map_np[y, x]
                if index != 0 and last_island_map_np[y, x] != 0:
                    if index in island_count:
                        island_count[index] = island_count[index] + 1
                    else:
                        island_count[index] = 1

        winner_index = -1
        winner_count = 0
        for index in island_count:
            if island_count[index] > winner_count:
                winner_index = index
                winner_count = island_count[index]

        return winner_index

    def remove_all_other_islands(image_np, island_map_np, index):
        for y in range(len(island_map_np)):
            for x in range(len(island_map_np[y])):
                if island_map_np[y, x] != index:
                    island_map_np[y, x] = 0
                    image_np[y, x] = 0

    def remove_islands_from_slides(image_np, prediction_threshold):
        image_np = np.copy(image_np)

        last_island_map_np = None
        for i in range(len(image_np)):
            island_map_np, island_sizes = get_island_map(image_np[i,:,:,0], prediction_threshold)
            if len(island_sizes) > 1:

                if i > 0:
                    winner_index = compare_island_maps(island_map_np, last_island_map_np)
                    remove_all_other_islands(image_np[i], island_map_np, winner_index)

            last_island_map_np = island_map_np

        return image_np

    def get_slice_pixel_count(image_np, threshold):
        count = 0
        for y in range(len(image_np)):
            for x in range(len(image_np[y])):
                if image_np[y, x] >= threshold:
                    count = count + 1
        return count

    def remove_small_slice_groups(image_np, threshold, max_group_drop_percentage):
        image_np = np.copy(image_np)

        group_list = []
        group_current = []
        group_pixel_count = 0
        group_pixel_count_max_slice = 0
        group_drop_percentage = 0

        for i in range(len(image_np)):
            pixel_count = get_slice_pixel_count(image_np[i,:,:,0], threshold)
            pixel_count_next = 0
            if i + 1 < len(image_np):
                pixel_count_next = get_slice_pixel_count(image_np[i+1,:,:,0], threshold)
            if pixel_count > 0:
                if group_pixel_count_max_slice > 0:
                    group_drop_percentage_current = (group_pixel_count_max_slice - pixel_count) / group_pixel_count_max_slice
                    if group_drop_percentage_current > group_drop_percentage:
                        group_drop_percentage = group_drop_percentage_current
                if pixel_count > group_pixel_count_max_slice:
                    group_pixel_count_max_slice = pixel_count
                elif pixel_count < pixel_count_next and group_drop_percentage >= max_group_drop_percentage:
                    if len(group_current) > 0:
                        group_list.append([group_pixel_count, group_current])
                        group_current = []
                        group_pixel_count = 0
                        group_pixel_count_max_slice = 0
                        group_drop_percentage = 0
                    continue

                group_current.append(i)
                group_pixel_count = group_pixel_count + pixel_count
            elif len(group_current) > 0:
                group_list.append([group_pixel_count, group_current])
                group_current = []
                group_pixel_count = 0
                group_pixel_count_max_slice = 0
                group_drop_percentage = 0

        if len(group_current) > 0:
            group_list.append([group_pixel_count, group_current])
            group_current = []
            group_pixel_count = 0

        max_group_indexes = []
        max_group_pixel_count = 0

        if len(group_list) > 1:
            for i in range(len(group_list)):
                pixel_count, indexes = group_list[i]
                if len(indexes) > len(max_group_indexes):
                    max_group_pixel_count = pixel_count
                    max_group_indexes = indexes

            for i in range(len(image_np)):
                if not i in max_group_indexes:
                    image_np[i,:,:,0] = 0

        return image_np

    def remove_small_slices(image_np, threshold, min_pixel_percentage):
        image_np = np.copy(image_np)

        min_pixel_count = len(image_np[0]) * len(image_np[0][0]) * min_pixel_percentage

        for i in range(len(image_np)):
            pixel_count = get_slice_pixel_count(image_np[i,:,:,0], threshold)
            if pixel_count <= min_pixel_count:
                image_np[i,:,:,0] = 0

        return image_np


    def evaluate_patient_full(patient_number, val_name, model, t2_val_dir, t2_val_path, use_prediction=True):
        print("---- %s ----" % (val_name))
        print("t2_val_path: " + t2_val_path)

        t2_val_list, t2_file_list = load_t2(t2_val_dir, t2_val_path, False, False, patient_number)
        print('t2_file_list: ' + str(t2_file_list))

        if(len(t2_val_list) == 0):
            return

        if(len(t2_file_list) > 1 and patient_number != None):
            print("Error: Found multiple patients matching the patient number. Use 'None' for all patients.")
            return

        prediction = model.predict(t2_val_list, verbose=0)

        prediction_threshold = 0.5
        
        prediction = remove_bottom_small_slides(prediction, prediction_threshold, 29, 29)        
        prediction = gaussian_filter(prediction, sigma=1)
        prediction = remove_bottom_small_slides(prediction, prediction_threshold, 29, 29)

        prediction[prediction >= prediction_threshold] = 1
        prediction[prediction < prediction_threshold] = 0

        #prediction = remove_small_slices(prediction, prediction_threshold, 0.0300)
        prediction = remove_small_slices(prediction, prediction_threshold, 0.0350)
        #prediction = remove_small_slices(prediction, prediction_threshold, 0.0200)
        #prediction = remove_small_slices(prediction, prediction_threshold, 0.0075)
        prediction = remove_islands_from_slides(prediction, prediction_threshold)
        prediction = remove_small_slice_groups(prediction, prediction_threshold, 0.40)
        prediction = remove_trailing_slides(prediction)
        prediction = remove_front_slides(prediction)
        prediction = remove_suddenly_increased_slides(prediction, prediction_threshold)
        
        prediction[prediction >= prediction_threshold] = 1
        prediction[prediction < prediction_threshold] = 0

        if(patient_number != None):
            showimagelist(t2_val_list[:,:,:,1 if stack_input_slices else 0])
            print("prediction (threshold %0.1f)" % (prediction_threshold))
            showimagelist(prediction[:,:,:,0])

            mr_name = "t2_"

            timestr = strftime("%Y%m%d-%H%M%S")
            prediction_file_name = os.path.join(pred_mask_label_path, val_name + '_' + patient_number + "_label_" + slice_file_name + timestr + ".nii");
            print("Filename: " + prediction_file_name)

            t2_val_image = sitk.ReadImage(os.path.join(t2_val_path, t2_file_list[0]))

            active_slice_range = None

            prediction_image = save_prediction(prediction, prediction_file_name, t2_val_image, active_slice_range)


    model = make_and_load_model()

    t2_val_path = os.path.join(outputDir, 'AllFusionImages_resample_normalize')

    mask_patients = [file.split('_')[0] for file in sorted(os.listdir(t2_val_path))]

    def evaluate_patient_excel(name, model, t2_path, use_prediction=True):
        count=1
        for patient in mask_patients:
            print(str(count) + '/' + str(len(mask_patients)))
            count=count+1
            evaluate_patient_partial(patient, name, model, t2_path, use_prediction)

    print(t2_val_path)
    evaluate_patient_excel("ProGNet_Prediction", model, t2_val_path)



def Step5(outputDir):
    excel_file = os.path.join(outputDir, 'fusionimagesdata.xls')

    # Directory containing .mha/nifti files (seg files)
    mhadir = os.path.join(outputDir, 'prognet_predictions')

    # Directory where seg files will be written
    dcmdir = os.path.join(outputDir, 'stl_output')
    
    if not os.path.isdir(dcmdir):
        os.mkdir(dcmdir)

    mhafilenames = os.listdir(mhadir)
    mhafilenames = [f for f in mhafilenames if os.path.isfile(os.path.join(mhadir, f)) and '.nii' in f]

    df1 = pandas.read_excel(excel_file)
    df2 = pandas.core.frame.DataFrame()
    df2[['accession', 'origfilename']] = df1[['accession', 'origfilename']]
    accession_list = [f for f in df2.values.tolist() if str(f[0]) in [f.split('_')[1] for f in mhafilenames]]
    accession_list = [[str(f[0]), f[1]] for f in accession_list]

    def removedashandspace(string):
        string = string.replace(' ', '')
        string = string.replace('-', '')
        string = string.replace('_', '')
        return string

    def nii_2_mesh (filename_nii, filename_stl, label):

        """
        Read a nifti file including a binary map of a segmented organ with label id = label. 
        Convert it to a smoothed mesh of type stl.
        filename_nii     : Input nifti binary map 
        filename_stl     : Output mesh name in stl format
        label            : segmented label id 
        """

        print("Loading: " + filename_nii)

        # read the file
        reader = vtk.vtkNIFTIImageReader()
        reader.SetFileName(filename_nii)
        reader.Update()

        # apply marching cube surface generation
        surf = vtk.vtkDiscreteMarchingCubes()
        surf.SetInputConnection(reader.GetOutputPort())
        surf.SetValue(0, label) # use surf.GenerateValues function if more than one contour is available in the file
        surf.Update()

        #smoothing the mesh
        smoother= vtk.vtkWindowedSincPolyDataFilter()
        if vtk.VTK_MAJOR_VERSION <= 5:
            smoother.SetInput(surf.GetOutput())
        else:
            smoother.SetInputConnection(surf.GetOutputPort())
        smoother.SetNumberOfIterations(30) 
        smoother.SetPassBand(0.00001)
        smoother.Update()

        # transform the data into DICOM 
        transform = vtk.vtkTransform()
        transform.SetMatrix(reader.GetQFormMatrix())
        transform.PostMultiply()

        pDataTF = vtk.vtkTransformPolyDataFilter()
        pDataTF.SetInputConnection(smoother.GetOutputPort())
        pDataTF.SetTransform(transform);
        pDataTF.Update();

        # save the output
        writer = vtk.vtkSTLWriter()
        writer.SetInputConnection(pDataTF.GetOutputPort())
        writer.SetFileTypeToASCII()
        writer.SetFileName(filename_stl)
        print("Saving: " + filename_stl)
        writer.Write()

    mhafilenames = os.listdir(mhadir)

    for mhafilename in mhafilenames :
      accession = mhafilename.split('_')[2]
      #output_filename = os.path.join(dcmdir, mhafilename + '.stl')
      output_filename = os.path.join(dcmdir, accession + '.stl')
      #print('Saving: ' + output_filename)

      input_filename = os.path.join(mhadir, mhafilename)

      nii_2_mesh(input_filename, output_filename, 255)

   

def CreateSegFile(PatientName, PatientID, PatientBirthDate, StudyInstanceUID, AccessionNumber, numberOfSurfacePoints, pointCoordinatesData, trianglePointIndexList):
    ds = pydicom.Dataset()
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    
    # Set creation date/time
    dt = datetime.datetime.now()
    ds.InstanceCreationDate = dt.strftime('%Y%m%d')
    timeStr = dt.strftime('%H%M%S.%f')  # long format with micro seconds
    ds.InstanceCreationTime = timeStr
    
    #sop_class_uid = pydicom._storage_sopclass_uids.SegmentationStorage
    
    ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.66.5'  
    ds.SOPInstanceUID = pydicom.uid.generate_uid()   
   
    ds.ContentDate = ds.InstanceCreationDate
    ds.ContentTime = ds.InstanceCreationTime
    ds.AccessionNumber = AccessionNumber
    ds.Modality = 'SEG'    
    ds.ReferringPhysicianName = ''
    ds.StudyDescription = ''
    ds.SeriesDescription = ''
    ds.PerformingPhysicianName = ''
    ds.PatientName = PatientName #<- must be original
    ds.PatientID = PatientID
    ds.PatientBirthDate = PatientBirthDate #<- must be original
    
    ds.StudyInstanceUID = StudyInstanceUID #<- must be original
    
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()


    ds.StudyID = '0'
    ds.SeriesNumber = '1'
    ds.InstanceNumber = '1'

    
    ds.SegmentSequence = pydicom.Sequence([pydicom.Dataset()])
    ds.SegmentSequence[0].AnatomicRegionSequence = pydicom.Sequence([pydicom.Dataset()])
    ds.SegmentSequence[0].AnatomicRegionSequence[0].CodeValue = 'T-9200B'
    ds.SegmentSequence[0].AnatomicRegionSequence[0].CodingSchemeDesignator = 'SRT'
    ds.SegmentSequence[0].AnatomicRegionSequence[0].CodeMeaning = 'Prostate'
    ds.SegmentSequence[0].SegmentedPropertyCategoryCodeSequence = pydicom.Sequence([pydicom.Dataset()])
    ds.SegmentSequence[0].SegmentedPropertyCategoryCodeSequence[0].CodeValue = 'T-D0050'
    ds.SegmentSequence[0].SegmentedPropertyCategoryCodeSequence[0].CodingSchemeDesignator = 'SRT'
    ds.SegmentSequence[0].SegmentedPropertyCategoryCodeSequence[0].CodeMeaning = 'Tissue'
    ds.SegmentSequence[0].SegmentNumber = 1
    ds.SegmentSequence[0].SegmentLabel = 'Prostate'
    ds.SegmentSequence[0].SegmentedPropertyTypeCodeSequence = pydicom.Sequence([pydicom.Dataset()])
    ds.SegmentSequence[0].SegmentedPropertyTypeCodeSequence[0].CodeValue = 'T-92000'
    ds.SegmentSequence[0].SegmentedPropertyTypeCodeSequence[0].CodingSchemeDesignator = 'SRT'
    ds.SegmentSequence[0].SegmentedPropertyTypeCodeSequence[0].CodeMeaning = 'Prostate'
    ds.SegmentSequence[0].SurfaceCount = 1
    ds.SegmentSequence[0].ReferencedSurfaceSequence = pydicom.Sequence([pydicom.Dataset()])
    ds.SegmentSequence[0].ReferencedSurfaceSequence[0].ReferencedSurfaceNumber  = 1
    ds.SegmentSequence[0].ReferencedSurfaceSequence[0].SegmentSurfaceGenerationAlgorithmIdentificationSequence = pydicom.Sequence([pydicom.Dataset()])
    ds.SegmentSequence[0].ReferencedSurfaceSequence[0].SegmentSurfaceGenerationAlgorithmIdentificationSequence[0].CodeValue = '123109'
    ds.SegmentSequence[0].ReferencedSurfaceSequence[0].SegmentSurfaceGenerationAlgorithmIdentificationSequence[0].CodingSchemeDesignator = 'DCM'
    ds.SegmentSequence[0].ReferencedSurfaceSequence[0].SegmentSurfaceGenerationAlgorithmIdentificationSequence[0].CodeMeaning = 'Manual Processing'

    
    ds.SurfaceSequence = pydicom.Sequence([pydicom.Dataset()])
    ds.SurfaceSequence[0].RecommendedDisplayGrayscaleValue = 255
    ds.SurfaceSequence[0].RecommendedDisplayCIELabValue = [65535, 32768, 32768]
    ds.SurfaceSequence[0].SurfaceNumber = 1
    ds.SurfaceSequence[0].SurfaceComments = 'ProFuse segmentation of prostate'
    ds.SurfaceSequence[0].SurfaceProcessing = 'NO'
    ds.SurfaceSequence[0].SurfaceProcessingRatio = 1.0
    ds.SurfaceSequence[0].RecommendedPresentationOpacity = 1.0
    ds.SurfaceSequence[0].RecommendedPresentationType = 'SURFACE'
    ds.SurfaceSequence[0].FiniteVolume = 'YES'
    ds.SurfaceSequence[0].Manifold = 'YES'
    ds.SurfaceSequence[0].SurfacePointsSequence = pydicom.Sequence([pydicom.Dataset()])
    ds.SurfaceSequence[0].SurfacePointsSequence[0].NumberOfSurfacePoints = numberOfSurfacePoints
    ds.SurfaceSequence[0].SurfacePointsSequence[0].PointCoordinatesData = pointCoordinatesData
    ds.SurfaceSequence[0].SurfaceMeshPrimitivesSequence = pydicom.Sequence([pydicom.Dataset()])
    ds.SurfaceSequence[0].SurfaceMeshPrimitivesSequence[0].TrianglePointIndexList = trianglePointIndexList


    ds.add_new(0x11291050, 'DS', "0.0")
    ds.add_new(0x11291056, 'IS', "0")
    ds.add_new(0x11291057, 'CS', "3.0")
    
    #print(ds)
    return ds
        

def Step6(output):

    hdpath =  output

    mhadir = os.path.join(output, 'prognet_predictions')
    mhafilenames = os.listdir(mhadir)
    mhafilenames = [f for f in mhafilenames if os.path.isfile(os.path.join(mhadir, f)) and '.nii' in f]

    mhafilenames = os.listdir(mhadir)

    for mhafilename in mhafilenames :
      accession = mhafilename.split('_')[1]

    # Path containing original exported T2 series and SEG files
    profuseexportpath = inputDir

    # Path where mesh .stl files are stored
    stlpath = os.path.join(output, 'stl_output')

    # Destination path where new DICOM segs will be written
    destpath = os.path.join(output, 'NewDCOM')
    if not os.path.isdir(destpath):
        os.mkdir(destpath)
            
    patientfilepaths = [f.path for f in os.scandir(profuseexportpath) if f.is_dir()]
    patientsegpaths = [os.path.join(f, 'SEG.dcm') for f in patientfilepaths]

    def int_to_hexletter(x) :
      if x == 10 :
        return 'A'
      elif x == 11 :
        return 'B'
      elif x == 12 :
        return 'C'
      elif x == 13 :
        return 'D'
      elif x == 14 :
        return 'E'
      elif x == 15 :
        return 'F'
      else :
        return str(x)


    # Load original DICOM SEG file that is exported for each patient and replace header with segmentation mesh
    for patientsegpath in patientsegpaths: # patientfilepaths :
      # Pull original SEG file
      # patientseries = os.listdir(patientfilepath)
      # origsegfilename = [f for f in patientseries if 'SEG' in f][0]
      # origsegpath = os.path.join(patientfilepath, origsegfilename)

      origsegpath = patientsegpath
      patientfilepath = patientsegpath[:-8]
    #  print(patientfilepath)
      patientfiles = os.listdir(patientfilepath)
      print(patientfiles)
      patientMRpaths = [os.path.join(patientfilepath, f) for f in patientfiles if 'MR' in f]
      iter = 1
      accession = ''
      seriesdate = ''
      PatientName = None
      PatientID = None
      PatientBirthDate = None
      StudyInstanceUID = None

      outputpatientdir = ''

      for patientMRpath in patientMRpaths :
        ds = pydicom.read_file(patientMRpath)
        if iter == 1 :
          accession = ds.AccessionNumber
          seriesdate = ds.SeriesDate
          PatientName = ds.PatientName
          PatientID = ds.PatientID
          PatientBirthDate = ds.PatientBirthDate
          StudyInstanceUID = ds.StudyInstanceUID
            
          outputpatientdir = os.path.join(destpath, accession + '_' + seriesdate)
          if not os.path.isdir(outputpatientdir) :
            os.mkdir(outputpatientdir)

        ds.save_as(os.path.join(outputpatientdir, 'MR' + str(iter) + '.dcm'))
        iter += 1

      #origsegdcm = pydicom.read_file(origsegpath)

      # Find patient's stl file with the accession
      #stlfilepath = os.path.join(stlpath, mhafilename + '.stl')
      stlfilepath = os.path.join(stlpath, accession + '.stl')
      print('Loading ' + stlfilepath)

      stlmesh =  trimesh.load(stlfilepath)
      stlmesh.apply_scale((-1, -1, 1))

      # Each element sublist of nine floats comprise the three 3D coordinate vectors for each triangle
      points = stlmesh.vertices
      points_list = []

      for i in points :
        points_list += i.tolist()

      tripoints = stlmesh.faces

      tripoints_totalhex_string = ""

      for tripoint in tripoints :
        for dim_value in tripoint :
          third_littleend_value = int(dim_value/(16**3))
          fourth_littleend_value = int(dim_value%(16**3)/(16**2))
          first_littleend_value = int(dim_value%(16**3)%(16**2)/16)
          second_littleend_value = int(dim_value%(16**3)%(16**2)%16)
          tripoints_totalhex_string += "".join(map(str, [int_to_hexletter(first_littleend_value), int_to_hexletter(second_littleend_value), int_to_hexletter(third_littleend_value), int_to_hexletter(fourth_littleend_value)]))

      tripoints_hexbytes = bytes([int(tripoints_totalhex_string[i:i+2], 16) for i in range(0, len(tripoints_totalhex_string), 2)])
      #print(tripoints_hexbytes)

      # Replace header information of the original SEG file with the mesh surface
      #origsegdcm.SurfaceSequence[0][0x66, 0x11][0][0x66, 0x16].value = points_list # Point coordinates data tag
      #origsegdcm.SurfaceSequence[0][0x66, 0x13][0][0x66, 0x23].value = tripoints_hexbytes # Triangle point index list tag

      newsegfilepath = os.path.join(outputpatientdir, 'SEG.dcm')
      #origsegdcm.save_as(newsegfilepath)

      #numberOfSurfacePoints = ds.SurfaceSequence[0].SurfacePointsSequence[0].NumberOfSurfacePoints
      #pointCoordinatesData = ds.SurfaceSequence[0].SurfacePointsSequence[0].PointCoordinatesData
      #trianglePointIndexList = ds.SurfaceSequence[0].SurfaceMeshPrimitivesSequence[0].TrianglePointIndexList  
        
      numberOfSurfacePoints = 601
      #numberOfSurfacePoints = int(len(points_list)/3)
      pointCoordinatesData = points_list
      trianglePointIndexList = tripoints_hexbytes
        
      ds_new = CreateSegFile(PatientName, PatientID, PatientBirthDate, StudyInstanceUID, accession, numberOfSurfacePoints, pointCoordinatesData, trianglePointIndexList)

      pydicom.dcmwrite(newsegfilepath, ds_new, True)


      #print(int(len(points_list)/3))
      #print(origsegdcm.SurfaceSequence[0][0x66, 0x11][0][0x66, 0x15])
      #print(origsegdcm)
        
      #except :
      #  print('Unable to load ' + stlfilepath)

     
inputDir = "Z:/Input"
outputDir = "Z:/Output"
standardHist = 'Z:/std_hist_T22.npy'
modelPath = 'Z:/prognet_t2.h5'

Step1(inputDir, outputDir)
Step2(outputDir)
Step3(outputDir, standardHist)
Step4(outputDir, modelPath)
Step5(outputDir)
Step6(outputDir)
print('Finished.')
