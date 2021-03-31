# general imports
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# image manipulation
from PIL import Image as im
import os
from keras.preprocessing.image import load_img, ImageDataGenerator

# keras/tensorflow
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

# Class NeuralNet

class NeuralNet():
    '''
    Takes in ___ and provides a slew of methods to preprocess data, visualize data, model data, and tune model.
    '''
    def __init__(self):
        
        # Directory paths
        
            # binary
        self.binary_train_path = '/chest_xray/train/'
        self.binary_test_path = '/chest_xray/test/'
        self.binary_train_pneumonia_path = '/chest_xray/train/PNEUMONIA/'
        self.binary_train_normal_path = '/chest_xray/train/NORMAL/'
        self.binary_test_pneumonia_path = '/chest_xray/test/PNEUMONIA/'
        self.binary_test_normal_path = '/chest_xray/test/NORMAL/'
            # ternary
        self.ternary_train_path = '/chest_xray/chest_xray_ternary/train/'
        self.ternary_test_path = '/chest_xray/chest_xray_ternary/test/'
        self.ternary_train_bacterial_path = '/chest_xray/chest_xray_ternary/train/BACTERIAL/'
        self.ternary_train_viral_path = '/chest_xray/chest_xray_ternary/train/VIRAL/'
        self.ternary_train_normal_path = '/chest_xray/chest_xray_ternary/train/NORMAL/'
        self.ternary_test_bacterial_path = '/chest_xray/chest_xray_ternary/test/BACTERIAL/'
        self.ternary_test_viral_path = '/chest_xray/chest_xray_ternary/test/VIRAL/'
        self.ternary_test_normal_path = '/chest_xray/chest_xray_ternary/test/NORMAL/'
        
        # List of images
        self.img_train_normal = []
        self.img_train_bacterial = []
        self.img_train_viral = []
        self.img_test_normal = []
        self.img_test_bacterial = []
        self.img_test_viral = []
        
        # Pandas Dataframe of images and info
        self.df_ = None
        
        # List of array-formatted images
        self.array_train_normal = []
        self.array_train_bacterial = []
        self.array_train_viral = []
        self.array_test_normal = []
        self.array_test_bacterial = []
        self.array_test_viral = []
        
        # List of gs_sums
        self.sums_train_normal = []
        self.sums_train_bacterial = []
        self.sums_train_viral = []
        self.sums_test_normal = []
        self.sums_test_bacterial = []
        self.sums_test_viral = []
        
        # List of model data
        self.binary_train_images = None
        self.binary_train_labels = None
        self.binary_test_images = None
        self.binary_test_labels = None
        
        self.ternary_train_images = None
        self.ternary_train_labels = None
        self.ternary_test_images = None
        self.ternary_test_labels = None

    def preprocess(self, folder='data', rotation_range=0.4, zoom_range=0.4):
        '''
        Works like a fit method, takes in name of folder (str) that stores data and then stores in class the following:
        - image list (PIL.Image)
        - array list
        - sum list
        - data to be inserted into model
        
        NOTE: MUST have directory structure as follows...
        
        folder
            >chest_xray
                >train
                    >PNEUMONIA
                    >NORMAL
                >test
                    >NORMAL
                    >PNEUMONIA
                >chest_xray_ternary
                    >train
                        >NORMAL
                        >PNEUMONIA
                            >BACTERIAL
                            >VIRAL
                    >test
                        >NORMAL
                        >PNEUMONIA
                            >BACTERIAL
                            >VIRAL
        '''
        # Binary data
        train_pneumonia=os.listdir(folder+self.binary_train_pneumonia_path)
        train_normal=os.listdir(folder+self.binary_train_normal_path)
        test_normal=os.listdir(folder+self.binary_test_normal_path)
        test_pneumonia=os.listdir(folder+self.binary_test_pneumonia_path)
        
        # Ternary data
        train_bacterial=os.listdir(folder+self.ternary_train_bacterial_path)
        train_viral=os.listdir(folder+self.ternary_train_viral_path)
        train_tern_normal=os.listdir(folder+self.ternary_train_normal_path)
        test_tern_normal=os.listdir(folder+self.ternary_test_normal_path)
        test_bacterial=os.listdir(folder+self.ternary_test_bacterial_path)
        test_viral=os.listdir(folder+self.ternary_test_viral_path)
        
        print('Image paths loaded from folder(s)...')
        
        # Create dataframes for each permutation of image
        
        arrays = [self.array_train_bacterial, self.array_train_viral, self.array_train_normal,
                  self.array_test_bacterial, self.array_test_viral, self.array_test_normal]
        
        images = [self.img_train_bacterial, self.img_train_viral, self.img_train_normal,
                  self.img_test_bacterial, self.img_test_viral, self.img_test_normal]
        
        gs_sums = [self.sums_train_bacterial, self.sums_train_viral, self.sums_train_normal, 
                   self.sums_test_bacterial, self.sums_test_viral, self.sums_test_normal]
        
        dirs = [train_bacterial, train_viral, train_tern_normal, 
                test_bacterial, test_viral, test_tern_normal]
        
        paths = [self.ternary_train_bacterial_path, self.ternary_train_viral_path, self.ternary_train_normal_path,
                 self.ternary_test_bacterial_path, self.ternary_test_viral_path, self.ternary_test_normal_path]
        
        for i in range(len(dirs)):
            for img in dirs[i]:
                image = im.open(folder+paths[i]+img)
                new_img = image.resize((224,224))
                images[i].append(new_img)
                arrays[i].append(np.array(new_img))
                gs_sums[i].append(np.array(new_img).sum())
        
        print('Converted images into PIL.Image.Image and array formats...')
        
        # Generate dataframe with images, label info, and grayscale sums
        
        train_bacterial_resized=pd.DataFrame(images[0], columns=['image'])
        train_bacterial_resized['label'] = 'bacterial'
        train_bacterial_resized['train'] = 1
        train_bacterial_resized['gs_sum'] = self.sums_train_bacterial
        
        train_viral_resized=pd.DataFrame(images[1], columns=['image'])
        train_viral_resized['label'] = 'viral'
        train_viral_resized['train'] = 1
        train_viral_resized['gs_sum'] = self.sums_train_viral
        
        train_normal_resized=pd.DataFrame(images[2], columns=['image'])
        train_normal_resized['label'] = 'normal'
        train_normal_resized['train'] = 1
        train_normal_resized['gs_sum'] = self.sums_train_normal
        
        test_bacterial_resized=pd.DataFrame(images[3], columns=['image'])
        test_bacterial_resized['label'] = 'bacterial'
        test_bacterial_resized['train'] = 0 
        test_bacterial_resized['gs_sum'] = self.sums_test_bacterial
        
        test_viral_resized=pd.DataFrame(images[4], columns=['image'])
        test_viral_resized['label'] = 'viral'
        test_viral_resized['train'] = 0
        test_viral_resized['gs_sum'] = self.sums_test_viral
        
        test_normal_resized=pd.DataFrame(images[5], columns=['image'])
        test_normal_resized['label'] = 'normal'
        test_normal_resized['train'] = 0
        test_normal_resized['gs_sum'] = self.sums_test_normal
        
        # Combine all the dfs
        self.df_ = pd.concat([train_bacterial_resized, train_viral_resized, train_normal_resized, 
                              test_bacterial_resized, test_viral_resized, test_normal_resized], axis=0)
        print('Stored dataframe of data in .df_ attribute...')
        
        # Generate images for modeling (batch size matches length of full dataset to allow for adjustable batch sizes later
        
        train_batch_size = len(self.img_train_normal)+len(self.img_train_bacterial)+len(self.img_train_viral)
        test_batch_size = len(self.img_test_normal)+len(self.img_test_bacterial)+len(self.img_test_viral)
        
        # BINARY
        binary_test_gen = ImageDataGenerator(rescale = 1/255.).flow_from_directory(folder+self.binary_test_path,
                                            target_size=(224, 224),
                                            batch_size=test_batch_size,
                                            color_mode = 'grayscale',                        
                                            class_mode='binary')

        
        binary_train_gen = ImageDataGenerator(rescale = 1/255., horizontal_flip=True, \
                                              rotation_range=rotation_range, \
                                              zoom_range=zoom_range).flow_from_directory(folder+self.binary_train_path,
                                                                                         target_size=(224, 224),
                                                                                         batch_size=train_batch_size,
                                                                                         color_mode = 'grayscale',                 
                                                                                         class_mode='binary')
        
        # TERNARY
        ternary_test_gen = ImageDataGenerator(rescale = 1/255.).flow_from_directory(folder+self.ternary_test_path,
                                            target_size=(224, 224),
                                            batch_size=test_batch_size,
                                            color_mode = 'grayscale',                        
                                            class_mode='categorical')

        
        ternary_train_gen = ImageDataGenerator(rescale = 1/255., horizontal_flip=True, \
                                               rotation_range=rotation_range, \
                                               zoom_range=zoom_range).flow_from_directory(folder+self.ternary_train_path,
                                                                                          target_size=(224, 224),
                                                                                          batch_size=train_batch_size,
                                                                                          color_mode = 'grayscale',               
                                                                                          class_mode='categorical')
        
        # Isolating data, reshaping for model
        
        # BINARY TRAIN
        binary_train_images, binary_train_labels = next(binary_train_gen)
        self.binary_train_images = binary_train_images.reshape(binary_train_images.shape[0], -1)
        self.binary_train_labels = np.reshape(binary_train_labels[:], (5232,1))
        # BINARY TEST
        binary_test_images, binary_test_labels = next(binary_test_gen)
        self.binary_test_images = binary_test_images.reshape(binary_test_images.shape[0], -1)
        self.binary_test_labels = np.reshape(binary_test_labels[:], (624,1))
        
        # TERNARY TRAIN
        ternary_train_images, ternary_train_labels = next(ternary_train_gen)
        self.ternary_train_images = ternary_train_images.reshape(ternary_train_images.shape[0], -1)
        self.ternary_train_labels = ternary_train_labels[:,0].reshape(-1,1)
        # TERNARY TEST
        ternary_test_images, ternary_test_labels = next(ternary_test_gen)
        self.ternary_test_images = ternary_test_images.reshape(ternary_test_images.shape[0], -1)
        self.ternary_test_labels = ternary_test_labels[:,0].reshape(-1,1)
                                                      
        print('Data is ready for modeling.\n\nYou can check out the preprocessed data with the following attributes: \n\n.binary_test_images\n.binary_train_images\n.binary_train_labels\n.ternary_train_images\n.ternary_test_images\n.ternary_train_labels\netc.')                                                
        
                                                        
    def build_model():
        '''
        Takes in dataset with correct shape, returns None, but stores fit model object in the class. If ternary=True, then builds model that distinguishes bacteria vs viral pneumonia.
        '''
        return None

    def get_results():
        '''
        Takes in model and returns confusion matrix, accuracy, summary table; diagnostics can be chosen, but by default all are returned. If user does not want to wait forever for a model to build, if a param is set to True, will return summary of previously built model. Also should have ability to return graph of loss and accuracy/recall growth across epochs. Don't know if this will have to be segmented via attributes.
        '''
        return None

    def tensorboard():
        '''
        Takes in _____ and launches Tensorboard interface AND/OR returns images taken for previously built model if user does not want to launch interface.
        '''
        return None

