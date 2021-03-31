# general imports
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import numpy as np

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
    def __init__():
        
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
        self.img_binary_normal = None
	    self.img_binary_pneumonia = None
	    self.img_ternary_normal = None
        self.img_ternary_bacterial = None
        self.img_ternary_viral = None
        
        # Pandas Dataframe of images and info
        self.df_ = None
        
        # List of array-formatted images
        self.array_binary_normal = None
	    self.array_binary_pneumonia = None
	    self.array_ternary_normal = None
        self.array_ternary_bacterial = None
        self.array_ternary_viral = None
        
        # List of model data
        self.binary_train = None
        self.binary_labels = None
        self.binary_test = None
        self.ternary_train = None
        self.ternary_labels = None
        self.ternary_test = None

    def preprocess(folder='data'):
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
        
        print('Data Loaded from folder(s).')
        
        #
        self.array_binary_normal = None
	    self.array_binary_pneumonia = None
	    self.array_ternary_normal = None
        self.array_ternary_bacterial = None
        self.array_ternary_viral = None
        
        # Generate dataframe with images, label info, and grayscale sums
        
        train_bacterial_resized=pd.DataFrame()
        train_bacterial_resized['label'] = 'bacterial'
        train_bacterial_resized['train'] = 1
        train_bacterial_resized['gs_sum'] = 
        
        train_viral_resized=pd.DataFrame()
        train_viral_resized['label'] = 'viral'
        train_viral_resized['train'] = 1
        train_viral_resized['gs_sum'] =
        
        train_normal_resized=pd.DataFrame()
        train_normal_resized['label'] = 'normal'
        train_normal_resized['train'] = 1
        train_normal_resized['gs_sum'] =
        
        test_normal_resized=pd.DataFrame()
        test_normal_resized['label'] = 'normal'
        test_normal_resized['train'] = 0
        test_normal_resized['gs_sum'] =
        
        test_bacterial_resized=pd.DataFrame()
        test_bacterial_resized['label'] = 'bacterial'
        test_bacterial_resized['train'] = 0 
        test_bacterial_resized['gs_sum'] =
        
        test_viral_resized=pd.DataFrame()
        test_viral_resized['label'] = 'viral'
        test_viral_resized['train'] = 0
        test_viral_resized['gs_sum'] =
        
        # Combine all the dfs
        self.df_ = pd.concat([train_bacterial_resized, train_viral_resized, train_normal_resized, 
                              test_bacterial_resized, test_viral_resized, test_normal_resized], axis=0)
        print('Stored dataframe of data in .df_ attribute.')
        
        # Generate images for modeling (batch size matches length of full dataset to allow for adjustable batch sizes later
        
        # BINARY
        binary_test_gen = ImageDataGenerator(rescale = 1/255.).flow_from_directory(test_path,
                                            target_size=(224, 224),
                                            batch_size=624,
                                            color_mode = 'grayscale',                        
                                            class_mode='binary')

        
        binary_train_gen = ImageDataGenerator(rescale = 1/255.).flow_from_directory(full_train_path,
                                            target_size=(224, 224),
                                            batch_size=5232,
                                            color_mode = 'grayscale',                        
                                            class_mode='binary')
        
        # TERNARY
         ternary_test_gen = ImageDataGenerator(rescale = 1/255.).flow_from_directory(test_path,
                                            target_size=(224, 224),
                                            batch_size=624,
                                            color_mode = 'grayscale',                        
                                            class_mode='binary')

        
        ternary_train_gen = ImageDataGenerator(rescale = 1/255.).flow_from_directory(full_train_path,
                                            target_size=(224, 224),
                                            batch_size=5232,
                                            color_mode = 'grayscale',                        
                                            class_mode='binary')
        
        # Isolating data, reshaping for model
        
        # BINARY TRAIN
        binary_train_images, binary_train_labels = next(binary_train_gen)
        binary_train_images = full_train_images.reshape(binary_train_images.shape[0], -1)
        binary_train_labels = np.reshape(binary_train_labels[:], (5232,1))
        # BINARY TEST
        binary_test_images, binary_test_labels = next(ternary_test_gen)
        binary_test_images = binary_test_images.reshape(binary_test_images.shape[0], -1)
        binary_test_labels = np.reshape(binary_test_labels[:], (624,1))
        
        # TERNARY TRAIN
        ternary_train_images, ternary_train_labels = next(ternary_train_gen)
        ternary_train_images = ternary_train_images.reshape(ternary_train_images.shape[0], -1)
        ternary_train_labels = np.reshape(ternary_train_labels[:], (5232,1))
        # TERNARY TEST
        ternary_test_images, ternary_test_labels = next(ternary_test_gen)
        ternary_test_images = ternary_test_images.reshape(ternary_test_images.shape[0], -1)
        ternary_test_labels = np.reshape(ternary_test_labels[:], (624,1))

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

