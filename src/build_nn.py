# general imports
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import numpy as np
import pandas as pd
import seaborn as sns

# image manipulation
from PIL import Image as im
import os
from keras.preprocessing.image import load_img, ImageDataGenerator

# keras/tensorflow
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import metrics
from keras.layers.advanced_activations import LeakyReLU
from tensorflow.random import set_seed

# Diagnostics/Analysis
from sklearn.metrics import roc_curve, auc, confusion_matrix

# Set global seed
set_seed(42)

####################### Class NeuralNet ########################

class NeuralNet():
    '''
    Takes in ___ and provides a slew of methods to preprocess data, visualize data, model data, and tune model.
    '''
    def __init__(self):
        
        # Directory paths
        
            # normal scans
        self.train_normal_path = '/chest_xray/train/NORMAL/'
        self.test_normal_path = '/chest_xray/test/NORMAL/'
            # binary
        self.binary_train_path = '/chest_xray/train/'
        self.binary_test_path = '/chest_xray/test/'
        self.binary_train_pneumonia_path = '/chest_xray/train/PNEUMONIA/'
        self.binary_test_pneumonia_path = '/chest_xray/test/PNEUMONIA/'
            # ternary
        self.ternary_train_path = '/chest_xray/chest_xray_ternary/train/'
        self.ternary_test_path = '/chest_xray/chest_xray_ternary/test/'
        self.ternary_train_bacterial_path = '/chest_xray/chest_xray_ternary/train/BACTERIAL/'
        self.ternary_train_viral_path = '/chest_xray/chest_xray_ternary/train/VIRAL/'
        self.ternary_test_bacterial_path = '/chest_xray/chest_xray_ternary/test/BACTERIAL/'
        self.ternary_test_viral_path = '/chest_xray/chest_xray_ternary/test/VIRAL/'
        
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
        
        # Model
        self.history = None
        self.model = None
        self.predictions = None

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
        # Using same normal data
        train_normal=os.listdir(folder+self.train_normal_path)
        test_normal=os.listdir(folder+self.test_normal_path)
        
        # Binary data
        train_pneumonia=os.listdir(folder+self.binary_train_pneumonia_path)
        test_pneumonia=os.listdir(folder+self.binary_test_pneumonia_path)
        
        # Ternary data
        train_bacterial=os.listdir(folder+self.ternary_train_bacterial_path)
        train_viral=os.listdir(folder+self.ternary_train_viral_path)
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
        
        dirs = [train_bacterial, train_viral, train_normal, 
                test_bacterial, test_viral, test_normal]
        
        paths = [self.ternary_train_bacterial_path, self.ternary_train_viral_path, self.train_normal_path,
                 self.ternary_test_bacterial_path, self.ternary_test_viral_path, self.test_normal_path]
        
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
        train_bacterial_resized['test'] = 0
        train_bacterial_resized['gs_sum'] = self.sums_train_bacterial
        
        train_viral_resized=pd.DataFrame(images[1], columns=['image'])
        train_viral_resized['label'] = 'viral'
        train_viral_resized['train'] = 1
        train_viral_resized['test'] = 0
        train_viral_resized['gs_sum'] = self.sums_train_viral
        
        train_normal_resized=pd.DataFrame(images[2], columns=['image'])
        train_normal_resized['label'] = 'normal'
        train_normal_resized['train'] = 1
        train_normal_resized['test'] = 0
        train_normal_resized['gs_sum'] = self.sums_train_normal
        
        test_bacterial_resized=pd.DataFrame(images[3], columns=['image'])
        test_bacterial_resized['label'] = 'bacterial'
        test_bacterial_resized['train'] = 0 
        test_bacterial_resized['test'] = 1
        test_bacterial_resized['gs_sum'] = self.sums_test_bacterial
        
        test_viral_resized=pd.DataFrame(images[4], columns=['image'])
        test_viral_resized['label'] = 'viral'
        test_viral_resized['train'] = 0
        test_viral_resized['test'] = 1
        test_viral_resized['gs_sum'] = self.sums_test_viral
        
        test_normal_resized=pd.DataFrame(images[5], columns=['image'])
        test_normal_resized['label'] = 'normal'
        test_normal_resized['train'] = 0
        test_normal_resized['test'] = 1
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
        self.binary_train_images, self.binary_train_labels = next(binary_train_gen)
#         self.binary_train_images = binary_train_images.reshape(binary_train_images.shape[0], -1)
#         self.binary_train_labels = np.reshape(binary_train_labels[:], (5232,1))
        # BINARY TEST
        self.binary_test_images, self.binary_test_labels = next(binary_test_gen)
#         self.binary_test_images = binary_test_images.reshape(binary_test_images.shape[0], -1)
#         self.binary_test_labels = np.reshape(binary_test_labels[:], (624,1))
        
        # TERNARY TRAIN
        self.ternary_train_images, self.ternary_train_labels = next(ternary_train_gen)
#         self.ternary_train_images = ternary_train_images.reshape(ternary_train_images.shape[0], -1)
#         self.ternary_train_labels = ternary_train_labels[:,0].reshape(-1,1)
        # TERNARY TEST
        self.ternary_test_images, self.ternary_test_labels = next(ternary_test_gen)
#         self.ternary_test_images = ternary_test_images.reshape(ternary_test_images.shape[0], -1)
#         self.ternary_test_labels = ternary_test_labels[:,0].reshape(-1,1)
                                                      
        print('Data is ready for modeling.\n\nYou can check out the preprocessed data with the following attributes: \n\n.binary_test_images\n.binary_train_images\n.binary_train_labels\n.ternary_train_images\n.ternary_test_images\n.ternary_train_labels\netc.')                                                
    def show_class_distribution(self):
        '''Uses df_ attribute to graph class distribution for train and test data.'''
        grouped = self.df_[['train', 'test', 'label']].groupby('label').sum()
        
        fig, ax = plt.subplots(figsize=(10,6))

        bottom_y = [grouped.loc['bacterial']['train'].sum(), grouped.loc['bacterial']['test'].sum()]
        middle_y = [grouped.loc['viral']['train'].sum(), grouped.loc['viral']['test'].sum()]
        top_y = [grouped.loc['normal']['train'].sum(), grouped.loc['normal']['test'].sum()]
        stacked = [(bottom_y[0]+middle_y[0]), (bottom_y[1]+middle_y[1])]

        labels = ['Train', 'Test']

        ax.bar(x=labels, height=bottom_y, label='Bacterial Pnuemonia', color='tab:green')
        ax.bar(x=labels, height=middle_y, bottom=bottom_y, label='Viral Pneumonia', color='tab:olive')
        ax.bar(x=labels, height=top_y, bottom=stacked, label='Normal (No Pneumonia)')

        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        ax.set_title('Class Distribution: Chest X-ray Image Classification', size=20)
        ax.set_ylabel('Number of Images', size=15)

        ax.legend()
    
    def grayscale_sum_dist(self):
        
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(22,8), sharey=True)

        sns.histplot(self.df_[self.df_['label']=='normal']['gs_sum'], alpha=0.9, label='Normal', ax=ax1)
        sns.histplot(self.df_[self.df_['label']!='normal']['gs_sum'], color='forestgreen', ax=ax1, alpha=0.3, label='Pneumonia')
        ax1.axvline(x=self.df_[self.df_['label']=='normal']['gs_sum'].mean(), ymin=0, ymax=300, lw=3.5, label='Normal GS-Sum Mean', color='navy')
        ax1.axvline(x=self.df_[self.df_['label']!='normal']['gs_sum'].mean(), ymin=0, ymax=300, lw=3.5, label='Pneumonia GS-Sum Mean', color='limegreen')
        ax1.set_title('Binary Classification', size=20)
        ax1.set_xlabel('GrayScale Sum', size=15)
        ax1.set_ylabel('Number of Images', size=15)
        ax1.legend(prop={"size":15})

        sns.histplot(self.df_[self.df_['label']=='normal']['gs_sum'], label='Normal', ax=ax2)
        sns.histplot(self.df_[self.df_['label']=='bacterial']['gs_sum'], color='tab:green', ax=ax2, alpha=0.3, label='Bacterial')
        sns.histplot(self.df_[self.df_['label']=='viral']['gs_sum'], color='tab:olive', ax=ax2, alpha=0.4, label='Viral')
        ax2.axvline(x=self.df_[self.df_['label']=='normal']['gs_sum'].mean(), ymin=0, ymax=300, lw=3.5, label='Normal GS-Sum Mean', color='navy')
        ax2.axvline(x=self.df_[self.df_['label']=='bacterial']['gs_sum'].mean(), ymin=0, ymax=300, lw=3.5, label='Bacterial GS-Sum Mean', color='lime')
        ax2.axvline(x=self.df_[self.df_['label']=='viral']['gs_sum'].mean(), ymin=0, ymax=300, lw=3.5, label='Viral GS-Sum Mean', color='yellow')
        ax2.set_title('Ternary Classification', size=20)
        ax2.set_xlabel('GrayScale Sum', size=15)
        ax2.set_ylabel('Number of Images', size=15)
        ax2.legend(prop={"size":15})

        fig.suptitle('Distribution of Gray Scale Sums', size=25)
        
    def dark_vs_light(self, graph_number):
        '''Takes in either 1 or 2 (int) to show different comparisons of lightest and darkest images in the dataset'''
        
        if graph_number not in [1,2]:
            print('Expecting one of the numbers in this list: [1,2] for graph_number param.')
        elif graph_number == 1:
            # Darkest vs lightest out of entire dataset
            darkest = self.df_[self.df_['gs_sum']==self.df_['gs_sum'].min()]
            lightest = self.df_[self.df_['gs_sum']==self.df_['gs_sum'].max()]
            
            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15,10), sharey=True)

            ax1.imshow(darkest['image'].values[0], cmap='gray', vmin=0, vmax=255)
            ax1.set_title('Darkest X-ray Chest Scan\nGS-Score = ~2.9 Million\nVIRAL', size=15)
            ax1.set_xlabel('X-axis Pixel Index')
            ax1.set_ylabel('Y-axis Pixel Index')
            ax1.grid(False)

            ax2.imshow(lightest['image'].values[0], cmap='gray', vmin=0, vmax=255)
            ax2.set_title('Lightest X-ray Chest Scan\nGS-Score = ~27.7 Million\nBACTERIAL', size=15)
            ax2.set_xlabel('X-axis Pixel Index')
            ax2.set_ylabel('Y-axis Pixel Index')
            ax2.grid(False)
    
        elif graph_number == 2:
            fig, ax = plt.subplots(ncols=2, nrows=3, figsize=(25,35), 
                       sharey='row', sharex='col')

            labels = ['normal', 'bacterial', 'viral']

            for r in range(3):
                df = self.df_[self.df_['label']==labels[r]]
                minimum = df['gs_sum'].min()
                darkest = df[df['gs_sum']==minimum]

                df = self.df_[self.df_['label']==labels[r]]
                maximum = df['gs_sum'].max()
                lightest = df[df['gs_sum']==maximum]

                graphs = [darkest, lightest]
                scores = [minimum, maximum]
                for c in range(2):
                    d_or_l = None

                    if c == 0:
                        d_or_l = 'Darkest'
                    else:
                        d_or_l = 'Lightest'

                    ax[r][c].imshow(graphs[c]['image'].values[0], cmap='gray', vmin=0, vmax=255)
                    ax[r][c].set_title(f'{d_or_l} X-ray Chest Scan\nGS-Score = {scores[c]}\n{labels[r].capitalize()}', size=15)
                    ax[r][c].grid(False)

            plt.subplots_adjust(left=0.125,
                                bottom=0.1, 
                                right=0.4, 
                                top=0.5, 
                                wspace=0.05, 
                                hspace=0.35)
            
            
            
    def build_model(self, layers, ternary, optimizer, loss, metrics, epochs, batch_size, validation_split):
        '''
        Uses in model-ready dataset attribute, returns None, but stores fit model object in the class. If ternary=True, then builds model that distinguishes bacteria vs viral pneumonia.
        First layer of network must contain input shape.
        
        params:
        --------
        :layers: a list of step functions in desired order, with desired specifications (Ex: [Dense(12, activation='relu', input_shape=(50176,), MaxPooling2D((2, 2)), Dense(1, activation='softmax')])
        :ternary: bool, whether or not you are building a binary/ternary classifier.
        :optimizer: choose one of the following (str): [
        :loss: choose one of the following (str): [
        :metrics: choose from the following (tuple): [
        :epochs: int; number of big-boy rounds.
        :batch_size: int; number of bony cliques.
        :validation_split: float; proportion of training data to be siphoned off to use for validation.
        '''
        self.model = Sequential()
        
        for layer in layers:
            model.add(layer)
        
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        if ternary == False:
            self.history = model.fit(self.binary_train_images,
                                     self.binary_train_labels,
                                     epochs = epochs,
                                     batch_size = batch_size,
                                     validation_split = validation_split)
                                     
        elif ternary == True:
            self.history = model.fit(self.ternary_train_images,
                                     self.ternary_train_labels,
                                     epochs = epochs,
                                     batch_size = batch_size,
                                     validation_split = validation_split)
        else:
            print("Must enter either bool depending on desired classifier: binary or ternary.")

    def get_results(self, graph_name, model_name):
        '''
        Takes in model and returns confusion matrix, accuracy, summary table; diagnostics can be chosen, but by default all are returned. If user does not want to wait forever for a model to build, if a param is set to True, will return summary of previously built model. Also should have ability to return graph of loss and accuracy/recall growth across epochs. Don't know if this will have to be segmented via attributes.
        
        Params:
        ---------
        :graph_name: str - one of the following - 'acc_recall', 'confusion_matrix', 'loss_roc'
        :model_name: str - name of model (Ex: 'FSM')
        '''
        if graph_name == 'acc_recall':
            model_epochs = self.history.epoch
            model_recall_train = self.history.history['recall']
            model_recall_val = self.history.history['val_recall']
            model_accuracy_train = self.history.history['accuracy']
            model_accuracy_val = self.history.history['val_accuracy']

            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize = (13,6))

            ax1.plot(model_epochs, model_accuracy_train, label = 'Training Data', lw=3)
            ax1.plot(model_epochs, model_accuracy_val, linestyle = '--', label = 'Validation Data', lw=3)
            ax1.set_xlabel("Epoch", size=15)
            ax1.set_ylabel("Accuracy", size=15)
            ax1.set_title('Accuracy\n10 Epochs')
            ax1.legend();

            ax2.plot(model_epochs, model_recall_train, label = 'Training Data', lw=3)
            ax2.plot(model_epochs, model_recall_val, linestyle = '--', label = 'Validation Data', lw=3)
            ax2.set_xlabel("Epoch", size=15)
            ax2.set_ylabel("Recall", size=15)
            ax2.set_title('Recall\n10 Epochs')
            ax2.legend();

            plt.suptitle(f'{model_name} Diagnostics', size=25)
        
        elif graph_name == 'loss_roc':
            self.predictions = cnn.model.predict(cnn.binary_test_images).ravel()
            y_true = cnn.binary_test_labels
            fpr, tpr, thresholds = roc_curve(y_true, y_pred)
            auc = auc(fpr, tpr)
            
            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(13,6))

            model_epochs = cnn.history.epoch
            train_loss = cnn.history.history['loss']
            val_loss = cnn.history.history['val_loss']

            ax1.plot(model_epochs, train_loss, label = 'Training Data', lw=3)
            ax1.plot(model_epochs, val_loss, linestyle = '--', label = 'Validation Data', lw=3)
            ax1.set_title('Loss', size=15)
            ax1.set_xlabel('Epoch', size=15)
            ax1.set_ylabel('Loss', size=15)
            ax1.legend(prop={"size":15})

            ax2.plot(fpr, tpr, label = f'Training Data: AUC = {auc}', lw=3)
            ax2.set_title('ROC Curve', size=15)
            ax2.set_xlabel('False Positive Rate', size=15)
            ax2.set_ylabel('False Negative Rate', size=15)
            ax2.legend(prop={"size":15})

            plt.suptitle(f'{model_name} Diagnostics', size=25)

    def tensorboard(self):
        '''
        Takes in _____ and launches Tensorboard interface AND/OR returns images taken for previously built model if user does not want to launch interface.
        '''
        return None

