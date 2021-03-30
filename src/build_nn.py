# package that can take in a list of directory paths, and can perform number of preprocessing and modeling steps, including building and evaluating a neural network.

class NeuralNet():

    def __init__():
        self.train = train
	self.test = test
	self.val = val

    def get_images():
        '''
        Takes in ___ and returns list of images files to use as visuals 
        '''
        return None

    def img_to_array():
        '''
        Takes in ___ and returns a converted list of image arrays.
        '''
        return None

    def augment_data():
        '''
        Takes in images and returns a dataset with balanced classes (balance ratio can be adjusted as a parameter). Flips/rotates/zooms out data as specified.
        '''
        return None


    def get_model_data():
        '''
        Takes in _____ and returns data in a shape that is suitable for tensorflow. Must specify ternary parameter to adjust the assignment of classes.
        '''
        return None

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

# Class EDA inherits values from the Neural Net class to build graphs/analyze/visualize data.

class EDA(NeuralNet):

    def __init__():
        '''Have to figure out how inheritance works again '''

    def class_distribution():
        '''
        Takes in dataset and returns graph showing class imbalance for ternary and binary versions of the data.
        '''
        return None

    def grayscale_sum_dist():
        '''
        Takes in image data (in array form) and returns histogram of "blackness" levels in normal vs pneumonia"; also returns specific images from key parts of distribution to use as examples (perhaps stored as attributes??)
        '''
        return None


