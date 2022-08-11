import numpy as np
from random import sample, shuffle
import os
import tensorflow as tf


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, directory, image_size, color_mode = 'grayscale', batch_size=32, shuffle=False, sample_size=None, train_test_split=False, test_size=.2, file_list=None, name='Generator'):
        self.batch_size = batch_size
        self.dir = directory
        self._image_size = image_size
        self._img_height = image_size[0]
        self._img_width = image_size[1]
        self._color_mode = color_mode
        if color_mode == 'grayscale':
            self._img_channels = 1
        elif color_mode == 'rgb':
            self._img_channels = 3

        self.shuffle = shuffle
        self.sample_size = sample_size
        self.test_size = test_size
        if file_list == None:
            self._files = self.__get_images_from_directory(directory)
        else:
            try:
                self._files = self.__collect_image_files(file_list)
            except TypeError:
                print('file_list is not a list')
        
        if train_test_split:
            self.__train_test_split(self._files)
        else:
            print(f'Found {len(self._files)} files for {name} set')
        self.size = len(self._files)  
        self.on_epoch_end()

    def __len__(self):
        return len(self._files) // self.batch_size

    def __getitem__(self, index, return_filename=False):
        batch = self._files[index*self.batch_size:index*self.batch_size+self.batch_size]
        
        X, y = self.__get_data(batch)
        if return_filename:
            return batch, X, y
        else:
            return X, y

    def on_epoch_end(self):
        if self.shuffle:
            shuffle(self._files)
        
        
    def __get_data(self, batch):
        X = np.empty((self.batch_size, self._img_height, self._img_width, self._img_channels))

        for i, file in enumerate(batch):
            path = self.dir + file
            img = tf.keras.preprocessing.image.load_img(path, color_mode=self._color_mode)
            scale = 1./255
            X[i,] = tf.convert_to_tensor(scale*np.array(tf.keras.preprocessing.image.img_to_array(img))) 
            
        y = X

        return X, y
    
    def take(self, index=1, return_filename=False):
        
        return self.__getitem__(index, return_filename)
    
    def __train_test_split(self, files):
        
        if self.shuffle:
            shuffle(files)
            
        file_list_length = len(files)
        test_split = int(file_list_length * (1 - self.test_size))
        
        train_files = files[:test_split]
        test_files = files[test_split:]
        
        self.train = DataGenerator(self.dir, self._image_size, shuffle=self.shuffle, file_list=train_files, name='Training')
        self.test = DataGenerator(self.dir, self._image_size, shuffle=self.shuffle, file_list=test_files, name='Test')
        
        
    def __collect_image_files(self, files):
        filetypes = ['png', 'jpg', 'jpeg', 'webp']
        return [file for file in files if file.split('.')[-1] in filetypes]
    
    def __get_images_from_directory(self, directory):
        files = os.listdir(directory)
        
        files = self.__collect_image_files(files)
        
        if self.shuffle:
            shuffle(files)
        
        if self.sample_size != None:
            files = sample(files, self.sample_size)
        
        return files