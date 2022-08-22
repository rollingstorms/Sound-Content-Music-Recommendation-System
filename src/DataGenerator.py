import numpy as np
from random import sample, shuffle
import os
import tensorflow as tf
import numpy.ma as ma


class AudioDataGenerator(tf.keras.utils.Sequence):
    def __init__(self,
                directory,
                image_size,
                color_mode = 'grayscale',
                batch_size=32, 
                shuffle=False,
                sample_size=None, 
                train_test_split=False,
                test_size=.2, 
                file_list=None,
                name='Generator',
                output_channel_index=None,
                num_output_channels=1,
                output_size=None,
                threshold_level=0,
                ):
        self.dir = directory
        self._image_size = image_size
        self._img_height = image_size[0]
        self._img_width = image_size[1]
        self._color_mode = color_mode
        if color_mode == 'grayscale':
            self._img_channels = 1
        elif color_mode == 'rgb':
            self._img_channels = 3

        self.output_channel_index = output_channel_index
        self.num_output_channels = num_output_channels
        self.output_size = output_size

        self.shuffle = shuffle
        self.sample_size = sample_size
        self.test_size = test_size
        self.batch_size = batch_size

        self.threshold_level = threshold_level
        
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

    def __getitem__(self, index, return_filename=False, get_all_tiles=False, num_tiles=4):
        batch = self._files[index*self.batch_size:index*self.batch_size+self.batch_size]
        
        X, y = self.__get_data(batch)
        
        if self.output_channel_index != None:
            X = X[:,:,:,self.output_channel_index:self.output_channel_index+self.num_output_channels]
            y = y[:,:,:,self.output_channel_index:self.output_channel_index+self.num_output_channels]

        if self.output_size != None:
            if get_all_tiles == False:
                if self.output_size[1] < self._img_width:
                    rand_x_index = np.random.randint(low=0, high=self._img_width - self.output_size[1])
                else:
                    rand_x_index = 0
                if self.output_size[0] < self._img_height:
                    rand_y_index = np.random.randint(low=0, high=self._img_height - self.output_size[0])
                else:
                    rand_y_index = 0

                X = X[:,rand_y_index:rand_y_index+self.output_size[0],rand_x_index:rand_x_index+self.output_size[1],:]
                y = X
            else:
                slice_size = (self._img_width - self.output_size[1]) // (num_tiles - 1)

                all_tiles = []
                new_batch = []
                for idx, img in enumerate(X):
                    for i in range(num_tiles):
                        all_tiles.append(img[:,i*slice_size:(i*slice_size)+self.output_size[1],:])
                        new_batch.append(batch[idx])
                            
                X = np.array(all_tiles)
                y = X
                
                if return_filename:
                    batch = new_batch

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
            img = scale*np.array(img)
            if self.threshold_level > 0:
                img_min = img.min()
                img_max = img.max()
                threshold = self.threshold_level * (img_max - img_min) + img_min
                mx = ma.masked_where(img < threshold, img, copy=True)
                img = (mx.filled(fill_value=threshold) - threshold)
                img = img  * (img_max / img.max())
            X[i,] = tf.convert_to_tensor(img)
            
        y = X

        return X, y
    
    def take(self, index=1, return_filename=False, get_all_tiles=False, num_tiles=4):
        
        return self.__getitem__(index, return_filename, get_all_tiles, num_tiles)
    
    def __train_test_split(self, files):
        
        if self.shuffle:
            shuffle(files)
            
        file_list_length = len(files)
        test_split = int(file_list_length * (1 - self.test_size))
        
        train_files = files[:test_split]
        test_files = files[test_split:]
        
        self.train = AudioDataGenerator(self.dir, self._image_size, batch_size=self.batch_size, color_mode=self._color_mode, shuffle=self.shuffle, file_list=train_files, name='Training', output_channel_index=self.output_channel_index, output_size=self.output_size)
        self.test = AudioDataGenerator(self.dir, self._image_size, batch_size=self.batch_size, color_mode=self._color_mode, shuffle=self.shuffle, file_list=test_files, name='Test', output_channel_index=self.output_channel_index, output_size=self.output_size)
        
        
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