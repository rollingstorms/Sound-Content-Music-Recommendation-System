import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyarrow import feather
import time
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler
from src.DataGenerator import AudioDataGenerator
from src.helper_functions import progress_bar


class LatentSpace:
    
    def __init__(self, autoencoder_path, image_dir, tracks_feather_path, sample_size=None, latent_dims=128, num_channels=1, output_size=(128,128)):
        self.batch_size = 32
        self.autoencoder = tf.keras.models.load_model(autoencoder_path)
        self.prediction_generator = AudioDataGenerator(directory=image_dir,
                                    image_size=(128,512),
                                    color_mode='rgb',
                                    batch_size=self.batch_size, 
                                    shuffle=False,
                                    sample_size = sample_size,
                                    output_channel_index=0,
                                    num_output_channels=num_channels,
                                    output_size=output_size)
        self.latent_cols = [f'latent_{i}' for i in range(latent_dims)]
        
        self._tracks_df_path = tracks_feather_path
        self.size = self.prediction_generator.size
        self._num_channels = num_channels
        
    def build(self):
        self.prediction_generator.batch_size=self.batch_size
        results = []
        print('Getting predictions from autoencoder...')
        start_time = time.time()
        search_range = self.prediction_generator.size // self.batch_size
        for i in range(search_range):
            filename, latent_img, _ = self.prediction_generator.take(i, return_filename=True, get_all_tiles=True)
            latent_space = np.array(self.autoencoder.encoder(latent_img))

            for j in range(len(latent_space)):

                result={
                    'id':str(filename[j]).split('.')[0],
                    'filename':str(filename[j]),
                      }
                for idx, col in enumerate(latent_space[j]):
                    result[f'latent_{idx}'] = col

                results.append(result)

            progress_bar(i+1, search_range)

        print('\n')
        print(round((time.time()-start_time)/60, 2),'minutes elapsed')
        
        start_time = time.time()
        print('Building tracks dataframe...')
        results_df = pd.DataFrame(results)
        self.results = results_df
        print('size of results', len(results_df))
        results_df = results_df.groupby('id').mean().reset_index()
        tracks_df = feather.read_feather(self._tracks_df_path)
        track_latents = results_df.merge(tracks_df, how='left', left_on='id', right_on='track_id')
        track_latents = track_latents.drop_duplicates(subset='id')
        track_latents = track_latents.reset_index(drop=True)
        
        scaler = StandardScaler()
        track_latent_scaled = scaler.fit_transform(track_latents[self.latent_cols])
        track_latents[self.latent_cols] = track_latent_scaled
        
        print(f'Track dataframe built. {round((time.time()-start_time)/60,2)} minutes elapsed')
        
        start_time = time.time()
        print('Building artist distributions...')
        artist_latents = track_latents.groupby('artist_name').mean().dropna()
        artist_latents.drop(columns=['track_popularity', 'artist_popularity'], inplace=True)
        print(f'Artist distributions built. {round((time.time()-start_time)/60,2)} minutes elapsed')
        
        start_time = time.time()
        print('Building genre distributions...')
        genre_rows = []
        for idx, row in track_latents.iterrows():
            for genre in row.artist_genres:
                new_row = row
                new_row['genre'] = genre
                genre_rows.append(new_row)        
        genre_latents = pd.DataFrame(genre_rows)
        genre_latents = genre_latents.groupby('genre').mean().dropna()
        genre_latents.drop(columns=['track_popularity', 'artist_popularity'], inplace=True)
        print(f'Genre distributions built. {round((time.time()-start_time)/60,2)} minutes elapsed')
        
        self.tracks = track_latents
        self.artists = artist_latents
        self.genres = genre_latents
        
        print('Latent Space Built.')
        
        self.prediction_generator.batch_size = 1
        
    def save(self, directory_to_save):
        feather.write_feather(self.tracks, directory_to_save+'/tracks.feather')
        feather.write_feather(self.artists, directory_to_save+'/artists.feather')
        feather.write_feather(self.genres, directory_to_save+'/genres.feather')
    
    def load(self, directory_to_load):
        self.tracks = feather.read_feather(directory_to_load+'/tracks.feather')
        self.artists = feather.read_feather(directory_to_load+'/artists.feather')
        self.genres = feather.read_feather(directory_to_load+'/genres.feather')
        self.prediction_generator.batch_size = 1
        
    def get_similar_tracks_by_index(self, df_index, num=10, similarity_measure='cosine'):

        track_similarity_df = self._get_similarity(self.tracks.iloc[[df_index]], self.tracks, subset=self.latent_cols, num=num, similarity_measure=similarity_measure)

        return track_similarity_df[['index','track_name','artist_name', 'track_uri','similarity']]
    
    def get_similar_artists_by_index(self, df_index, num=10, similarity_measure='cosine'):

        artist_similarity_df = self._get_similarity(self.tracks.iloc[[df_index]], self.artists, subset=self.latent_cols, num=num, similarity_measure=similarity_measure)

        return artist_similarity_df[['artist_name','similarity']]

    def get_similar_genres_by_index(self, df_index, num=10, similarity_measure='cosine'):

        genre_similarity_df = self._get_similarity(self.tracks.iloc[[df_index]], self.genres, subset=self.latent_cols, num=num, similarity_measure=similarity_measure)

        return genre_similarity_df[['genre','similarity']]

    def _get_similarity(self, df1, df2, subset, num=10, similarity_measure='cosine'):
        if similarity_measure == 'cosine':
            similarity_measure_fn = cosine_similarity
            sort = False
        elif similarity_measure == 'euclidean':
            similarity_measure_fn = euclidean_distances
            sort = True
        else:
            raise ValueError('similarity_measure must be "cosine" or "euclidean"')

        similarity = similarity_measure_fn(np.array(df1[subset]), np.array(df2[subset]))

        similarity_df = df2.copy()
        similarity_df['similarity'] = similarity.T

        return similarity_df.sort_values(by='similarity', ascending=sort).reset_index()[:num]


    def get_image_data_by_index(self, index):
        if index < self.size:
            return self.prediction_generator.take(index)[0]
        else:
            raise ValueError(f'Index must be less than total size of generator: {self.size}')
            
    def plot_reconstruction(self, index):
        test_img = self.get_image_data_by_index(index)
        prediction = np.array(self.autoencoder(test_img))
        
        fig, ax = plt.subplots(ncols=2, figsize=(10,3))
        ax[0].title.set_text('Original image')
        ax[0].imshow(test_img[0])
        ax[1].title.set_text('Reconstructed image')
        ax[1].imshow(prediction[0])
        plt.tight_layout()
        plt.show()

        if self._num_channels==3:
                fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(10,5))
                ax[0][0].title.set_text('Original Mel Spectogram')
                ax[0][0].imshow(np.array(test_img[0][:,:,0]), cmap='Reds')
                ax[1][0].title.set_text('Original MFCC')
                ax[1][0].imshow(np.array(test_img[0][:,:,1]), cmap='Greens')
                ax[2][0].title.set_text('Original Chromagram')
                ax[2][0].imshow(np.array(test_img[0][:,:,2]), cmap='Blues')
                ax[0][1].title.set_text('Reconstructed Mel Spectogram')
                ax[0][1].imshow(np.array(prediction[0][:,:,0]), cmap='Reds')
                ax[1][1].title.set_text('Reconstructed MFCC')
                ax[1][1].imshow(np.array(prediction[0][:,:,1]), cmap='Greens')
                ax[2][1].title.set_text('Reconstructed Chromagram')
                ax[2][1].imshow(np.array(prediction[0][:,:,2]), cmap='Blues')
                plt.tight_layout()
                plt.show()