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
from joblib import dump, load
import json
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials



class LatentSpace:
    
    def __init__(self,
                autoencoder_path,
                image_dir,
                tracks_feather_path,
                sample_size=None,
                latent_dims=128,
                num_channels=1,
                output_size=(128,128),
                scale=True,
                threshold_level=0,
                num_tiles=4):
        self._batch_size = 1
        self.autoencoder = tf.keras.models.load_model(autoencoder_path)
        self.prediction_generator = AudioDataGenerator(directory=image_dir,
                                    image_size=(128,512),
                                    color_mode='rgb',
                                    batch_size=1, 
                                    shuffle=False,
                                    sample_size = sample_size,
                                    output_channel_index=0,
                                    num_output_channels=num_channels,
                                    output_size=output_size,
                                    threshold_level=threshold_level)
        self.latent_cols = [f'latent_{i}' for i in range(latent_dims)]
        
        self._tracks_df_path = tracks_feather_path
        self.size = self.prediction_generator.size
        self._num_channels = num_channels
        self._scale = scale
        self._num_tiles = num_tiles

        f = open('data/apikeys/.apikeys.json')
        apikeys = json.load(f)
        self.client_id = apikeys['clientId']
        self.client_secret = apikeys['clientSecret']

        credentials_manager = SpotifyClientCredentials(client_id=self.client_id, client_secret=self.client_secret)

        self._spotify = spotipy.Spotify(client_credentials_manager=credentials_manager)
        
    def build(self):
        results = []
        print('Getting predictions from autoencoder...')
        start_time = time.time()
        search_range = self.prediction_generator.size
        for i in range(search_range):
            filename, latent_img, _ = self.prediction_generator.take(i, return_filename=True, get_all_tiles=True, num_tiles=self._num_tiles)
            latent_space = np.array(self.autoencoder.encoder(latent_img)).mean(axis=0)

            for j in range(self._batch_size ):

                result={
                    'id':filename[j].split('.')[0],
                    'filename':filename[j],
                      }
                for idx, col in enumerate(latent_space):
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

        tracks_df = feather.read_feather(self._tracks_df_path)
        track_latents = results_df.merge(tracks_df, how='left', left_on='id', right_on='track_id')
        track_latents = track_latents.drop_duplicates(subset='id')
        track_latents = track_latents.reset_index(drop=True)
        
        if self._scale:
            self._scaler = StandardScaler()
            track_latent_scaled = self._scaler.fit_transform(track_latents[self.latent_cols])
            track_latents[self.latent_cols] = track_latent_scaled
        self.tracks = track_latents
        print(f'Track dataframe built. {round((time.time()-start_time)/60,2)} minutes elapsed')
        
        start_time = time.time()
        print('Building artist distributions...')
        artist_latents = track_latents.groupby(['artist_id','artist_name']).mean().reset_index()
        artist_latents.drop(columns=['track_popularity'], inplace=True)
        self.artists = artist_latents
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
        genre_latents = genre_latents.reset_index()
        print(f'Genre distributions built. {round((time.time()-start_time)/60,2)} minutes elapsed')
        
        self.genres = genre_latents
        
        print('Latent Space Built.')


    def save(self, directory_to_save, save_full_results=False):

        directories = directory_to_save.split('/')
        save_folders=[directories[0]]
        for directory in directories[1:]:
            save_folders.append(save_folders[-1] + '/' + directory)
        for folder in save_folders:
            try:
                os.mkdir(folder)
            except:
                pass

        feather.write_feather(self.tracks, directory_to_save+'/tracks.feather')
        feather.write_feather(self.artists, directory_to_save+'/artists.feather')
        feather.write_feather(self.genres, directory_to_save+'/genres.feather')
        if self._scale:
            dump(self._scaler, directory_to_save+'/std_scaler.bin', compress=True)

        if save_full_results:
            feather.write_feather(self.results, directory_to_save+'/results.feather')

    def load(self, directory_to_load, load_full_results=False):
        try:
            self.tracks = feather.read_feather(directory_to_load+'/tracks.feather')
            print('Loaded tracks.')
        except:
            print('Failed to load tracks.')

        try:
            self.artists = feather.read_feather(directory_to_load+'/artists.feather')
            print('Loaded artists.')
        except:
            print('Failed to load artists.')

        try:
            self.genres = feather.read_feather(directory_to_load+'/genres.feather')
            print('Loaded genres.')
        except:
            print('Failed to load genres.')

        try:
            self._scaler=load(directory_to_load+'/std_scaler.bin')
            print('loaded scaler')
        except:
            print('Failed to load scaler.')

        if load_full_results:
            try:
                self.genres = feather.read_feather(directory_to_load+'/results.feather')
                print('Loaded full results.')
            except:
                print('Failed to load full results')
        
        
        self.prediction_generator.batch_size = 1
        
    def get_similar_tracks_by_index(self, df_index, num=10, similarity_measure='cosine', scope='all'):

        if scope == 'all':
            track_similarity_df = self._get_similarity(self.tracks.iloc[[df_index]], self.tracks, subset=self.latent_cols, num=num, similarity_measure=similarity_measure)
        elif scope == 'similar_artists':
            similar_artist_tracks = pd.DataFrame()
            for artist_id in self.get_similar_artists_by_index(df_index, num=100).artist_id:
                similar_artist_tracks = pd.concat([similar_artist_tracks, self.get_index_by_artist_id(artist_id)])
            similar_artist_tracks = self.tracks[self.tracks.track_id.isin(similar_artist_tracks.track_id)]
            
            track_similarity_df = self._get_similarity(self.tracks.iloc[[df_index]], similar_artist_tracks, subset=self.latent_cols, num=num, similarity_measure=similarity_measure)
        else:
            raise ValueError('scope must be "all" or "similar_artists"')
        
        return track_similarity_df[['index','track_name','artist_name', 'track_uri','similarity']]

    
    def get_similar_artists_by_index(self, df_index, num=10, similarity_measure='cosine'):

        artist_similarity_df = self._get_similarity(self.tracks.iloc[[df_index]], self.artists, subset=self.latent_cols, num=num, similarity_measure=similarity_measure)

        return artist_similarity_df[['artist_id','artist_name','similarity']]

    def get_similar_genres_by_index(self, df_index, num=10, similarity_measure='cosine'):

        genre_similarity_df = self._get_similarity(self.tracks.iloc[[df_index]], self.genres, subset=self.latent_cols, num=num, similarity_measure=similarity_measure)

        return genre_similarity_df[['genre','similarity']]


    def get_similarity(self, df1, df2, subset, num=10, similarity_measure='cosine', popularity_threshold=0, sort_tracks=True):
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

        similarity_df = similarity_df[similarity_df.track_popularity > popularity_threshold]

        if sort_tracks:
            similarity_df = similarity_df.sort_values(by='similarity', ascending=sort).reset_index()

        return similarity_df[:num]


    def get_image_data_by_index(self, index, get_all_tiles=False, num_tiles=4):
        if index < self.size:
            return self.prediction_generator.take(index, get_all_tiles=get_all_tiles, num_tiles=num_tiles)[0]
        else:
            raise ValueError(f'Index must be less than total size of generator: {self.size}')
            
    def plot_reconstruction(self, index):
        test_img = self.get_image_data_by_index(index)
        prediction = np.array(self.autoencoder(test_img))
        
        fig, ax = plt.subplots(ncols=3, figsize=(15,5))
        ax[0].title.set_text('Original image')
        ax[0].imshow(test_img[0])
        ax[1].title.set_text('Reconstructed image')
        ax[1].imshow(prediction[0])
        ax[2].title.set_text('Difference')
        ax[2].imshow(prediction[0] - test_img[0], cmap="Spectral")
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

    def get_index_by_artist_name(self, artist_name):
        return self.tracks[self.tracks.artist_name.str.contains(artist_name)][['track_id', 'track_name', 'artist_name']]

    def get_index_by_artist_id(self, artist_id):
        return self.tracks[self.tracks.artist_id == artist_id][['track_id', 'track_name', 'artist_name']]

    def add_artist_id_to_artists(self):
        self.artists = self.tracks.groupby(['artist_id','artist_name']).mean().reset_index()

    def get_vector_by_name(self, name, scope):
        if scope == 'artist':
            return self.artists[self.artists.artist_name == name][self.latent_cols]
        if scope == 'genre':
            return self.genres[self.genres.genre == name][self.latent_cols]

    def get_vector_from_preview_link(self, link, track_id):
        img = self.prediction_generator.get_vector_from_preview_link(link, track_id, num_tiles=self._num_tiles)
        vector = np.array(self.autoencoder.encoder(img[0])).mean(axis=0)
        vector = self._scaler.transform(pd.DataFrame([vector], columns=self.latent_cols))
        vector = pd.DataFrame(vector, columns=self.latent_cols)
        return vector

    def plot_reconstruction_from_vector(self, vector):

        vector = self._scaler.inverse_transform(vector)
        
        prediction = np.array(self.autoencoder.decoder(np.array(vector)))
        
        fig, ax = plt.subplots(ncols=1, figsize=(3,3))
        
        ax.title.set_text('Reconstructed image')
        ax.imshow(prediction[0])
        plt.tight_layout()
        plt.show()

    def search_for_recommendations(self, query, num=10, popularity_threshold=10, get_time_and_freq=False):
        id_ = self._spotify.search(query, type='track')['tracks']['items'][0]['id']
        track = self._spotify.track(id_)
        link = track['preview_url']
        print(track['name'])
        print(track['artists'][0]['name'])
        print(link)

        if link is not None:

            vector = self.get_vector_from_preview_link(link, id_)
            similarity = self.get_similarity(vector, self.tracks, subset=self.latent_cols, num=num, popularity_threshold=popularity_threshold)
            if get_time_and_freq:
                similarity['time_similarity'] = self.get_similarity(vector, similarity, subset=self.latent_cols[:len(self.latent_cols)//2], num=num, popularity_threshold=popularity_threshold, sort_tracks=False)['similarity']
                similarity['frequency_similarity'] = self.get_similarity(vector, similarity, subset=self.latent_cols[len(self.latent_cols)//2:], num=num, popularity_threshold=popularity_threshold, sort_tracks=False)['similarity']
            
                return similarity[['track_name','track_uri','artist_name','similarity','track_popularity','time_similarity','frequency_similarity']]
            else:
                return similarity[['track_name','track_uri','artist_name','similarity','track_popularity']]
        else:
            print('No Preview Available. Try a different search.')