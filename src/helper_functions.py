import matplotlib.pyplot as plt

def progress_bar(progress, total, display_length=60):
        left_ratio = display_length * progress//total
        right_ratio = display_length - left_ratio
        
        print('['+ '='*left_ratio + '>' + '.'*right_ratio + f'] {progress} / {total}', end='\r') 

def plot_reconstruction(image, prediction, num_channels):
        
        
        fig, ax = plt.subplots(ncols=3, figsize=(10,3))
        ax[0].title.set_text('Original image')
        ax[0].imshow(image[0])
        ax[1].title.set_text('Reconstructed image')
        ax[1].imshow(prediction[0])
        ax[2].title.set_text('Difference')
        ax[2].imshow(prediction[0] - image[0], cmap='Spectral')
        plt.tight_layout()
        plt.show()

        if num_channels==3:
                fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(10,5))
                ax[0][0].title.set_text('Original Mel Spectogram')
                ax[0][0].imshow(np.array(image[0][:,:,0]), cmap='Reds')
                ax[1][0].title.set_text('Original MFCC')
                ax[1][0].imshow(np.array(image[0][:,:,1]), cmap='Greens')
                ax[2][0].title.set_text('Original Chromagram')
                ax[2][0].imshow(np.array(image[0][:,:,2]), cmap='Blues')
                ax[0][1].title.set_text('Reconstructed Mel Spectogram')
                ax[0][1].imshow(np.array(prediction[0][:,:,0]), cmap='Reds')
                ax[1][1].title.set_text('Reconstructed MFCC')
                ax[1][1].imshow(np.array(prediction[0][:,:,1]), cmap='Greens')
                ax[2][1].title.set_text('Reconstructed Chromagram')
                ax[2][1].imshow(np.array(prediction[0][:,:,2]), cmap='Blues')
                plt.tight_layout()
                plt.show()