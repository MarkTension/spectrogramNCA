import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from utils import AttributeDict
from pydub import AudioSegment
from pydub.playback import play
import imageio
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pickle import dump
import os

class StftTransformer:
    """
    class is responsible to convert an audio to and from complex coordinates / complex numbers, 
    and visualizing the file with an audio spectrogram
    """
    def __init__(self, n_fft, rate, audio_array, paths:AttributeDict, freq_bin_cutoff:int=None):
        self.experiment_path = paths.experiment     # root of our experiment
        self.paths = paths
        self.rate = rate
        self.hop_length = n_fft // 4                # hop length standard value
        self.audio_out = None                       # for reconstructed audio
        # self.scaler_real, self.scaler_imag = None, None
        self.scalers = {}
        self.minimums = 0 # the min val of the complex numbers
        self.complex_coords_new = None
        self.cutoff_residue = None
        self.freq_bin_cutoff = freq_bin_cutoff

        # converts audio to complex numbers
        self.complex_coords, self.amplitudes = self._audio_to_complex(audio_array)
        self._plot_spectrogram(paths.spectrogram)


    def _audio_to_complex(self, audio_array: np.array):
        """sets the complex coordinates and amplitudes instance variable from the audio array"""

        complex_coords = librosa.stft(
            audio_array, hop_length=self.hop_length, window='hann', center=True)  # n_fft=2048,
        amplitudes = np.abs(complex_coords)
        return complex_coords, amplitudes

    def complex_to_audio(self, outfile_path=None):
        """
        inverse short term fourrier transform for audio file reconstruction from complex coordinates.
        saves audiofile if path is provided
        """

        assert str(type(self.complex_coords)) == "<class 'numpy.ndarray'>"

        audio_array = librosa.istft(
            self.complex_coords, hop_length=self.hop_length, window='hann')  # n_fft=2048,
        if (outfile_path != None):
            sf.write(outfile_path, audio_array, self.rate, subtype='PCM_24')
            print(f"saved the audio file to {outfile_path}")

        # sets the audio out (array) instance variable from the saved file
        self.audio_out = AudioSegment.from_wav(outfile_path)


    def _scale_complex(self, real, imaginary):
        """ scales real and imaginary with sklearn scaler """        
        output = []
        for name, values in [("real", real), ("imag", imaginary)]:

            self.scalers[name] = MinMaxScaler()
            self.scalers[name].fit(values)
            scaled = self.scalers[name].transform(values)
            dump(self.scalers[name], open(os.path.join(self.experiment_path, f"scaler_{name}.pkl"), 'wb'))
            output.append(scaled)

        return output
        

    def _transform_complex(self, complex_cords):
        """ separates real and imaginary, applies log transform and transforms separately with minmax scaler """
        
        complex_cords = np.log(complex_cords)
        # put the real and imaginary numbers into separate dimensions
        real = np.real(complex_cords)
        imaginary = np.imag(complex_cords)

        # self.minimums = -np.min(np.stack([real, imaginary])) + 0.00001
        # real += self.minimums
        # imaginary += self.minimums

        # log transform
        # real = np.log(real)
        # imaginary = np.log(imaginary)
        # scale with minmax
        real_scaled, imag_scaled = self._scale_complex(real, imaginary)

        return real_scaled, imag_scaled


    def inverse_convert_complex(self, complex_converted:np.array):
        """ input is 1. log transformed, 2. minmax-scaled and saved. 
        inverts back to complex numbers, ready for inverse-stft

        Args:
            complex_converted (np.array): minmax-scaled and log transformed 3-channel spectrogram

        Returns:
            complex_values (np.array): complex numbers that are ready for inverse-stft
        """
        # TODO: load scalers if not present
        # invert the minmax
        real = self.scalers['real'].inverse_transform(complex_converted[:,:,0])
        imaginary = self.scalers['imag'].inverse_transform(complex_converted[:,:,1])
        # put into complex numbers
        complex_numbers = real + 1j*imaginary
        complex_numbers = np.exp(complex_numbers)

        return complex_numbers


    def truncate_length(self, complex_coords):
        
        print(f"cutting off frequency bins. Shape before is {complex_coords.shape}")

        self.cutoff_residue = complex_coords[self.freq_bin_cutoff:]
        complex_new = complex_coords[:self.freq_bin_cutoff]
        print(f"shape after is {complex_new.shape}")

        return complex_new


    def convert_complex_for_nca(self):
        """
        saves complex coordinates to png image
        1. separate real and fake
        2. scale both independently, and stack
        3. write to png file
        """
        assert str(type(self.complex_coords)) == "<class 'numpy.ndarray'>"

        real_transf, imag_transf = self._transform_complex(self.complex_coords)
        # stack to save it as an image
        self.complex_transf = np.stack([real_transf, imag_transf, np.ones(real_transf.shape, dtype=np.float)], axis=2)

        # cutoff a set of frequencies to add again later
        if (self.freq_bin_cutoff != None):
            self.complex_transf = self.truncate_length(self.complex_transf)

        # write to file
        imageio.imwrite(uri=self.paths.complex_coords, im=self.complex_transf)
        imageio.imwrite(uri=os.path.join(self.experiment_path, "real_scaled.png"), im=real_transf)
        imageio.imwrite(uri=os.path.join(self.experiment_path, "imag_scaled.png"), im=imag_transf)
        print(f"saved the png files to {self.experiment_path}")
        return self.complex_transf

    def _plot_spectrogram(self, spectrogram_path):
        """ saves spectrogram image """

        fig, ax = plt.subplots()
        img = librosa.display.specshow(librosa.amplitude_to_db(self.amplitudes,
                                                               ref=np.max),
                                       y_axis='log', x_axis='time', ax=ax)
        ax.set_title('Power spectrogram')
        # fig.colorbar(img, ax=ax, format="%+2.0f dB")
        fig.subplots_adjust(bottom=0)
        fig.subplots_adjust(top=1)
        fig.subplots_adjust(right=1)
        fig.subplots_adjust(left=0)
        # fig.show()
        fig.savefig(spectrogram_path)


    def play_sounds(self):
        assert self.audio_out != None
        play(self.audio_out)
