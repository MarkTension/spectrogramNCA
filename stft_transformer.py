import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from utils import AttributeDict
from pydub import AudioSegment
from pydub.playback import play
import imageio
from sklearn.preprocessing import MinMaxScaler
from pickle import dump
import os

class StftTransformer:
    """
    class is responsible to convert an audio to and from complex coordinates / complex numbers, 
    and visualizing the file with an audio spectrogram
    """
    def __init__(self, n_fft, rate, audio_array, paths:AttributeDict):
        self.experiment_path = paths.experiment     # root of our experiment
        self.paths = paths
        self.rate = rate
        self.hop_length = n_fft // 4                # hop length standard value
        self.audio_out = None                       # for reconstructed audio
        self.scaler_real, self.scaler_imag = None   # for scaling the complex numbers

        # converts audio to complex numbers
        self.complex_coords, self.amplitudes = self._audio_to_complex(audio_array)
        self._plot_spectrogram(paths.spectrogram)


    def _audio_to_complex(self, audio_array: np.array):
        """sets the complex coordinates and amplitudes instance variable from the audio array"""

        complex_coords = librosa.stft(
            audio_array, hop_length=self.hop_length, window='hann', center=True)  # n_fft=2048,
        amplitudes = np.abs(self.complex_coords)
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
                
        # make reversible scaler
        self.scaler_real = MinMaxScaler()
        self.scaler_imag = MinMaxScaler()
        
        self.scaler_real.fit(real)
        self.scaler_imag.fit(imaginary)

        real_scaled = self.scaler_real.transform(real)
        imag_scaled = self.scaler_imag.transform(imaginary)

        # save scalers for inverse later
        dump(self.scaler_real, os.path.join(self.experiment_path, "scaler_real.pkl"))
        dump(self.scaler_imag, os.path.join(self.experiment_path, "scaler_imag.pkl"))

        return real_scaled, imag_scaled
        


    def complex_to_png(self):
        """
        saves complex coordinates to png image
        """
        assert str(type(self.complex_coords)) == "<class 'numpy.ndarray'>"

        # put the real and imaginary numbers into separate dimensions
        real = np.real(self.complex_coords)
        imaginary = np.imag(self.complex_coords)
        real_scaled, imag_scaled = self._scale_complex(real, imaginary)
        realcomplex = np.stack([real_scaled, imag_scaled, np.ones(real.shape, dtype=np.float)], axis=2)

        # realcomplex = np.abs(realcomplex* 255)
        imageio.imwrite(uri=self.paths.complex_coords, im=realcomplex)
        imageio.imwrite(uri=os.path.join(self.experiment_path, "real_scaled.png"), im=real_scaled)
        imageio.imwrite(uri=os.path.join(self.experiment_path, "imag_scaled.png"), im=imag_scaled)
        print(f"saved the png files to {self.experiment_path}")


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
