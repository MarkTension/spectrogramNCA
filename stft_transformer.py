import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from utils import AttributeDict, plot_spectrogram
from pydub import AudioSegment
import imageio
from sklearn.preprocessing import MinMaxScaler
from pickle import dump
import os
import numpy as np


class Truncater():
    """truncates the complex numbers' frequency bins, and saves the residue for later use"""

    def __init__(self, complex_coords: np.array, truncate_length: int) -> None:
        self._removed = None
        self._truncated = self._truncate(complex_coords, truncate_length)

    def _truncate(self, complex_coords: np.array, truncate_length: int):
        self._removed = complex_coords[truncate_length:]
        return complex_coords[:truncate_length]

    def restore(self, complex_incomplete: np.array):
        return np.concatenate([complex_incomplete, self._removed], axis=0)

    @property
    def truncated(self):
        return self._truncated


class StftTransformer:
    """
    class is responsible to convert an audio to and from complex numbers.
    """

    def __init__(self, n_fft: int, rate: int, audio_array: np.array, paths: AttributeDict, sample_len: int, freq_bin_cutoff: int = None, method='cqt'):
        """ initializes class. Converts audio array to complex values. Plots spectrogram.

        Args:
            n_fft (_type_): _description_
            rate (_type_): _description_
            audio_array (_type_): _description_
            paths (AttributeDict): _description_
            sample_len (int): _description_
            freq_bin_cutoff (int, optional): _description_. Defaults to None.
            method (str, optional): _description_. Defaults to 'cqt'.
        """
        self.experiment_path = paths.experiment     # root of our experiment
        self.paths = paths
        self.rate = rate
        self.hop_length = n_fft // 4                # hop length standard value
        self.audio_out = None                       # for reconstructed audio
        self.scalers = {}                           # minmax scalers for the complex numberes
        self.freq_bin_cutoff = freq_bin_cutoff
        self.truncater = None
        self.transformer = librosa.stft if method == 'stft' else librosa.cqt
        self.inverse_transformer = librosa.istft if method == 'stft' else librosa.icqt
        self.sample_len = sample_len

        # converts audio to complex numbers
        complex_coords, amplitudes = self._audio_to_complex(
            audio_array)
        # converts complex numbers to more ML-friendly format
        self.complex_coords = self._convert_complex(complex_coords)
        plot_spectrogram(paths.spectrogram, amplitudes)

    def _audio_to_complex(self, audio_array: np.array):
        """sets the complex coordinates and amplitudes instance variable from the audio array"""
        complex_coords = self.transformer(
            audio_array, hop_length=self.hop_length, window='hann')
        return complex_coords, np.abs(complex_coords)

    def complex_to_audio(self, complex_coords: np.array, outfile_path=None):
        """
        inverse short term fourrier transform for audio file reconstruction from complex coordinates.
        saves audiofile if path is provided
        """

        assert str(type(complex_coords)) == "<class 'numpy.ndarray'>"

        # complex numbers are still transformed and log-scaled. First inverse-convert complex.
        complex_coords = self._inverse_convert_complex(complex_coords)

        # check if truncation happened. If so, restore to original size
        if (self.freq_bin_cutoff != None):
            complex_coords = self.truncater.restore(complex_coords)

        # convert complex numbers to audio with stft or q-transform
        audio_array = self.inverse_transformer(
            complex_coords, hop_length=self.hop_length, window='hann')
        if (outfile_path != None):
            sf.write(outfile_path, audio_array, self.rate, subtype='PCM_24')

        # sets the audio out (array) instance variable from the saved file
        self.audio_out = AudioSegment.from_wav(outfile_path)

    def _scale_minmax(self, real, imaginary):
        """ scales real and imaginary with sklearn scaler """
        output = []
        for name, values in [("real", real), ("imag", imaginary)]:

            self.scalers[name] = MinMaxScaler()
            self.scalers[name].fit(values)
            scaled = self.scalers[name].transform(values)
            dump(self.scalers[name], open(os.path.join(
                self.experiment_path, f"scaler_{name}.pkl"), 'wb'))
            output.append(scaled)

        return output

    def _transform_complex(self, complex_cords):
        """ separates real and imaginary, applies log transform and transforms separately with minmax scaler """

        complex_cords = np.log(complex_cords)
        # put the real and imaginary numbers into separate dimensions
        real = np.real(complex_cords)
        imaginary = np.imag(complex_cords)
        return self._scale_minmax(real, imaginary)

    def _inverse_convert_complex(self, complex_converted: np.array):
        """ inverts back to complex numbers, makes complex numbers ready for transforming to audio

        Args:
            complex_converted (np.array): minmax-scaled and log transformed 3-channel spectrogram

        Returns:
            complex_values (np.array): complex numbers that are ready for inverse-stft
        """

        # invert the minmax
        real = self.scalers['real'].inverse_transform(
            complex_converted[:, :, 0])
        imaginary = self.scalers['imag'].inverse_transform(
            complex_converted[:, :, 1])
        # put into complex numbers
        complex_numbers = real + 1j*imaginary
        complex_numbers = np.exp(complex_numbers)

        return complex_numbers

    def _complex_to_png(self, real_transf, imag_transf, mask, complex_coords_transf):
        # write to file
        imageio.imwrite(uri=self.paths.complex_coords,
                        im=complex_coords_transf)
        imageio.imwrite(uri=os.path.join(self.experiment_path,
                        "real_scaled.png"), im=real_transf*mask)
        imageio.imwrite(uri=os.path.join(self.experiment_path,
                        "imag_scaled.png"), im=imag_transf*mask)
        print(f"saved the png files to {self.experiment_path}")

    def _convert_complex(self, complex_coords):
        """
        Conversion to ML-friendly format: truncates freq bins for memory, log-scales + normalizes them
        """
        # TODO: remove this. temporary time contraint
        complex_coords = complex_coords[:, :500]

        # truncate a set of frequencies to add again later
        if (self.freq_bin_cutoff != None):
            self.truncater = Truncater(
                complex_coords, self.freq_bin_cutoff)
            complex_coords = self.truncater.truncated

        # transform to
        real_transf, imag_transf = self._transform_complex(complex_coords)

        # stack to save it as a .png
        complex_coords_transf = np.stack(
            [real_transf, imag_transf, np.zeros_like(real_transf)], axis=2)

        # set low amplitudes to 0
        mask = real_transf > 0.0
        complex_coords_transf = complex_coords_transf * \
            np.expand_dims(mask, axis=-1)

        self._complex_to_png(real_transf, imag_transf,
                             mask, complex_coords_transf)
        return complex_coords_transf
