import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

from pydub import AudioSegment
from pydub.playback import play


class StftTransformer:
    """
    class is responsible to convert an audio to and from polar coordinates / complex numbers, 
    and visualizing the file with an audio spectrogram
    """

    def __init__(self, n_fft=200, rate=48000):
        self.rate = rate
        self.hop_length = n_fft // 4
        self.amplitudes = None
        self.polar_coords = None
        self.audio_out = None

    def audio_to_polar(self, audio_array: np.array):
        """sets the polar coordinates and amplitudes instance variable from the audio array"""
        # open file
        self.polar_coords = librosa.stft(
            audio_array, hop_length=self.hop_length, window='hann', center=True)  # n_fft=2048,
        self.amplitudes = np.abs(self.polar_coords)

    def polar_to_audio(self, outfile_path=None):
        """
        inverse short term fourrier transform for audio file reconstruction from polar coordinates.
        saves audiofile if path is provided
        """

        assert str(type(self.polar_coords)) == "<class 'numpy.ndarray'>"

        audio_array = librosa.istft(
            self.polar_coords, hop_length=self.hop_length, window='hann')  # n_fft=2048,
        if (outfile_path != None):
            sf.write(outfile_path, audio_array, self.rate, subtype='PCM_24')
            print(f"saved the audio file to {outfile_path}")

        # sets the audio out (array) instance variable from the saved file
        self.audio_out = AudioSegment.from_wav(outfile_path)

    def plot_spectrogram(self, spectrogram_path):
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
        fig.show()
        fig.savefig(spectrogram_path)

        # # and also do the same with only the content
        # content = self.amplitudes
        # fig = plt.figure(frameon=False)
        # fig.set_size_inches(5,5)
        # ax = plt.Axes(fig, [0., 0., 1., 1.])
        # ax.set_axis_off()
        # fig.add_axes(ax)
        # ax.imshow(content, aspect='auto')
        # fig.savefig("nobords.png")

    def play_sounds(self):
        assert self.audio_out != None
        play(self.audio_out)
