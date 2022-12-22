import librosa
import librosa.display
import numpy as  np
from enum import Enum

from stft_transformer import StftTransformer
from train import train

def generate_spectrogram(transformer:StftTransformer, spectrogram_path:str):
    transformer.plot_spectrogram(spectrogram_path) # which saves it


class Experiment(Enum):
    RGB = 1 # using the RGB image of the spectrogram to train
    POLARS = 2 # using the raw polar coordinates resulting from the fourrier transform


def main():

    # set configuration. TODO: make yaml config
    rate = 48000
    n_fft = 2000
    # hop_length = n_fft // 4
    training = Experiment.POLARS
    
    # files
    sound_name = "bellPlate"
    path = f"samples/{sound_name}.wav"
    outfile_path = f"samples/{sound_name}_reconstruced.wav"
    spectrogram_path = f"spectrograms/{sound_name}.png"

    data, output_length = librosa.load(path, sr=rate)

    # make sure data is square

    # use transformer to fourrier transform and back, and play sound
    transformer = StftTransformer(n_fft=n_fft, rate=rate)
    transformer.audio_to_polar(audio_array=data)

    # TEST 1: generate spectrogram by image. Will do a NCA on this
    if (training == Experiment.RGB):
        print("Experiment with RGB values of spectrogram")
        generate_spectrogram(transformer, spectrogram_path)
        train(image_path=spectrogram_path)

    if (training == Experiment.POLARS):
        print("Experiment with polar coordinates")
        transformer.polar_to_png("polar_coords.png")
        train(image_path="polar_coords.png")

    # to test if reverse works
    transformer.polar_to_audio(outfile_path) # which saves it
    # transformer.play_sounds() # plays sound
    

print('done')

if __name__ == "__main__":
    main()