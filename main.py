import librosa
import librosa.display
from enum import Enum
import os
from datetime import datetime

from utils import AttributeDict
from stft_transformer import StftTransformer
from train import train
import matplotlib.pyplot as plt
import numpy as np

"""
some things to do
TODO: return one numpy array at every training interval besides an image
TODO: convert each of these training arrays back to wav
TODO: remove all phase and amplitudes where it is below a certain value


- can we make it rectangular? YES!
- can we go for 2 channels instead of 3?
"""

class Experiment(Enum):
    RGB = 1
    COMPLEX = 2

def set_paths(sound_name):
    "puts all the paths in a dict"
    
    experiment_root = os.path.join("experiments",datetime.now().strftime("%m%d_%H%M%S"))
    nca_results = os.path.join(experiment_root, "nca_results")
    os.mkdir(experiment_root)
    os.mkdir(nca_results)

    paths = {
        "experiment" : experiment_root,
        "nca_results" : nca_results,
        "nca_video": os.path.join(nca_results, "outvideo.mp4"),
        "input_wav" : os.path.join("samples", f"{sound_name}.wav"),
        "reconstructed_wav" : os.path.join(experiment_root, f"{sound_name}_reconstruced.wav"),
        "spectrogram" : os.path.join(experiment_root, f"{sound_name}_spect.png"),
        "complex_coords" : os.path.join(experiment_root, f"{sound_name}_complex_coords.png"),
    }

    return AttributeDict(paths)


class config:
    rate = 48000
    n_fft = 2000
    experiment = Experiment.COMPLEX
    sound_name = 'texture1' #"bellPlate"
    freq_bin_cutoff = 256

def main():
    # set configuration. TODO: make yaml config    
    print(f"doing {config.experiment}")
    paths = set_paths(config.sound_name)

    data, sample_length = librosa.load(paths.input_wav, sr=config.rate)

    # use transformer object to fourrier transform
    transformer = StftTransformer(  n_fft=config.n_fft, 
                                    rate=config.rate, 
                                    audio_array=data, 
                                    paths=paths,
                                    freq_bin_cutoff=config.freq_bin_cutoff,
                                    method='cqt')

    # TEST 1: generate NCA on image of spectrogram
    if (config.experiment == Experiment.RGB):
        train(image_path=paths.spectrogram)

    # TEST 2: generate NCA on complex values of spectrogram 
    if (config.experiment == Experiment.COMPLEX):
        complex_numbers = transformer.convert_complex()
        recon_complex_numbers = transformer.inverse_convert_complex(complex_numbers)
        transformer.complex_coords = recon_complex_numbers
        transformer.complex_to_audio(paths.reconstructed_wav)

        train(image=complex_numbers, paths=paths, transformer=transformer)

    # to test if reverse works
    transformer.complex_to_audio(paths.reconstructed_wav) # which saves it
    # transformer.play_sounds() # plays sound # doesn't work on windows
    

print('done')

if __name__ == "__main__":
    main()