import librosa
import librosa.display
from enum import Enum
import os
from datetime import datetime

from utils import AttributeDict, load_config, write_config, set_paths
from stft_transformer import StftTransformer
from train import train, sample

"""
some things to do
TODO: for audio reconstruction, clip amplitudes above a certain threshold.
TODO: auto-generate audio sequence
TODO: integrate new loss function with style loss function
"""

class Experiment(Enum):
    RGB = 1
    COMPLEX = 2

def main():

    config = load_config("config.yaml")
    # or load the old config 
    load_model_path = None if config.load_model_path == "None" else config.load_model_path # eval in case None
    if load_model_path != None:
        config = load_config(os.path.join("experiments" ,load_model_path, "config.yaml"))

    paths = set_paths(config.sound_name, load_model_path)

    if load_model_path is None:
        write_config(paths.experiment, config)

    data, sample_length = librosa.load(paths.input_wav, sr=config.rate)

    # use transformer object to fourrier transform
    transformer = StftTransformer(  n_fft=config.n_fft, 
                                    rate=config.rate, 
                                    audio_array=data, 
                                    paths=paths,
                                    freq_bin_cutoff=eval(config.freq_bin_cutoff),
                                    sample_len = sample_length,
                                    method='cqt')

    # TEST 1: generate NCA on image of spectrogram
    if (eval(config.experiment) == Experiment.RGB):
        train(image_path=paths.spectrogram)

    # TEST 2: generate NCA on complex values of spectrogram
    if (eval(config.experiment) == Experiment.COMPLEX):
        # convert to scaledcomplex numbers
        complex_numbers = transformer.complex_coords
        transformer.complex_to_audio(complex_numbers, paths.reconstructed_wav)

        if load_model_path is None:
            train(image=complex_numbers, paths=paths, transformer=transformer, config=config)
        else:
            sample(transformer, paths, complex_numbers.shape[:2])

    # to test if reverse works
    # transformer.complex_to_audio(paths.reconstructed_wav) # which saves it
    # transformer.play_sounds() # plays sound # doesn't work on windows
    

print('done')

if __name__ == "__main__":
    main()