import librosa
import librosa.display
from enum import Enum
import os
from datetime import datetime

from utils import AttributeDict
from stft_transformer import StftTransformer
from train import train

class Experiment(Enum):
    RGB = 1
    COMPLEX = 2

def set_paths(sound_name):
    "puts all the paths in a dict"
    
    experiment_root = os.path.join("experiments",datetime.now().strftime("%m%d_%H%M%S"))
    os.mkdir(experiment_root)

    paths = {
        "experiment" : experiment_root,
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
    sound_name = "bellPlate"

def main():
    # set configuration. TODO: make yaml config    
    print(f"doing {config.experiment}")
    paths = set_paths(config.sound_name)

    data, sample_length = librosa.load(paths.input_wav, sr=config.rate)

    # use transformer object to fourrier transform
    transformer = StftTransformer(  n_fft=config.n_fft, 
                                    rate=config.rate, 
                                    audio_array=data, 
                                    spect_path=paths.spectrogram)

    # TEST 1: generate spectrogram by image. Do NCA on this
    if (config.experiment == Experiment.RGB):
        train(image_path=paths.spectrogram)

    # TEST 2: generate spectrogram by complex values. Do NCA on this
    if (config.experiment == Experiment.COMPLEX):
        transformer.complex_to_png()
        train(image_path=paths.complex_coords)

    # to test if reverse works
    transformer.complex_to_audio(paths.reconstructed_wav) # which saves it
    # transformer.play_sounds() # plays sound # doesn't work on windows
    

print('done')

if __name__ == "__main__":
    main()