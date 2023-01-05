import librosa
import librosa.display
from enum import Enum
import os
from datetime import datetime

from utils import AttributeDict, load_config, write_config
from stft_transformer import StftTransformer
from train import train, sample

"""
some things to do
TODO: return one numpy array at every training interval besides an image
TODO: for audio reconstruction, clip amplitudes above a certain threshold.
TODO: save the models, and allow loading them
TODO: auto-generate audio sequence
TODO: make yaml config
TODO: integrate new loss function with style loss function
TODO: look at cowbell reconstruction error!
"""

class Experiment(Enum):
    RGB = 1
    COMPLEX = 2

def set_paths(sound_name, load_model_path):
    "puts all the paths in a dict"

    if load_model_path == None:
        experiment_root = os.path.join("experiments", datetime.now().strftime("%m%d_%H%M%S"))
    else:
        experiment_root = os.path.join("experiments", load_model_path)
    nca_results = os.path.join(experiment_root, "nca_results")
    wav_out = os.path.join(nca_results, "audio_output")
    nca_results_training = os.path.join(nca_results, "training")

    paths = {
        "experiment" : experiment_root,
        "nca_results" : nca_results,
        "model_path" : os.path.join(experiment_root, "model.pkl"),
        "nca_video": os.path.join(nca_results, "outvideo.mp4"),
        "nca_audio": os.path.join(wav_out, "outaudio.wav"),
        "nca_results_training": nca_results_training,
        "input_wav" : os.path.join("samples", f"{sound_name}.wav"),
        "reconstructed_wav" : os.path.join(experiment_root, f"{sound_name}_reconstruced.wav"),
        "spectrogram" : os.path.join(experiment_root, f"{sound_name}_spect.png"),
        "complex_coords" : os.path.join(experiment_root, f"{sound_name}_complex_coords.png"),
    }
    if os.path.exists(experiment_root):
        return AttributeDict(paths)

    os.mkdir(experiment_root)
    os.mkdir(nca_results)
    os.mkdir(wav_out)
    os.mkdir(nca_results_training)

    return AttributeDict(paths)


# class config:
#     rate = 48000
#     n_fft = 1000 # 2000
#     experiment = Experiment.COMPLEX
#     sound_name = "texture1"#"808_cowbell_48k" #"kicksnare" #'drum' #'texture1' #"bellPlate"
#     freq_bin_cutoff = None # 256
#     load_model_path = "blabla" # set to the experiment id (the datetime string) to load a previous model

def main():

    config = load_config("config.yaml")
    # or load the old config 
    load_model_path = None if config.load_model_path == "None" else config.load_model_path # eval in case None
    if load_model_path != None:
        config = load_config(os.path.join("experiments" ,load_model_path, "config.yaml"))

    paths = set_paths(config.sound_name, load_model_path)

    if load_model_path == None:
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
        complex_numbers = transformer.convert_complex()
        recon_complex_numbers = transformer.inverse_convert_complex(complex_numbers)
        transformer.complex_coords = recon_complex_numbers
        transformer.complex_to_audio(paths.reconstructed_wav)

        if load_model_path == None:
            train(image=complex_numbers, paths=paths, transformer=transformer)
        else:
            sample(transformer, paths, complex_numbers.shape[:2])
    
    # to test if reverse worksb
    transformer.complex_to_audio(paths.reconstructed_wav) # which saves it
    # transformer.play_sounds() # plays sound # doesn't work on windows
    

print('done')

if __name__ == "__main__":
    main()