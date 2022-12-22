import librosa
import librosa.display
import numpy as  np

from stft_transformer import StftTransformer
from train import train

def generate_spectrogram(transformer:StftTransformer, spectrogram_path:str):
    transformer.plot_spectrogram(spectrogram_path) # which saves it


def main():

    # set configuration. TODO: make yaml config
    rate = 48000
    n_fft = 200
    # hop_length = n_fft // 4
    
    # files
    sound_name = "bellPlate"
    path = f"samples/{sound_name}.wav"
    outfile_path = f"samples/{sound_name}_reconstruced.wav"
    spectrogram_path = f"spectrograms/{sound_name}.wav"

    data, output_length = librosa.load(path, sr=rate)

    # use transformer to fourrier transform and back, and play sound
    transformer = StftTransformer(n_fft=n_fft, rate=rate)
    transformer.audio_to_polar(audio_array=data)

    # TEST 1: generate spectrogram by image. Will do a NCA on this
    generate_spectrogram(transformer, spectrogram_path)
    train(image_path=spectrogram_path)

    # to test if reverse works
    transformer.polar_to_audio(outfile_path) # which saves it
    transformer.play_sounds() # plays sound
    


print('done')

if __name__ == "__main__":
    main()