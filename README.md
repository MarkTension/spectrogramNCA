## README

This code is for doing neural cellular automata on audio.
For this, the audio-spectrogram is used.

Note that the audio spectrogram consists of amplitude, and phase information. 
they are stored as complex numbers: amplitude = real, phase=imaginary.

The goal is to generate audio from evolving NCA, where I can interactively destroy parts of the signal, and it grows back to its original again.

### experiment 1
do NCA on an image of the audio-spectrogram (just the amplitudes - decibel values, neglecting the phase information)
### experiment 2
do NCA on both the amplitude and phase information that are stored in the complex values. 

## components
There is an transformer class that can generate audio (wav) <--> spectrograms/complex values


### NOTES
Taking the full sized complex numbers goes quite slow. 
Taking only a slice of 500 x 500 goes faster, but sometimes stops. will try without halo infinte on the side