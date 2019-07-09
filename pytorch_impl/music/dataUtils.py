import scipy.io.wavfile
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import torch

def info(filename):
    fs, wave = scipy.io.wavfile.read(filename)
    #Note that this particular file has a single channel. Most audio files will have two (stereo) channels.

    print('Data:{}'.format(wave))
    print('Data Dimention:{}'.format(wave.shape))
    print('Sampling rate:{}'.format(fs))
    print('Audio length: {} seconds'.format( wave.size/fs))
    if(wave.ndim >1):
        wave = wave[:,0]

    print('Lowest amplitud:{}'.format(min(wave)))
    print('Highest amplitud:{}'.format(max(wave)))
    return fs, wave


def plotwave(fs, signal, maxf=None):
    """Visualize (a segment of) a wave file."""
    # maxf = maximum number of frames
    frames = scipy.arange(signal.size)   # x-axis
    if maxf:
        plt.plot(frames[:maxf], signal[:maxf])
        plt.xticks(scipy.arange(0, maxf, 0.5*fs), scipy.arange(0, maxf/fs, 0.5))
        plt.show()
    else:
        plt.plot(frames, signal)
        plt.xticks(scipy.arange(0, signal.size, 0.5*fs), scipy.arange(0, signal.size/fs, 0.5))
        plt.show()


def downsample(filename, factor):
    """Lower the sampling rate by factor."""
    newfilename = filename[:-4]+'-down'+str(factor)+'.wav'
    fs, wave = scipy.io.wavfile.read(filename)
    wave = wave[:,0]#wave.flatten()
    newfs = np.int(fs/factor)
    # fill in the rest
    indices = range(0, wave.size, factor)
    wave = wave[indices]
    scipy.io.wavfile.write(newfilename, newfs, wave)

def spectrogram(wavfile):
    fs, wave = scipy.io.wavfile.read(wavfile)
    spec = plt.specgram(wave.flatten(), NFFT=int(fs*0.005), Fs=fs, pad_to=256, noverlap=int(fs*0.0025))
    plt.show()


def generate_spectro(data,freq):
    frequencies, times, spectro = signal.spectrogram(data, freq)
    print(spectro.shape)
    print(frequencies.shape)
    print(times.shape)
    print(data[:, 0].shape)
    return frequencies, times, spectro

def generate_spectro_from_file(musicfile):
    freq,data = info(musicfile)
    frequencies, times, spectro = signal.spectrogram(data, freq)
    print("Shape of Spctrogram= {}".format(spectro.shape))
    print("Shape of Frequencies= {}".format(frequencies.shape))
    print("Shape of Time= {}".format(times.shape))
    return freq, data, frequencies, times, spectro


def to_tensor(data):
    return torch.from_numpy(data).float()

