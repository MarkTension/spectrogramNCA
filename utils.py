import matplotlib.pyplot as plt
import imageio
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
import moviepy.editor as mvp
import os
import numpy as np
import matplotlib.pylab as pl
import librosa
import yaml

os.environ['FFMPEG_BINARY'] = 'ffmpeg'

class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def load_config(path):
    with open(path, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    return AttributeDict(config)

def write_config(path, config):
    with open(os.path.join(path, "config.yaml"), "w") as file:
        config = yaml.dump(dict(config), file)


def plot_spectrogram(spectrogram_path:str, amplitudes:np.array):
    """ saves spectrogram image """

    fig, ax = plt.subplots()
    img = librosa.display.specshow(librosa.amplitude_to_db(amplitudes,
                                                            ref=np.max),
                                    y_axis='linear', x_axis='time', ax=ax)
    ax.set_title('Power spectrogram')
    # fig.colorbar(img, ax=ax, format="%+2.0f dB")
    fig.subplots_adjust(bottom=0)
    fig.subplots_adjust(top=1)
    fig.subplots_adjust(right=1)
    fig.subplots_adjust(left=0)
    fig.savefig(spectrogram_path)


def imsave(image, id=None, count=0, path='results'):
    imageio.imwrite(os.path.join(
        path, f"sample_{count}_{id}.png"), (image*255).astype(np.uint8))


def grab_plot(close=True):
    """Return the current Matplotlib figure as an image"""
    fig = pl.gcf()
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer._renderer)
    a = np.float32(img[..., 3:]/255.0)
    img = np.uint8(255*(1.0-a) + img[..., :3] * a)  # alpha
    if close:
        pl.close()
    return img


def zoom(img, scale=4):
    img = np.repeat(img, scale, 0)
    img = np.repeat(img, scale, 1)
    return img


class VideoWriter:
    def __init__(self, filename='_autoplay.mp4', fps=30.0, **kw):
        self.writer = None
        self.params = dict(filename=filename, fps=fps, **kw)

    def add(self, img):
        img = np.asarray(img)
        if self.writer is None:
            h, w = img.shape[:2]
            self.writer = FFMPEG_VideoWriter(size=(w, h), **self.params)
        if img.dtype in [np.float32, np.float64]:
            img = np.uint8(img.clip(0, 1)*255)
        if len(img.shape) == 2:
            img = np.repeat(img[..., None], 3, -1)
        self.writer.write_frame(img)

    def close(self):
        if self.writer:
            self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *kw):
        self.close()
        if self.params['filename'] == '_autoplay.mp4':
            self.show()

    def show(self, **kw):
        self.close()


class LoopWriter(VideoWriter):
    def __init__(self, *a, cross_len=1.0, **kw):
        super().__init__(*a, **kw)
        self._intro = []
        self._outro = []
        self.cross_len = int(cross_len*self.params['fps'])

    def add(self, img):
        if len(self._intro) < self.cross_len:
            self._intro.append(img)
            return
        self._outro.append(img)
        if len(self._outro) > self.cross_len:
            super().add(self._outro.pop(0))

    def close(self):
        for t in np.linspace(0, 1, len(self._intro)):
            img = self._intro.pop(0)*t + self._outro.pop(0)*(1.0-t)
            super().add(img)
        super().close()
