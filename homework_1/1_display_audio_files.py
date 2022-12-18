import numpy as np
import soundfile as sf
from matplotlib import plt

figure, axes_list = plt.subplots(ncols=2)
file_names = ['speech.wav', 'impulse_response.wav']

for file, axes in zip(file_names, axes_list):
    data, sample_rate = sf.read(file, always_2d=True)
    first_channel = np.rot90(data)[0]
    first_channel_normalized = first_channel / np.max(np.abs(first_channel))

    x = np.arange(0, len(first_channel)) / sample_rate

    axes.set_title(file)
    axes.set_ylim([-1, 1])
    axes.set_xlabel('time in seconds')
    axes.set_ylabel('amplitude')
    axes.plot(x, first_channel_normalized)

figure.show()
