import matplotlib.pyplot as plt
import numpy as np


def generate_sin(n, f, fs):
    w0 = f / fs
    x = np.arange(n)
    return np.sin(2 * np.pi * w0 * x)


f = 15_000

figure, axes = plt.subplots()
axes.set_xlabel('time (ms)')
axes.set_ylabel('amplitude')

for fs, plot_function in zip(
    [192_000, 16_000],
    [lambda x, y: axes.plot(x, y, color='orange'),
     lambda x, y: axes.stem(x, y)]
):
    length = fs // 1000 + 1
    x = np.arange(0, length) / fs * 1000
    y = generate_sin(length, f, fs)
    plot_function(x, y)

figure.show()
