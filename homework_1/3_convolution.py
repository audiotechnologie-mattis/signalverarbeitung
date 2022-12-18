import matplotlib.pyplot as plt
import numpy as np
import numpy.testing


def convolve(a, b):
    length = len(a) + len(b) - 1
    return [
        sum([
            a[k] * b[n - k]
            for k in range(0, n + 1)
            if k < len(a) and n - k < len(b)
        ])
        for n in range(length)
    ]


def convolve_david(a, b):
    length = len(a) + len(b) - 1
    b_r = b + (length - len(b)) * [0]
    b_r.reverse()
    a_factor = a + (length-len(a)) * [0]
    c = length * [0]
    for t in range(length):
        b_factor = b_r[(length-t - 1):len(b_r)] + (length-t - 1) * [0]
        c[t] = sum([x * y for x, y in zip(a_factor, b_factor)])
    return c


a = [1, -1]
b = [2, 0, 0, 1]
result = convolve(a, b)
expected_result = [2, -2, 0, 1, -1]
numpy.testing.assert_equal(result, expected_result)


# sin generation
def generate_sin(n, f, fs):
    w0 = f / fs
    x = np.arange(n)
    return np.sin(2 * np.pi * w0 * x)


h = [0.2] * 5

n = 16
fs = 16000

frequencies = [1000, 4000]
figure, axes_list = plt.subplots(2, 2)

for f, [axis_1, axis_2] in zip(frequencies, axes_list):
    sin = generate_sin(n, f, fs)
    convolved_sin = convolve(sin, h)

    axis_1.set_title(f'sin at {f} Hz')
    axis_1.set_ybound(-1, 1)
    axis_1.plot(sin)
    axis_2.set_title(f'convolved sin at {f} Hz')
    axis_2.plot(convolved_sin)
    axis_2.set_ybound(-1, 1)

figure.show()


# comparisons
figure, [axes_1, axes_2] = plt.subplots(2)

x1 = generate_sin(16, 1000, 16000)
x2 = generate_sin(16, 2000, 16000)

axes_1.plot(convolve(x1, h), label='x * h')
axes_1.plot(convolve(h, x1), label='h * x', linestyle='dashed')
axes_1.legend()

axes_2.plot(
    np.add(convolve(x1, h), convolve(x2, h)),
    label='(x1[n] * h[n]) + (x2[n] * h[n])'
)
axes_2.plot(
    convolve(np.add(x1, x2), h),
    label='(x1[n] + x2[n]) * h[n]',
    linestyle='dashed'
)
axes_2.legend()

figure.show()
