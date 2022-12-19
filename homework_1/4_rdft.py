import timeit

import numpy as np
import numpy.testing
import cmath


def rdft_mattis(x, fs):
    length = len(x) // 2 + 1

    delta_f = fs / len(x)
    frequencies = [n * delta_f for n in range(length)]

    X = [
        sum([
            x[n] * cmath.exp(-2j * cmath.pi / len(x) * k * n)
            for n in range(len(x))
        ])
        for k in frequencies
    ]

    return X, frequencies


def rdft_flo(time_signal, sampling_rate):
    N = len(time_signal)
    delta_f = sampling_rate / N
    freq_vector = [k * delta_f for k in range(N//2 + 1)]
    spectrum = []

    for k in range(N//2 + 1):

        omega_k = 2 * np.pi * k / N
        n_loop = [time_signal[n] * np.exp(-1j * omega_k * n) for n in range(N)]
        spectrum.append(np.sum(n_loop))

    return (spectrum, freq_vector)


def rdft(x, fs):
    N = len(x)
    length = N // 2 + 1
    n = np.arange(N)

    k = n[:length].reshape((length, 1))
    matrix = np.exp(-2j * np.pi / N * k * n)
    X = np.dot(matrix, x)

    delta_f = fs / N
    f = n[:length] * delta_f
    return X, f


N = [1024, 1025, 1026, 1027, 4096, 8192]

# result tests
for n in N:
    x = np.random.rand(n) * 2 - 1
    fs = n

    X, frequencies = rdft(x, fs)
    expected_result = np.fft.rfft(x)
    expected_frequencies = np.fft.rfftfreq(n, 1 / fs)
    numpy.testing.assert_allclose(X, expected_result)
    numpy.testing.assert_equal(frequencies, expected_frequencies)

# time measures
repetitions = 5
print(
    f'{"n":9}'
    f'{"rdft average time (ms)":27}'
    f'{"numpy rfft average time (ms)":27}'
)
for n in N:
    x = np.random.rand(n) * 2 - 1
    seconds_taken_rdft = timeit.timeit(
        'rdft(x, n)',
        number=repetitions,
        globals={"rdft": rdft, "x": x, "n": n}
    )
    seconds_taken_rfft = timeit.timeit(
        'np.fft.rfft(x), np.fft.rfftfreq(n, 1 / n)',
        setup='import numpy as np',
        number=repetitions,
        globals={"x": x, "n": n}
    )
    print(
        f'{n:<9}'
        f'{(seconds_taken_rdft / repetitions * 1000):<27.5f}'
        f'{(seconds_taken_rfft / repetitions * 1000):<27.5f}'
    )
