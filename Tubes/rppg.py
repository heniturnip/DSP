# rppg.py
import numpy as np
import scipy.signal as signal

def cpu_POS(input_signal, **kargs):
    """
    Metode POS pada CPU menggunakan Numpy.

    Fungsi ini mengimplementasikan metode ekstraksi sinyal Photoplethysmography (PPG)
    menggunakan Prinsip Ruang (POS). Fungsi ini memproses sinyal RGB input untuk
    mengekstrak sinyal detak jantung.

    Parameter:
    - input_signal (numpy.ndarray): Array 3D dengan bentuk (e, c, f) di mana e adalah jumlah estimator,
      c adalah jumlah saluran warna (3 untuk RGB), dan f adalah jumlah frame.
    - kargs (dict): Argumen tambahan, khususnya:
      - 'fps' (float): Frame per detik dari video input.

    Mengembalikan:
    - numpy.ndarray: Array 2D dengan bentuk (e, f) yang berisi sinyal detak jantung yang diekstrak.
    """
    eps = 10**-9
    X = input_signal
    e, c, f = X.shape            # e = #estimators, c = 3 rgb ch., f = #frames
    w = int(1.6 * kargs['fps'])   # panjang jendela

    P = np.array([[0, 1, -1], [-2, 1, 1]])
    Q = np.stack([P for _ in range(e)], axis=0)

    H = np.zeros((e, f))
    for n in np.arange(w, f):
        m = n - w + 1
        Cn = X[:, :, m:(n + 1)]
        M = 1.0 / (np.mean(Cn, axis=2) + eps)
        M = np.expand_dims(M, axis=2)  # bentuk [e, c, w]
        Cn = np.multiply(M, Cn)

        S = np.dot(Q, Cn)
        S = S[0, :, :, :]
        S = np.swapaxes(S, 0, 1)    # menghapus dimensi ke-3

        S1 = S[:, 0, :]
        S2 = S[:, 1, :]
        alpha = np.std(S1, axis=1) / (eps + np.std(S2, axis=1))
        alpha = np.expand_dims(alpha, axis=1)
        Hn = np.add(S1, alpha * S2)
        Hnm = Hn - np.expand_dims(np.mean(Hn, axis=1), axis=1)
        H[:, m:(n + 1)] = np.add(H[:, m:(n + 1)], Hnm)

    return H

def filter_rppg(rppg_signal, fps):
    """
    Memfilter sinyal rPPG menggunakan filter bandpass.

    Parameter:
    - rppg_signal (numpy.ndarray): Sinyal rPPG yang akan difilter.
    - fps (float): Frame per detik dari video input.

    Mengembalikan:
    - numpy.ndarray: Sinyal rPPG yang telah difilter.
    """
    lowcut = 0.9
    highcut = 2.4
    order = 3
    b, a = signal.butter(order, [lowcut, highcut], btype='band', fs=fps)
    filtered_rppg = signal.filtfilt(b, a, rppg_signal)
    return filtered_rppg

def calculate_respiratory_signal(rppg_signal, fps):
    """
    Menghitung sinyal pernapasan dari sinyal rPPG.

    Parameter:
    - rppg_signal (numpy.ndarray): Sinyal rPPG yang digunakan untuk menghitung sinyal pernapasan.
    - fps (float): Frame per detik dari video input.

    Mengembalikan:
    - numpy.ndarray: Sinyal pernapasan yang telah difilter.
    """
    lowcut_respirasi = 0.1
    highcut_respirasi = 0.3
    order = 3
    b_respirasi, a_respirasi = signal.butter(order, [lowcut_respirasi, highcut_respirasi], btype='band', fs=fps)
    filtered_respirasi = signal.filtfilt(b_respirasi, a_respirasi, rppg_signal)
    return filtered_respirasi
