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

  eps = 10**-9                                # Nilai kecil untuk menghindari pembagian dengan nol
  X = input_signal
  e, c, f = X.shape                           # Mendapatkan dimensi input: estimator, channel, frame
  w = int(1.6 * kargs['fps'])                 # Menghitung panjang window berdasarkan fps

  P = np.array([[0, 1, -1], [-2, 1, 1]])      # Matriks proyeksi untuk transformasi warna
  Q = np.stack([P for _ in range(e)], axis=0)  # Membuat tumpukan matriks P untuk setiap estimator

  H = np.zeros((e, f))                         # Inisialisasi array output
  for n in np.arange(w, f):
    m = n - w + 1
    Cn = X[:, :, m:(n + 1)]                 # Mengambil window dari sinyal input
    M = 1.0 / (np.mean(Cn, axis=2) + eps)   # Menghitung faktor normalisasi
    M = np.expand_dims(M, axis=2)            # Menambah dimensi untuk broadcasting
    Cn = np.multiply(M, Cn)                  # Normalisasi sinyal

    S = np.dot(Q, Cn)                       # Proyeksi sinyal ke ruang warna baru
    S = S[0, :, :, :]
    S = np.swapaxes(S, 0, 1)                # Menyesuaikan dimensi array

    S1 = S[:, 0, :]                         # Komponen pertama sinyal
    S2 = S[:, 1, :]                         # Komponen kedua sinyal
    alpha = np.std(S1, axis=1) / (eps + np.std(S2, axis=1))  # Menghitung rasio standar deviasi
    alpha = np.expand_dims(alpha, axis=1)    # Menambah dimensi untuk broadcasting
    Hn = np.add(S1, alpha * S2)             # Kombinasi komponen dengan bobot alpha
    Hnm = Hn - np.expand_dims(np.mean(Hn, axis=1), axis=1)  # Menghilangkan mean
    H[:, m:(n + 1)] = np.add(H[:, m:(n + 1)], Hnm)  # Menyimpan hasil ke array output

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
    lowcut = 0.9                                   # Frekuensi cut-off bawah untuk filter
    highcut = 2                              # Frekuensi cut-off atas untuk filter
    order = 3                                   # Orde filter Butterworth
    b, a = signal.butter(order, [lowcut, highcut], btype='band', fs=fps)  # Koefisien filter
    filtered_rppg = signal.filtfilt(b, a, rppg_signal)  # Menerapkan filter
    return filtered_rppg

def calculate_respiratory_signal(rppg_signal, fps):
    """
    Menghitung sinyal respirasi dari sinyal rPPG.

    Parameter:
    - rppg_signal (numpy.ndarray): Sinyal rPPG input.
    - fps (float): Frame per detik dari video input.

    Mengembalikan:
    - numpy.ndarray: Sinyal respirasi yang telah difilter.
    """
    lowcut_respirasi = 0.1                      # Frekuensi cut-off bawah untuk respirasi
    highcut_respirasi = 0.5                     # Frekuensi cut-off atas untuk respirasi
    order = 3                                   # Orde filter Butterworth
    b_respirasi, a_respirasi = signal.butter(order, [lowcut_respirasi, highcut_respirasi], btype='band', fs=fps) # Koefisien filter
    filtered_respirasi = signal.filtfilt(b_respirasi, a_respirasi, rppg_signal)  # Menerapkan filter
    return filtered_respirasi
