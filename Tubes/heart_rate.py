# heart_rate.py
import numpy as np
import scipy.signal as signal

def calculate_heart_rate(filtered_rppg, fps):
    """
    Menghitung detak jantung dari sinyal rPPG yang telah difilter.

    Parameter:
    - filtered_rppg (numpy.ndarray): Sinyal rPPG yang telah difilter.
    - fps (float): Frame per detik dari video input.

    Mengembalikan:
    - float: Detak jantung dalam BPM (beats per minute).
    """
    # Normalisasi Sinyal
    filtered_rppg = (filtered_rppg - np.mean(filtered_rppg)) / np.std(filtered_rppg)

    # Mencari puncak sinyal
    peaks, _ = signal.find_peaks(
        x=filtered_rppg,
        prominence=0.5,
    )

    # Menghitung heart rate
    heart_rate = 60 * len(peaks) / (len(filtered_rppg) / fps)
    return heart_rate, peaks
