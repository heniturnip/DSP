# main.py
import numpy as np
import matplotlib.pyplot as plt
from video_capture import capture_video
from rppg import cpu_POS, filter_rppg, calculate_respiratory_signal
from heart_rate import calculate_heart_rate
import scipy.signal as signal

def main():
    # Mengambil sinyal dari video
    r_signal, g_signal, b_signal, fps = capture_video()

    if r_signal is None or g_signal is None or b_signal is None or fps is None:
        print("Gagal menangkap video.")
        return

    # Menghitung rPPG menggunakan Metode POS
    rgb_signals = np.array([r_signal, g_signal, b_signal])
    rgb_signals = rgb_signals.reshape(1, 3, -1)
    rppg_signal = cpu_POS(rgb_signals, fps=fps)
    rppg_signal = rppg_signal.reshape(-1)

    # Memfilter Sinyal rPPG
    filtered_rppg = filter_rppg(rppg_signal, fps)

    # Menghitung Sinyal Respirasi
    filtered_respirasi = calculate_respiratory_signal(rppg_signal, fps)

    # Menghitung Heart Rate
    heart_rate, peaks = calculate_heart_rate(filtered_rppg, fps)

    # Menghitung Respiratory Rate
    # Normalisasi Sinyal
    filtered_respirasi = (filtered_respirasi - np.mean(filtered_respirasi)) / np.std(filtered_respirasi)

    # Mencari puncak sinyal
    peaks_respirasi, _ = signal.find_peaks(
        x=filtered_respirasi,
        prominence=0.5,
    )

    # Menghitung respiratory rate
    respiratory_rate = 60 * len(peaks_respirasi) / (len(filtered_respirasi) / fps)

    # Menampilkan grafik sinyal RGB
    fig, ax = plt.subplots(3, 1, figsize=(20, 6))
    ax[0].plot(r_signal, color='red')
    ax[0].set_title('Sinyal Merah')
    ax[1].plot(g_signal, color='green')
    ax[1].set_title('Sinyal Hijau')
    ax[2].plot(b_signal, color='blue')
    ax[2].set_title('Sinyal Biru')
    plt.tight_layout()
    plt.show()

    # Menampilkan grafik sinyal rPPG
    fig, ax = plt.subplots(3, 1, figsize=(20, 6))
    ax[0].plot(rppg_signal, color='black')
    ax[0].set_title('Sinyal rPPG')
    ax[1].plot(filtered_rppg, color='black')
    ax[1].set_title('Sinyal rPPG yang Telah Difilter')
    ax[2].plot(filtered_respirasi, color='blue')
    ax[2].set_title('Sinyal Pernapasan yang Telah Difilter')
    plt.tight_layout()
    plt.show()

    # Menampilkan puncak sinyal Heart Rate dan Respiratory Rate
    fig, ax = plt.subplots(2, 1, figsize=(20, 6))

    # Menampilkan puncak sinyal Heart Rate
    ax[0].plot(filtered_rppg, color='black', label='Sinyal rPPG yang Telah Difilter')
    ax[0].plot(peaks, filtered_rppg[peaks], 'x', color='red', label='Puncak Detak Jantung')
    ax[0].set_title(f'Detak Jantung: {heart_rate:.2f} BPM')
    ax[0].legend()

    # Menampilkan puncak sinyal Respiratory Rate
    ax[1].plot(filtered_respirasi, color='blue', label='Sinyal Pernapasan yang Telah Difilter')
    ax[1].plot(peaks_respirasi, filtered_respirasi[peaks_respirasi], 'x', color='red', label='Puncak Pernapasan')
    ax[1].set_title(f'Laju Pernapasan: {respiratory_rate:.2f} BPM')
    ax[1].legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
