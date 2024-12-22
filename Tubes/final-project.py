import numpy as np
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import scipy.signal as signal

def cpu_POS(input_signal, **kargs):
    """
    POS method on CPU using Numpy.

    The dictionary parameters are: {'fps':float}.

    Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2016). Algorithmic principles of remote PPG. IEEE Transactions on Biomedical Engineering, 64(7), 1479-1491. 
    """
    eps = 10**-9
    X = input_signal
    e, c, f = X.shape            # e = #estimators, c = 3 rgb ch., f = #frames
    w = int(1.6 * kargs['fps'])   # window length

    P = np.array([[0, 1, -1], [-2, 1, 1]])
    Q = np.stack([P for _ in range(e)], axis=0)

    H = np.zeros((e, f))
    for n in np.arange(w, f):
        m = n - w + 1
        Cn = X[:, :, m:(n + 1)]
        M = 1.0 / (np.mean(Cn, axis=2) + eps)
        M = np.expand_dims(M, axis=2)  # shape [e, c, w]
        Cn = np.multiply(M, Cn)

        S = np.dot(Q, Cn)
        S = S[0, :, :, :]
        S = np.swapaxes(S, 0, 1)    # remove 3rd dim

        S1 = S[:, 0, :]
        S2 = S[:, 1, :]
        alpha = np.std(S1, axis=1) / (eps + np.std(S2, axis=1))
        alpha = np.expand_dims(alpha, axis=1)
        Hn = np.add(S1, alpha * S2)
        Hnm = Hn - np.expand_dims(np.mean(Hn, axis=1), axis=1)
        H[:, m:(n + 1)] = np.add(H[:, m:(n + 1)], Hnm)

    return H

def main():
    # Inisialisasi MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    # Memuat video dari webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    # Mempersiapkan beberapa variabel
    r_signal, g_signal, b_signal = [], [], []
    f_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Perulangan Utama
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Mengkonversi frame ke RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Memproses frame menggunakan face_detection
            results = face_detection.process(frame_rgb)

            if results.detections:  # Jika ada wajah terdeteksi
                for detection in results.detections:  # Loop melalui semua wajah yang terdeteksi
                    # Mendapatkan bounding box dari wajah
                    bbox = detection.location_data.relative_bounding_box
                    # Mendapatkan lebar dan tinggi frame
                    h, w, _ = frame.shape
                    # Mengkonversi bounding box ke koordinat piksel
                    x, y = int(bbox.xmin * w), int(bbox.ymin * h)
                    width, height = int(bbox.width * w), int(bbox.height * h)

                    # Melakukan penyesuaian pada bounding box
                    bbox_size_from_center = 70

                    bbox_center_x = x + width // 2
                    bbox_center_y = y + height // 2
                    new_x = bbox_center_x - bbox_size_from_center
                    new_y = bbox_center_y - bbox_size_from_center
                    new_width = bbox_size_from_center * 2
                    new_height = bbox_size_from_center * 2

                    # Menggambar bounding box pada frame
                    cv2.rectangle(frame, (new_x, new_y), (new_x + new_width, new_y + new_height), (0, 255, 0), 2)

                    # Mendapatkan nilai rata-rata piksel dari ROI dan menambahkannya ke signal
                    roi = frame[new_y:new_y+new_height, new_x:new_x+new_width]
                    r_signal.append(np.mean(roi[:, :, 0]))
                    g_signal.append(np.mean(roi[:, :, 1]))
                    b_signal.append(np.mean(roi[:, :, 2]))

            # Menampilkan frame dengan bounding box
            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            f_count += 1

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        cap.release()
        cv2.destroyAllWindows()

    # Menghitung rPPG menggunakan Metode POS
    rgb_signals = np.array([r_signal, g_signal, b_signal])
    rgb_signals = rgb_signals.reshape(1, 3, -1)
    rppg_signal = cpu_POS(rgb_signals, fps=fps)
    rppg_signal = rppg_signal.reshape(-1)

    # Memfilter Sinyal rPPG
    fs = fps; lowcut = 0.9; highcut = 2.4; order = 3
    b, a = signal.butter(order, [lowcut, highcut], btype='band', fs=fs)
    filtered_rppg = signal.filtfilt(b, a, rppg_signal)

    # Menghitung Sinyal Respirasi
    lowcut_respirasi = 0.1; highcut_respirasi = 0.3
    b_respirasi, a_respirasi = signal.butter(order, [lowcut_respirasi, highcut_respirasi], btype='band', fs=fs)
    filtered_respirasi = signal.filtfilt(b_respirasi, a_respirasi, rppg_signal)

    # Menampilkan grafik sinyal
    fig, ax = plt.subplots(4, 1, figsize=(20, 15))
    ax[0].plot(r_signal, color='red')
    ax[0].set_title('Red Signal')
    ax[1].plot(g_signal, color='green')
    ax[1].set_title('Green Signal')
    ax[2].plot(b_signal, color='blue')
    ax[2].set_title('Blue Signal')
    ax[3].plot(rppg_signal, color='black')
    ax[3].set_title('rPPG Signal')
    plt.tight_layout()
    plt.show()

    # Menampilkan grafik Sinyal rPPG sebelum dan sesudah penyaringan
    fig, ax = plt.subplots(2, 1, figsize=(20, 6))
    ax[0].plot(rppg_signal, color='black')
    ax[0].set_title('rPPG Signal - Before Filtering')
    ax[1].plot(filtered_rppg, color='black')
    ax[1].set_title('rPPG Signal - After Filtering')
    plt.tight_layout()
    plt.show()

    # Menampilkan grafik Sinyal Respirasi sebelum dan sesudah penyaringan
    fig, ax = plt.subplots(2, 1, figsize=(20, 6))
    ax[0].plot(rppg_signal, color='black')
    ax[0].set_title('Respiratory Signal - Before Filtering')
    ax[1].plot(filtered_respirasi, color='black')
    ax[1].set_title('Respiratory Signal - After Filtering')
    plt.tight_layout()
    plt.show()

    # Menghitung Heart Rate
    ## Normalisasi Sinyal
    filtered_rppg = (filtered_rppg - np.mean(filtered_rppg)) / np.std(filtered_rppg)

    ## Mencari puncak sinyal
    peaks, _ = signal.find_peaks(
        x=filtered_rppg,
        prominence=0.5,
    )

    ## Menghitung heart rate
    heart_rate = 60 * len(peaks) / (len(filtered_rppg) / fs)

    ## Menampilkan grafik puncak sinyal
    plt.figure(figsize=(20, 5))
    plt.plot(filtered_rppg, color='black')
    plt.plot(peaks, filtered_rppg[peaks], 'x', color='red')
    plt.title(f'Heart Rate: {heart_rate:.2f}')
    plt.tight_layout()
    plt.show()

    # Menghitung Respiratory Rate
    ## Normalisasi Sinyal
    filtered_respirasi = (filtered_respirasi - np.mean(filtered_respirasi)) / np.std(filtered_respirasi)

    ## Mencari puncak sinyal
    peaks_respirasi, _ = signal.find_peaks(
        x=filtered_respirasi,
        prominence=0.5,
    )

    ## Menghitung respiratory rate
    respiratory_rate = 60 * len(peaks_respirasi) / (len(filtered_respirasi) / fs)

    ## Menampilkan grafik puncak sinyal
    plt.figure(figsize=(20, 5))
    plt.plot(filtered_respirasi, color='black')
    plt.plot(peaks_respirasi, filtered_respirasi[peaks_respirasi], 'x', color='red')
    plt.title(f'Respiratory Rate: {respiratory_rate:.2f}')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
