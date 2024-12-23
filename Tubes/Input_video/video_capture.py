import cv2
import mediapipe as mp
import numpy as np

def capture_video():
    # Inisialisasi MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    # Memuat video dari webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return None, None, None, None

    # Mempersiapkan beberapa variabel
    r_signal, g_signal, b_signal = [], [], []
    f_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Variabel untuk kontrol mulai dan berhentis
    recording = False

    # Teks instruksi
    start_text = "Press 's' to start recording"
    stop_text = "Press 'q' to stop recording"
    error_text = "Error: Camera not working properly"

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
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

        # Menampilkan instruksi
        if not recording:
            cv2.putText(frame, start_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, stop_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Menampilkan frame dengan bounding box
        cv2.imshow('frame', frame)

        # Kontrol tombol
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            recording = True
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if not r_signal or not g_signal or not b_signal:
        print("No video recorded or camera not working properly")
        return None, None, None, None

    return r_signal, g_signal, b_signal, fps
