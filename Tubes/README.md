# Tugas Mata Kuliah Pengolahan Sinyal Digital (IF3024)
Dosen Pengampu :  Martin Clinton Tosima Manullang S.T., M.T.

## Deskripsi Proyek
Proyek ini mengembangkan sistem real-time yang menggabungkan pengukuran sinyal respirasi dan remote-photopletysmography (rPPG). Sistem ini menggunakan webcam untuk menangkap video dan memproses secara langsung untuk menghasilkan visualisasi sinyal respirasi dan rPPG. Implementasi menggunakan teknik pemrosesan sinyal digital untuk ekstraksi data vital secara non-invasif.

## Fitur Utama
- Pengambilan video real-time melalui webcam
- Pemrosesan sinyal respirasi secara langsung
- Analisis rPPG untuk pengukuran detak jantung
- Visualisasi data menggunakan matplotlib dan OpenCV

## Kebutuhan Sistem
- Webcam (disarankan memiliki resolusi minimal 720p)
- Pencahayaan yang cukup baik
- Laptop/PC

## Teknologi
- Python
- OpenCV untuk pengolahan video
- Matplotlib untuk visualisasi data
- NumPy untuk pemrosesan sinyal

## Informansi Anggota
1. Heni Artha Uli br Turnip (121140080), ID : heniturnip
2. Pannes Diba Sabila (121140117), ID : Diba-sabila

## Logbook Progress

| Nama | Tanggal | Progres |
|------|----------|---------|
| Heni Artha Uli br Turnip | 2023-12-13 | Inisialisasi proyek dan membuat dasar proses sinyal rPPG |
| Pannes Diba Sabila | 2023-12-17 | Membuat visualisasi sinyal rPPG|
| Pannes Diba Sabila | 2023-12-22 | Update visualisasi sinyal rPPG|
| Heni Artha Uli br Turnip | 2023-12-22 | Menambahkan fitur webcam|
| Pannes Diba Sabila | 2023-12-23 | Membuar requirements.txt dan menambahkan frame record video|
| Heni Artha Uli br Turnip | 2023-12-24 | Membuat file readme dan file python proses rppg, heart rate, dan main program |
| Pannes Diba Sabila | 2023-12-24 | Membuat dokumen Report |

## Instruksi Instalasi dan Penggunaan Program
### Prasyarat
- Python
- Anaconda/conda (opsional)

### Langkah - Langkah Instalasi
1. Download folder Tubes
2. Membuat environment (opsional)
3. Instal Library:
   - Pastikan Anda berada di dalam direktori proyek dan enviroment yang dibuat (jika digunakan).
   - Instal semua library yang diperlukan dengan menjalankan: `pip install -r requirements.txt`
   catatan :  Jika code tidak berjalan, Anda dapat menginstal paket yang diperlukan secara manual dengan perintah berikut: `pip install numpy matplotlib scipy opencv-python mediapipe`
4. Jika instalasi library selesai, Program sudah dapat dijalankan

### Langkah - Langkah pengguanaan Program
1. Jalankan program file `main.py`
2. Izinkan Akses Kamera
   - Program akan meminta akses ke kamera Anda. Pastikan untuk mengizinkannya agar program dapat menangkap video.
3. Deteksi Wajah dan Pengambilan Sinyal
   - Program akan mulai mendeteksi wajah dan mengambil sinyal RGB dari area wajah yang terdeteksi.
   - Anda akan melihat jendela yang menampilkan video dari kamera dengan bounding box di sekitar wajah yang terdeteksi.
   - Tekan `s` untuk merekam video dan `q` untuk berhenti merekam. 
4. Tunggu Hasil
   - Setelah beberapa detik, program akan menghitung detak jantung dan laju pernapasan berdasarkan sinyal yang diambil dari video yang sudah direkam.
   - Grafik yang menunjukkan sinyal RGB, sinyal rPPG, dan puncak detak jantung serta laju pernapasan akan ditampilkan.

### Catatan
- Pastikan Anda menjalankan program di lingkungan yang memiliki pencahayaan yang baik untuk meningkatkan akurasi deteksi wajah.
- Jika Anda mengalami masalah, periksa kembali langkah-langkah instalasi dan pastikan semua dependensi terinstal dengan benar.


