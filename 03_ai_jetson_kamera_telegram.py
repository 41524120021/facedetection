import cv2
import time
import requests
import os
from datetime import datetime

# --- KONFIGURASI ---
VIDEO_URL = "http://192.168.100.15:8080/video"  # ganti dengan IP kamera HP kamu
OUTPUT_DIR = "rekaman_wajah"
TELEGRAM_BOT_TOKEN = "8359492075:AAGZBZrgZQU45aaXk8IGvRBjnTlP1BZ92ek"
TELEGRAM_CHAT_ID = "1048144404"

# --- Pastikan folder penyimpanan ada ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load detektor wajah (Haar Cascade OpenCV) ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- Inisialisasi Kamera ---
print("📷 Membuka kamera HP...")
cap = cv2.VideoCapture(VIDEO_URL)
time.sleep(2)

if not cap.isOpened():
    print("❌ Gagal membuka stream kamera.")
    exit()

print("🚀 Sistem deteksi wajah siap! Tekan ESC untuk keluar.\n")

recording = False
out = None
last_detect_time = 0
save_interval = 10  # detik minimal antar rekaman

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Gagal membaca frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Jika ada wajah terdeteksi
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_img = frame[y:y + h, x:x + w]

            now = time.time()
            if now - last_detect_time > save_interval:
                last_detect_time = now
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                img_path = os.path.join(OUTPUT_DIR, f"wajah_{timestamp}.jpg")
                video_path = os.path.join(OUTPUT_DIR, f"rekaman_{timestamp}.mp4")

                # Simpan gambar wajah
                cv2.imwrite(img_path, face_img)
                print(f"📸 Wajah disimpan: {img_path}")

                # Kirim notifikasi ke Telegram
                try:
                    message = f"👤 Wajah terdeteksi pada {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}"
                    url_msg = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
                    requests.post(url_msg, data={"chat_id": TELEGRAM_CHAT_ID, "text": message})

                    url_photo = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
                    with open(img_path, "rb") as f:
                        requests.post(url_photo, data={"chat_id": TELEGRAM_CHAT_ID}, files={"photo": f})
                    print("📩 Notifikasi Telegram terkirim.")
                except Exception as e:
                    print("⚠️ Gagal mengirim ke Telegram:", e)

                # Mulai rekam video singkat (5 detik)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(video_path, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
                start_time = time.time()
                recording = True
                print("🎥 Mulai merekam video...")

    # Jika sedang merekam
    if recording:
        out.write(frame)
        if time.time() - start_time >= 5:
            recording = False
            out.release()
            print("💾 Rekaman selesai disimpan.\n")

    cv2.imshow("AI Security - Face Detection", frame)

    if cv2.waitKey(1) == 27:  # ESC untuk keluar
        break

cap.release()
if out:
    out.release()
cv2.destroyAllWindows()
print("🔚 Program selesai.")

