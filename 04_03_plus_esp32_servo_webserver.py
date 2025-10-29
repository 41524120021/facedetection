import cv2
import time
import requests
import os
import face_recognition
from datetime import datetime

# --- KONFIGURASI ---
VIDEO_URL = "ip_camera"  # IP kamera HP
ESP32_URL = "ip ESP32"       # ganti dengan IP ESP32 kamu
OUTPUT_DIR = "rekaman_wajah"
KNOWN_FACES_DIR = "database_wajah"              # folder wajah yang dikenal
TELEGRAM_BOT_TOKEN = "token_telegram_bot"
TELEGRAM_CHAT_ID = "chat_id_telegram"

# --- Pastikan folder ada ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# --- Load wajah dikenal ---
print("📂 Memuat database wajah...")
known_encodings = []
known_names = []
for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        img_path = os.path.join(KNOWN_FACES_DIR, filename)
        img = face_recognition.load_image_file(img_path)
        enc = face_recognition.face_encodings(img)
        if len(enc) > 0:
            known_encodings.append(enc[0])
            known_names.append(os.path.splitext(filename)[0])
print(f"✅ {len(known_encodings)} wajah dikenal dimuat.\n")

# --- Inisialisasi kamera ---
print("📷 Membuka kamera...")
cap = cv2.VideoCapture(VIDEO_URL)
time.sleep(2)
if not cap.isOpened():
    print("❌ Gagal membuka stream kamera.")
    exit()

print("🚀 Sistem pengenalan wajah aktif! Tekan ESC untuk keluar.\n")

last_detect_time = 0
save_interval = 10  # detik antar rekaman
servo_triggered = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Tidak dapat membaca frame.")
        break

    # Konversi ke RGB untuk face_recognition
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_enc in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_enc, tolerance=0.45)
        name = "Tidak Dikenal"

        if True in matches:
            matched_idx = matches.index(True)
            name = known_names[matched_idx]

            # Tampilkan nama di frame
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            now = time.time()
            if now - last_detect_time > save_interval:
                last_detect_time = now
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                img_path = os.path.join(OUTPUT_DIR, f"{name}_{timestamp}.jpg")
                cv2.imwrite(img_path, frame)
                print(f"✅ {name} dikenali dan disimpan: {img_path}")

                # --- Kirim notifikasi ke Telegram ---
                try:
                    msg = f"🔓 {name} dikenali pada {datetime.now().strftime('%H:%M:%S')}."
                    requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                                  data={"chat_id": TELEGRAM_CHAT_ID, "text": msg})
                    with open(img_path, "rb") as f:
                        requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto",
                                      data={"chat_id": TELEGRAM_CHAT_ID}, files={"photo": f})
                    print("📩 Notifikasi terkirim ke Telegram.")
                except Exception as e:
                    print("⚠️ Gagal kirim ke Telegram:", e)

                # --- Kirim perintah ke ESP32 untuk gerakkan servo ---
                try:
                    response = requests.get(f"{ESP32_URL}?angle=90", timeout=3)
                    print(f"🤖 Servo di ESP32 digerakkan: {response.status_code}")
                    servo_triggered = True
                    time.sleep(3)
                    requests.get(f"{ESP32_URL}?angle=0", timeout=3)
                    servo_triggered = False
                except Exception as e:
                    print("⚠️ Gagal kirim perintah ke ESP32:", e)

        else:
            # Gambar kotak merah untuk wajah tak dikenal
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow("AI Security - Face Recognition", frame)

    if cv2.waitKey(1) == 27:  # ESC untuk keluar
        break

cap.release()
cv2.destroyAllWindows()
print("🔚 Program selesai.")

