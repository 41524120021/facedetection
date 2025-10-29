import cv2
import time
import requests
import os
import serial
import face_recognition
from datetime import datetime

# --- KONFIGURASI ---
VIDEO_URL = "http://192.168.100.15:8080/video"   # IP Webcam Android
OUTPUT_DIR = "rekaman_wajah"
KNOWN_FACES_DIR = "database_wajah"
LOG_FILE = "log.txt"
TELEGRAM_BOT_TOKEN = "8359492075:AAGZBZrgZQU45aaXk8IGvRBjnTlP1BZ92ek"
TELEGRAM_CHAT_ID = "1048144404"
SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 115200

# --- PERSIAPAN FOLDER ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# --- LOAD DATABASE WAJAH ---
def load_known_faces():
    known_encodings, known_names = [], []
    print("📂 Memuat database wajah...")
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(KNOWN_FACES_DIR, filename)
            img = face_recognition.load_image_file(path)
            encs = face_recognition.face_encodings(img)
            if len(encs) > 0:
                known_encodings.append(encs[0])
                name = os.path.splitext(filename)[0].split("_")[0]
                known_names.append(name)
    print(f"✅ {len(known_encodings)} wajah dikenal dimuat.\n")
    return known_encodings, known_names

known_encodings, known_names = load_known_faces()

# --- LOGGING ---
def write_log(event):
    with open(LOG_FILE, "a") as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {event}\n")

# --- SERIAL ESP32 ---
esp32 = None
try:
    esp32 = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    print(f"🔌 Terhubung ke ESP32 ({SERIAL_PORT})")
except Exception as e:
    print("⚠️ Tidak bisa membuka port serial:", e)

# --- KAMERA ---
cap = cv2.VideoCapture(VIDEO_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
time.sleep(1)
if not cap.isOpened():
    print("❌ Kamera gagal dibuka.")
    exit()

print("🚀 Sistem pengenalan wajah aktif! Tekan 'n' untuk tambah wajah baru.\n")

last_detect_time = 0
save_interval = 8

def send_to_telegram(message, image_path=None):
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                      data={"chat_id": TELEGRAM_CHAT_ID, "text": message}, timeout=3)
        if image_path:
            with open(image_path, "rb") as f:
                requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto",
                              data={"chat_id": TELEGRAM_CHAT_ID}, files={"photo": f}, timeout=3)
    except Exception as e:
        print("⚠️ Gagal kirim ke Telegram:", e)

def control_esp32(command):
    if esp32:
        try:
            esp32.write((command + "\n").encode())
            print(f"📡 Perintah dikirim ke ESP32: {command}")
        except Exception as e:
            print("⚠️ Serial error:", e)

# --- LOOP UTAMA ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame kosong.")
        time.sleep(1)
        continue

    small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_small)
    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

    for (top, right, bottom, left), enc in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, enc, tolerance=0.55)
        name = "Tidak Dikenal"

        if True in matches:
            name = known_names[matches.index(True)]
            now = time.time()
            if now - last_detect_time > save_interval:
                last_detect_time = now
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                img_path = os.path.join(OUTPUT_DIR, f"{name}_{timestamp}.jpg")
                cv2.imwrite(img_path, frame)

                # Kirim Telegram + ESP32 (servo+buzzer)
                send_to_telegram(f"🔓 Akses dibuka untuk {name} ({timestamp})", img_path)
                control_esp32("SERVO:90")
                control_esp32("BUZZER:ON")
                time.sleep(15)
                control_esp32("BUZZER:OFF")
                control_esp32("SERVO:0")

                write_log(f"{name} dikenali dan diberi akses.")
                print(f"✅ {name} dikenali!")

        else:
            # Wajah tidak dikenal
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_path = os.path.join(OUTPUT_DIR, f"unknown_{timestamp}.jpg")
            cv2.imwrite(img_path, frame)
            send_to_telegram(f"🚨 Wajah tidak dikenal terdeteksi! ({timestamp})", img_path)
            write_log("Wajah tidak dikenal terdeteksi.")
            control_esp32("BUZZER:ON")
            time.sleep(15)
            control_esp32("BUZZER:OFF")

        # Gambar kotak di frame
        top, right, bottom, left = top*2, right*2, bottom*2, left*2
        color = (0, 255, 0) if name != "Tidak Dikenal" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("AI Security - Face Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC keluar
        break
    elif key == ord('n'):  # tambah wajah baru
        print("🧠 Mode tambah wajah baru.")
        name = input("Masukkan nama orang baru: ").strip()
        if name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(KNOWN_FACES_DIR, f"{name}_{timestamp}.jpg")
            cv2.imwrite(save_path, frame)
            print(f"📸 Disimpan: {save_path}")
            send_to_telegram(f"📚 Wajah baru '{name}' ditambahkan ke database.")
            write_log(f"Wajah baru {name} ditambahkan ke database.")
            known_encodings, known_names = load_known_faces()

cap.release()
cv2.destroyAllWindows()
print("🔚 Sistem berhenti.")

