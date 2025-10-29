import cv2
import serial
import time

VIDEO_URL = "http://192.168.100.15:8080/video"
SERIAL_PORT = "/dev/ttyUSB0"
BAUDRATE = 115200

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

try:
    ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=2)
    time.sleep(2)
except:
    ser = None
    print("⚠️ ESP32 tidak ditemukan.")

cap = cv2.VideoCapture(VIDEO_URL)
if not cap.isOpened():
    print("❌ Gagal membuka kamera.")
    exit()

last_status = "CLEAR"
print("🚀 Sistem deteksi wajah aktif! Tekan ESC untuk keluar.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    detected = len(faces) > 0
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    if detected and last_status != "ALARM":
        print("🚨 Wajah terdeteksi! Mengirim ALARM.")
        if ser:
            ser.write(b"ALARM\n")
        last_status = "ALARM"

    elif not detected and last_status != "CLEAR":
        print("✅ Tidak ada wajah, mengirim CLEAR.")
        if ser:
            ser.write(b"CLEAR\n")
        last_status = "CLEAR"

    cv2.imshow("AI Security - Face Detection", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
if ser:
    ser.close()
cv2.destroyAllWindows()

