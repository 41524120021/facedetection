import cv2
import time
import requests
import os
import serial
import face_recognition
from datetime import datetime
import threading
import numpy as np
from collections import deque

# ============================================================================
# KONFIGURASI
# ============================================================================
VIDEO_URL = 0
OUTPUT_DIR = "rekaman_wajah"
KNOWN_FACES_DIR = "database_wajah"
LOG_DIR = os.path.join(os.path.dirname(__file__) if __file__ else ".", "logs")

# Generate log filename per tanggal
def get_log_file():
    """Generate log filename dengan format: system_YYYY-MM-DD.log"""
    date_str = datetime.now().strftime("%Y-%m-%d")
    return os.path.join(LOG_DIR, f"system_{date_str}.log")

LOG_FILE = get_log_file()  # Will update daily automatically

TELEGRAM_BOT_TOKEN = "token_bot_telegram"
TELEGRAM_CHAT_ID = "your_chat_ID"
SERIAL_PORT = "COM3"
BAUD_RATE = 115200

# Performance settings
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 540
PROCESS_SCALE = 0.25  # Process at 25% size for speed
PROCESS_EVERY_N_FRAMES = 2  # Process every 2 frames

# Telegram rate limiting
TELEGRAM_COOLDOWN = 60  # Kirim telegram max 1x per 60 detik per orang
SAVE_INTERVAL = 8  # Save foto setiap 8 detik

# ============================================================================
# PERSIAPAN FOLDER
# ============================================================================
for folder in [OUTPUT_DIR, KNOWN_FACES_DIR, LOG_DIR]:
    os.makedirs(folder, exist_ok=True)

# ============================================================================
# GLOBAL VARIABLES
# ============================================================================
known_encodings = []
known_names = []
last_faces = []  # Store last detected faces for display
current_name_input = ""
input_mode = False
fps_queue = deque(maxlen=30)
esp32 = None

# ============================================================================
# LOGGING
# ============================================================================
def write_log(event, level="INFO"):
    """Write log dengan file per tanggal (auto-rotate daily)"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_line = f"[{timestamp}] [{level:7s}] {event}\n"
    try:
        # Generate current log file (akan berubah otomatis setiap hari)
        current_log_file = get_log_file()
        
        with open(current_log_file, "a", encoding="utf-8") as f:
            f.write(log_line)
            f.flush()
            os.fsync(f.fileno())
    except Exception as e:
        print(f"❌ Log error: {e}")

# ============================================================================
# DATABASE WAJAH
# ============================================================================
def load_known_faces():
    global known_encodings, known_names
    known_encodings, known_names = [], []
    print("📂 Loading face database...")
    
    if not os.path.exists(KNOWN_FACES_DIR):
        print(f"⚠️  Directory not found: {KNOWN_FACES_DIR}")
        return
    
    files = [f for f in os.listdir(KNOWN_FACES_DIR) 
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for i, filename in enumerate(files, 1):
        try:
            path = os.path.join(KNOWN_FACES_DIR, filename)
            img = face_recognition.load_image_file(path)
            encs = face_recognition.face_encodings(img)
            
            if len(encs) > 0:
                known_encodings.append(encs[0])
                name = os.path.splitext(filename)[0].split("_")[0]
                known_names.append(name)
                print(f"   ✅ [{i}/{len(files)}] {name}")
            else:
                print(f"   ⚠️  [{i}/{len(files)}] No face in {filename}")
        except Exception as e:
            print(f"   ❌ Error loading {filename}: {e}")
    
    print(f"✅ {len(known_encodings)} faces loaded\n")
    write_log(f"Loaded {len(known_encodings)} known faces", "INFO")

# ============================================================================
# ESP32 CONNECTION
# ============================================================================
def connect_esp32():
    global esp32
    try:
        esp32 = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        print(f"✅ Connected to ESP32 ({SERIAL_PORT})")
        write_log(f"Connected to ESP32 on {SERIAL_PORT}", "SUCCESS")
        return True
    except Exception as e:
        print(f"⚠️  ESP32 connection failed: {e}")
        write_log(f"ESP32 connection failed: {e}", "WARNING")
        esp32 = None
        return False

def control_esp32(command):
    if esp32:
        try:
            esp32.write((command + "\n").encode())
            write_log(f"ESP32 command: {command}", "INFO")
        except Exception as e:
            print(f"⚠️  ESP32 error: {e}")

# ============================================================================
# TELEGRAM
# ============================================================================
def send_to_telegram(message, image_path=None):
    def _send():
        try:
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                data={"chat_id": TELEGRAM_CHAT_ID, "text": message},
                timeout=5
            )
            if image_path and os.path.exists(image_path):
                with open(image_path, "rb") as f:
                    requests.post(
                        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto",
                        data={"chat_id": TELEGRAM_CHAT_ID},
                        files={"photo": f},
                        timeout=5
                    )
            write_log(f"Telegram sent: {message[:50]}", "INFO")
        except Exception as e:
            write_log(f"Telegram error: {e}", "WARNING")
    
    # Run in thread to avoid blocking
    threading.Thread(target=_send, daemon=True).start()

# ============================================================================
# FACE RECOGNITION THREAD
# ============================================================================
class FaceRecognitionThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.frame_to_process = None
        self.result_faces = []
        self.lock = threading.Lock()
        self.running = True
        self.last_detect_time = {}
        self.last_telegram_time = {}  # Track last telegram send per person
        self.save_interval = SAVE_INTERVAL
        self.telegram_interval = TELEGRAM_COOLDOWN
        
    def set_frame(self, frame):
        with self.lock:
            self.frame_to_process = frame.copy()
    
    def get_faces(self):
        with self.lock:
            return self.result_faces.copy()
    
    def run(self):
        while self.running:
            frame = None
            with self.lock:
                if self.frame_to_process is not None:
                    frame = self.frame_to_process
                    self.frame_to_process = None
            
            if frame is not None:
                try:
                    # Process at smaller resolution
                    small = cv2.resize(frame, (0, 0), fx=PROCESS_SCALE, fy=PROCESS_SCALE)
                    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                    
                    face_locations = face_recognition.face_locations(rgb_small, model="hog")
                    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)
                    
                    detected_faces = []
                    scale_factor = int(1 / PROCESS_SCALE)
                    
                    for (top, right, bottom, left), enc in zip(face_locations, face_encodings):
                        matches = face_recognition.compare_faces(known_encodings, enc, tolerance=0.5)
                        face_distances = face_recognition.face_distance(known_encodings, enc)
                        
                        name = "Tidak Dikenal"
                        confidence = 0
                        
                        if len(face_distances) > 0 and True in matches:
                            best_match_index = np.argmin(face_distances)
                            if matches[best_match_index]:
                                name = known_names[best_match_index]
                                confidence = (1 - face_distances[best_match_index]) * 100
                                
                                # Handle access grant
                                now = time.time()
                                if name not in self.last_detect_time or \
                                   now - self.last_detect_time[name] > self.save_interval:
                                    self.last_detect_time[name] = now
                                    self.handle_recognized_face(name, frame)
                        else:
                            # Unknown face
                            self.handle_unknown_face(frame)
                        
                        # Scale coordinates back
                        detected_faces.append({
                            'name': name,
                            'confidence': confidence,
                            'box': (left * scale_factor, top * scale_factor,
                                   right * scale_factor, bottom * scale_factor)
                        })
                    
                    with self.lock:
                        self.result_faces = detected_faces
                        
                except Exception as e:
                    print(f"❌ Recognition error: {e}")
                    write_log(f"Recognition error: {e}", "ERROR")
            else:
                time.sleep(0.01)
    
    def handle_recognized_face(self, name, frame):
        threading.Thread(target=self._handle_recognized, args=(name, frame.copy()), daemon=True).start()
    
    def _handle_recognized(self, name, frame):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_path = os.path.join(OUTPUT_DIR, f"{name}_{timestamp}.jpg")
        cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        
        # Check if we should send telegram (batasi per orang)
        now = time.time()
        should_send_telegram = False
        
        if name not in self.last_telegram_time:
            should_send_telegram = True
            self.last_telegram_time[name] = now
        elif now - self.last_telegram_time[name] > self.telegram_interval:
            should_send_telegram = True
            self.last_telegram_time[name] = now
        
        if should_send_telegram:
            send_to_telegram(f"🔓 Access granted: {name}\n{timestamp}", img_path)
            print(f"✅ {name} recognized! (Telegram sent)")
        else:
            time_left = int(self.telegram_interval - (now - self.last_telegram_time[name]))
            print(f"✅ {name} recognized! (Telegram cooldown: {time_left}s)")
        
        control_esp32("SERVO:90")
        control_esp32("BUZZER:ON")
        time.sleep(2)
        control_esp32("BUZZER:OFF")
        control_esp32("SERVO:0")
        
        write_log(f"{name} recognized and granted access", "SUCCESS")
    
    def handle_unknown_face(self, frame):
        threading.Thread(target=self._handle_unknown, args=(frame.copy(),), daemon=True).start()
    
    def _handle_unknown(self, frame):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_path = os.path.join(OUTPUT_DIR, f"unknown_{timestamp}.jpg")
        cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        # Check if we should send telegram for unknown (batasi spam)
        now = time.time()
        should_send_telegram = False
        
        if "unknown" not in self.last_telegram_time:
            should_send_telegram = True
            self.last_telegram_time["unknown"] = now
        elif now - self.last_telegram_time["unknown"] > self.telegram_interval:
            should_send_telegram = True
            self.last_telegram_time["unknown"] = now
        
        if should_send_telegram:
            send_to_telegram(f"🚨 Unknown face detected!\n{timestamp}", img_path)
            print(f"⚠️  Unknown face detected! (Telegram sent)")
        else:
            time_left = int(self.telegram_interval - (now - self.last_telegram_time["unknown"]))
            print(f"⚠️  Unknown face detected! (Telegram cooldown: {time_left}s)")
        
        control_esp32("BUZZER:ON")
        time.sleep(1)
        control_esp32("BUZZER:OFF")
        
        write_log("Unknown face detected", "WARNING")
    
    def stop(self):
        self.running = False

# ============================================================================
# UI DRAWING FUNCTIONS
# ============================================================================
def draw_overlay_panel(frame, faces, fps, known_count, telegram_cooldown_info=""):
    """Draw modern UI overlay with info panel"""
    h, w = frame.shape[:2]
    
    # Top bar (semi-transparent)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (30, 30, 30), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    
    # Bottom bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h-50), (w, h), (30, 30, 30), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    
    # Top info
    cv2.putText(frame, "AI SECURITY SYSTEM", (10, 25), 
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Telegram cooldown info
    if telegram_cooldown_info:
        cv2.putText(frame, telegram_cooldown_info, (10, 75), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
    
    # Known faces count
    cv2.putText(frame, f"Known: {known_count}", (w-150, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, datetime.now().strftime("%H:%M:%S"), (w-150, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    # Telegram rate limit info
    cv2.putText(frame, f"Telegram: {TELEGRAM_COOLDOWN}s cooldown", (w-230, 75), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
    
    # Bottom controls
    cv2.putText(frame, "[ESC] Exit  |  [N] Add Face  |  [R] Reload DB", 
                (10, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)
    
    return frame

def draw_face_box(frame, face):
    """Draw face box with name and confidence"""
    left, top, right, bottom = face['box']
    name = face['name']
    confidence = face['confidence']
    
    # Color based on known/unknown
    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
    
    # Draw box
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
    
    # Draw label background
    label = f"{name}"
    if confidence > 0:
        label += f" ({confidence:.0f}%)"
    
    (text_width, text_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1
    )
    
    cv2.rectangle(frame, (left, top-30), (left+text_width+10, top), color, -1)
    cv2.putText(frame, label, (left+5, top-8), 
                cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

def draw_input_overlay(frame, current_text):
    """Draw input overlay for adding new face"""
    h, w = frame.shape[:2]
    
    # Semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (w//4, h//2-60), (3*w//4, h//2+60), (50, 50, 50), -1)
    frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)
    
    # Border
    cv2.rectangle(frame, (w//4, h//2-60), (3*w//4, h//2+60), (0, 255, 255), 2)
    
    # Title
    cv2.putText(frame, "Enter Name:", (w//4+20, h//2-30), 
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
    
    # Input text
    cv2.putText(frame, current_text + "_", (w//4+20, h//2+10), 
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)
    
    # Instructions
    cv2.putText(frame, "[ENTER] Save  |  [ESC] Cancel", (w//4+20, h//2+45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    return frame

# ============================================================================
# MAIN
# ============================================================================
def list_log_files():
    """List all log files in LOG_DIR"""
    try:
        log_files = sorted([f for f in os.listdir(LOG_DIR) if f.startswith("system_") and f.endswith(".log")])
        return log_files
    except Exception:
        return []

def view_log_file(log_filename=None, lines=50):
    """View contents of a log file (default: today's log, last 50 lines)"""
    if log_filename is None:
        log_filename = os.path.basename(get_log_file())
    
    log_path = os.path.join(LOG_DIR, log_filename)
    
    if not os.path.exists(log_path):
        print(f"\n❌ Log file not found: {log_filename}")
        return
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
            
        # Get last N lines
        display_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
        
        print("\n" + "="*70)
        print(f"📋 Log File: {log_filename}")
        print(f"   Total Lines: {len(all_lines)} | Showing Last: {len(display_lines)}")
        print("="*70)
        
        for line in display_lines:
            print(line.rstrip())
        
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error reading log file: {e}\n")

def main():
    global current_name_input, input_mode, last_faces
    
    print("\n" + "="*70)
    print("🚀 AI SECURITY - FACE RECOGNITION SYSTEM")
    print("="*70)
    
    # Show log file info
    current_log = get_log_file()
    print(f"📋 Log file: {os.path.basename(current_log)}")
    
    # Show available log files
    log_files = list_log_files()
    if log_files:
        print(f"📂 Available logs: {len(log_files)} files")
        if len(log_files) <= 5:
            for lf in log_files:
                print(f"   - {lf}")
        else:
            print(f"   - {log_files[0]} ... {log_files[-1]}")
    
    print("="*70 + "\n")
    
    # Load database
    load_known_faces()
    
    # Connect ESP32
    connect_esp32()
    
    # Open camera
    print("\n📹 Opening camera...")
    cap = cv2.VideoCapture(VIDEO_URL)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print("❌ Camera failed to open!")
        return
    
    print("✅ Camera opened successfully")
    write_log("System started", "SUCCESS")
    
    # Start face recognition thread
    face_thread = FaceRecognitionThread()
    face_thread.start()
    
    print("\n" + "="*70)
    print("Controls:")
    print("  ESC - Exit")
    print("  N   - Add face (auto-select largest/closest face)")
    print("        • Known face → Auto-save training photo")
    print("        • Unknown face → Enter name manually")
    print("  R   - Reload database")
    print("  L   - List all log files")
    print("  V   - View today's log")
    print("="*70 + "\n")
    
    frame_count = 0
    prev_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️  Empty frame, retrying...")
                time.sleep(0.1)
                continue
            
            # Resize for display
            display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            
            # Send frame to recognition thread every N frames
            if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                face_thread.set_frame(frame)
            
            # Get recognition results
            last_faces = face_thread.get_faces()
            
            # Calculate FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
            prev_time = curr_time
            fps_queue.append(fps)
            avg_fps = sum(fps_queue) / len(fps_queue)
            
            # Get telegram cooldown info
            telegram_info = ""
            if hasattr(face_thread, 'last_telegram_time') and face_thread.last_telegram_time:
                cooldowns = []
                curr_time = time.time()
                for person, last_time in face_thread.last_telegram_time.items():
                    remaining = TELEGRAM_COOLDOWN - (curr_time - last_time)
                    if remaining > 0:
                        cooldowns.append(f"{person}: {int(remaining)}s")
                if cooldowns:
                    telegram_info = "Cooldown: " + ", ".join(cooldowns[:2])  # Show max 2
            
            # Draw UI
            display_frame = draw_overlay_panel(
                display_frame, last_faces, avg_fps, len(known_names), telegram_info
            )
            
            # Draw face boxes
            scale_x = DISPLAY_WIDTH / FRAME_WIDTH
            scale_y = DISPLAY_HEIGHT / FRAME_HEIGHT
            
            for face in last_faces:
                scaled_face = face.copy()
                left, top, right, bottom = face['box']
                scaled_face['box'] = (
                    int(left * scale_x), int(top * scale_y),
                    int(right * scale_x), int(bottom * scale_y)
                )
                draw_face_box(display_frame, scaled_face)
            
            # Input mode overlay
            if input_mode:
                display_frame = draw_input_overlay(display_frame, current_name_input)
            
            # Show frame
            cv2.imshow("AI Security System", display_frame)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            
            if input_mode:
                if key == 27:  # ESC - cancel
                    input_mode = False
                    current_name_input = ""
                    print("❌ Cancelled")
                elif key == 13:  # ENTER - save
                    if current_name_input.strip() and last_faces:
                        name = current_name_input.strip()
                        
                        # Ambil wajah terbesar (terdekat dengan kamera)
                        largest_face = max(last_faces, key=lambda f: (f['box'][2] - f['box'][0]) * (f['box'][3] - f['box'][1]))
                        left, top, right, bottom = largest_face['box']
                        
                        # Crop wajah dengan margin
                        margin = 20
                        h, w = frame.shape[:2]
                        top = max(0, top - margin)
                        bottom = min(h, bottom + margin)
                        left = max(0, left - margin)
                        right = min(w, right + margin)
                        
                        face_img = frame[top:bottom, left:right]
                        
                        # Save
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        save_path = os.path.join(KNOWN_FACES_DIR, f"{name}_{timestamp}.jpg")
                        cv2.imwrite(save_path, face_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        print(f"✅ Saved: {name} (largest face selected)")
                        send_to_telegram(f"📚 New face added: {name}")
                        write_log(f"New face added: {name}", "SUCCESS")
                        load_known_faces()
                    input_mode = False
                    current_name_input = ""
                elif key == 8:  # BACKSPACE
                    current_name_input = current_name_input[:-1]
                elif 32 <= key <= 126:  # Printable characters
                    current_name_input += chr(key)
            else:
                if key == 27:  # ESC - exit
                    print("\n⏹️  Exiting...")
                    break
                elif key == ord('n') or key == ord('N'):
                    # Improvement 1: Hanya ambil wajah terbesar (terdekat dengan kamera)
                    if not last_faces:
                        print("⚠️  No face detected! Please face the camera.")
                        continue
                    
                    # Cari wajah terbesar berdasarkan luas box
                    largest_face = max(last_faces, key=lambda f: (f['box'][2] - f['box'][0]) * (f['box'][3] - f['box'][1]))
                    
                    # Improvement 2: Jika wajah sudah dikenal, langsung save tanpa input nama
                    if largest_face['name'] != "Tidak Dikenal":
                        # Wajah sudah dikenal, langsung save dengan nama yang ada
                        name = largest_face['name']
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        save_path = os.path.join(KNOWN_FACES_DIR, f"{name}_{timestamp}.jpg")
                        cv2.imwrite(save_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        print(f"✅ Added training photo for: {name} (confidence: {largest_face['confidence']:.1f}%)")
                        send_to_telegram(f"📸 Training photo added for: {name}")
                        write_log(f"Training photo added: {name} (confidence: {largest_face['confidence']:.1f}%)", "SUCCESS")
                        load_known_faces()
                    else:
                        # Wajah belum dikenal, minta input nama
                        input_mode = True
                        current_name_input = ""
                        print("✏️  Enter name for new face (type in video window)...")
                elif key == ord('r') or key == ord('R'):
                    print("🔄 Reloading database...")
                    load_known_faces()
                elif key == ord('l') or key == ord('L'):
                    log_files = list_log_files()
                    print("\n" + "="*70)
                    print(f"📋 Available Log Files ({len(log_files)}):")
                    print("="*70)
                    if log_files:
                        for i, log_file in enumerate(log_files, 1):
                            file_path = os.path.join(LOG_DIR, log_file)
                            file_size = os.path.getsize(file_path) / 1024  # KB
                            print(f"  {i}. {log_file} ({file_size:.1f} KB)")
                    else:
                        print("  No log files found")
                    print("="*70 + "\n")
                elif key == ord('v') or key == ord('V'):
                    view_log_file()
            
            frame_count += 1
            
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🛑 Shutting down...")
        face_thread.stop()
        face_thread.join(timeout=2)
        cap.release()
        cv2.destroyAllWindows()
        if esp32:
            esp32.close()
        write_log("System stopped", "INFO")
        print("🔚 System stopped\n")

if __name__ == "__main__":
    main()

