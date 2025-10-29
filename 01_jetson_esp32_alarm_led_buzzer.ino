/*
  AI Security ESP32
  ------------------
  Menerima perintah dari Jetson Nano melalui Serial:
    - "ALARM" → LED berkedip + buzzer ON
    - "CLEAR" → LED dan buzzer OFF
*/

#define LED_PIN 2        // Pin LED (bisa pakai LED internal bawaan ESP32)
#define BUZZER_PIN 15    // Pin buzzer aktif (bisa diganti sesuai wiring)

String command = "";
bool alarmActive = false;
unsigned long previousMillis = 0;
const long blinkInterval = 300; // interval LED berkedip (ms)

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  digitalWrite(BUZZER_PIN, LOW);

  Serial.println("✅ ESP32 AI Security siap menerima perintah...");
}

void loop() {
  // Cek apakah ada data masuk dari Jetson
  if (Serial.available()) {
    command = Serial.readStringUntil('\n');
    command.trim(); // hapus spasi atau newline

    if (command == "ALARM") {
      alarmActive = true;
      Serial.println("🚨 Perintah ALARM diterima!");
    } 
    else if (command == "CLEAR") {
      alarmActive = false;
      Serial.println("✅ Perintah CLEAR diterima.");
      digitalWrite(LED_PIN, LOW);
      digitalWrite(BUZZER_PIN, LOW);
    }
  }

  // Jika alarm aktif → LED berkedip dan buzzer hidup
  if (alarmActive) {
    unsigned long currentMillis = millis();

    // Buzzer ON selama alarm aktif
    digitalWrite(BUZZER_PIN, HIGH);

    // LED berkedip setiap 300 ms
    if (currentMillis - previousMillis >= blinkInterval) {
      previousMillis = currentMillis;
      int ledState = digitalRead(LED_PIN);
      digitalWrite(LED_PIN, !ledState);
    }
  }
}
