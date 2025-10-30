#include <Arduino.h>
#include <ESP32Servo.h>

Servo servo;
const int SERVO_PIN = 22;
const int BUZZER_PIN = 15;
const int LED_PIN = 2;

void setup() {
  Serial.begin(115200);
  servo.attach(SERVO_PIN);
  servo.write(0);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);
  digitalWrite(LED_PIN, LOW);
  Serial.println("✅ ESP32 siap menerima perintah serial dari Jetson.");
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd.startsWith("SERVO:")) {
      int angle = cmd.substring(6).toInt();
      angle = constrain(angle, 0, 180);
      servo.write(angle);
      Serial.printf("Servo bergerak ke %d°\n", angle);
    } 
    else if (cmd == "BUZZER:ON") {
      digitalWrite(BUZZER_PIN, HIGH);
      digitalWrite(LED_PIN, HIGH);
      Serial.println("Buzzer ON");
      Serial.println("LED ON");
    } 
    else if (cmd == "BUZZER:OFF") {
      digitalWrite(BUZZER_PIN, LOW);
      digitalWrite(LED_PIN, LOW);
      Serial.println("Buzzer OFF");
      Serial.println("LED OFF");
    } 
    else {
      Serial.println("Perintah tidak dikenal.");
    }
  }
}
