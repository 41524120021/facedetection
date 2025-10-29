#include <Arduino.h>
#include <ESP32Servo.h>

Servo servo;
const int SERVO_PIN = 22; // sesuaikan pin PWM
int currentAngle = 0;

void setup() {
  Serial.begin(115200);
  servo.attach(SERVO_PIN);
  servo.write(0);
  Serial.println("✅ ESP32 siap menerima perintah serial dari Jetson.");
}

void loop() {
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();

    if (command.startsWith("SERVO:")) {
      int angle = command.substring(6).toInt();
      angle = constrain(angle, 0, 180);
      servo.write(angle);
      currentAngle = angle;

      Serial.printf("Servo bergerak ke %d derajat\n", angle);
      delay(15); // biar halus
    } else {
      Serial.println("Perintah tidak dikenal.");
    }
  }
}
