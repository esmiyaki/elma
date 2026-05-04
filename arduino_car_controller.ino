#include <Servo.h>

// --- Serial protocol ---
// Raspberry Pi sends:
//   "CMD <servo_deg> <throttle>\n"
//     servo_deg: float degrees
//     throttle:  int in [-255..255]  (+ forward, - reverse)
//   "STOP\n"
//
// Arduino replies (optional):
//   "OK\n"

// --- Steering servo constraints (must match rpi_stanley_controller.py) ---
static const int SERVO_PIN = 11;
static const float SERVO_CENTER_DEG = 80.0f;   // straight-ahead command
static const float SERVO_MIN_DEG = 55.0f;    // max left (mechanical)
static const float SERVO_MAX_DEG = 105.0f;   // max right (mechanical)
static const int SERVO_MIN_US = 500;   // adjust if needed
static const int SERVO_MAX_US = 2500;  // adjust if needed

// --- DC motor driver (assume L298N/L293D style) ---
// Adjust pins to your wiring.
static const int MOTOR_EN_PWM = 5;  // PWM pin
static const int MOTOR_IN1 = 2;
static const int MOTOR_IN2 = 8;

// --- Safety ---
static const unsigned long CMD_TIMEOUT_MS = 400;  // stop if no command arrives

Servo steering;
unsigned long lastCmdMs = 0;

static float clampf(float x, float lo, float hi) {
  if (x < lo) return lo;
  if (x > hi) return hi;
  return x;
}

static int clampi(int x, int lo, int hi) {
  if (x < lo) return lo;
  if (x > hi) return hi;
  return x;
}

static void motorStop() {
  analogWrite(MOTOR_EN_PWM, 0);
  digitalWrite(MOTOR_IN1, LOW);
  digitalWrite(MOTOR_IN2, LOW);
}

static void motorDrive(int throttle) {
  throttle = clampi(throttle, -255, 255);
  int pwm = abs(throttle);
  if (pwm < 10) {  // deadband
    motorStop();
    return;
  }
  if (throttle >= 0) {
    digitalWrite(MOTOR_IN1, HIGH);
    digitalWrite(MOTOR_IN2, LOW);
  } else {
    digitalWrite(MOTOR_IN1, LOW);
    digitalWrite(MOTOR_IN2, HIGH);
  }
  analogWrite(MOTOR_EN_PWM, pwm);
}

static void setSteeringDeg(float deg) {
  deg = clampf(deg, SERVO_MIN_DEG, SERVO_MAX_DEG);
  steering.write(deg);
}

void setup() {
  Serial.begin(115200);

  pinMode(MOTOR_EN_PWM, OUTPUT);
  pinMode(MOTOR_IN1, OUTPUT);
  pinMode(MOTOR_IN2, OUTPUT);

  steering.attach(SERVO_PIN, SERVO_MIN_US, SERVO_MAX_US);
  setSteeringDeg(SERVO_CENTER_DEG);
  motorStop();

  lastCmdMs = millis();
}

void loop() {
  // Safety timeout
  if (millis() - lastCmdMs > CMD_TIMEOUT_MS) {
    motorStop();
    setSteeringDeg(SERVO_CENTER_DEG);
  }

  if (!Serial.available()) return;

  String line = Serial.readStringUntil('\n');
  line.trim();
  if (line.length() == 0) return;

  if (line.equals("STOP")) {
    motorStop();
    setSteeringDeg(SERVO_CENTER_DEG);
    lastCmdMs = millis();
    Serial.println("OK");
    return;
  }

  if (line.startsWith("CMD ")) {
    // Parse: CMD servo throttle
    // Example: CMD 78.5 150
    int firstSpace = line.indexOf(' ');
    int secondSpace = line.indexOf(' ', firstSpace + 1);
    if (secondSpace < 0) return;

    String sServo = line.substring(firstSpace + 1, secondSpace);
    String sThr = line.substring(secondSpace + 1);

    float servoDeg = sServo.toFloat();
    int throttle = sThr.toInt();

    setSteeringDeg(servoDeg);
    motorDrive(throttle);
    lastCmdMs = millis();
    Serial.println("OK");
    return;
  }
}

