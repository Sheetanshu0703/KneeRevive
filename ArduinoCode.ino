#include <Wire.h>
#include <MPU6050.h>
#include "TensorFlowLite.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

MPU6050 mpu;

// Flex sensor pin
const int flexPin = 34;
const int minFlex = 1500;
const int maxFlex = 3000;

const unsigned long interval = 500;
unsigned long previousMillis = 0;

// --- TensorFlow Lite Setup ---
extern const unsigned char model_tflite[];
extern const unsigned int model_tflite_len;

constexpr int kTensorArenaSize = 10 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

tflite::MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;

void setup() {
  Serial.begin(115200);
  Wire.begin();

  // Initialize MPU6050
  Serial.println("Initializing MPU6050...");
  mpu.initialize();
  if (!mpu.testConnection()) {
    Serial.println("MPU6050 connection failed");
    while (1);
  }
  Serial.println("MPU6050 connected!");

  // Load model
  const tflite::Model* model = tflite::GetModel(model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version mismatch!");
    while (1);
  }

  // Interpreter and tensor allocation
  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("Tensor allocation failed");
    while (1);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("Model ready for inference!");
}

void loop() {
  unsigned long currentMillis = millis();
  if (currentMillis - previousMillis >= interval) {
    previousMillis = currentMillis;

    // --- Read MPU6050 data ---
    int16_t ax, ay, az, gx, gy, gz;
    mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

    // Normalize accelerometer values (convert to 'g')
    float acc_x = ax / 16384.0;
    float acc_y = ay / 16384.0;
    float acc_z = az / 16384.0;

    // Normalize gyroscope values (convert to °/s)
    float gyro_x = gx / 131.0;
    float gyro_y = gy / 131.0;
    float gyro_z = gz / 131.0;

    // Read flex sensor and compute knee angle
    int flexRaw = analogRead(flexPin);
    float kneeAngle = map(flexRaw, minFlex, maxFlex, 90, 180);
    kneeAngle = constrain(kneeAngle, 90, 180);

    // Input data for the model (already normalized)
    float inputData[7] = {
      acc_x, acc_y, acc_z,
      gyro_x, gyro_y, gyro_z,
      kneeAngle
    };

    // Load data into model input
    if (input->type != kTfLiteFloat32) {
      Serial.println("Model input type mismatch!");
      return;
    }

    for (int i = 0; i < 7; i++) {
      input->data.f[i] = inputData[i];
    }

    // Run inference
    if (interpreter->Invoke() != kTfLiteOk) {
      Serial.println("Invoke failed!");
      return;
    }

    // Output classification results
    Serial.print("Model output: ");
    int output_size = output->dims->data[output->dims->size - 1];
    for (int i = 0; i < output_size; i++) {
      Serial.print(output->data.f[i], 4);
      Serial.print(" ");
    }
    Serial.println();
    Serial.println("-----------------------------");
  }
}
