import numpy as np
import tensorflow as tf

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="C:/Users/sheet/Desktop/kneeRevive/KneeRevive/model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Print expected input shape
expected_shape = input_details[0]['shape']
print("Expected input shape:", expected_shape)

# Create valid input data (shape must match [1, 7])
# Replace these dummy values with actual data if needed
input_data = np.array([[2.29, -3.2, 0.91, 0.4, -2.5, 0.6, 0.7]], dtype=np.float32)

# (Optional) Reshape if needed
input_data = np.reshape(input_data, expected_shape)

# Set the tensor and run inference
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# Get the output
output_data = interpreter.get_tensor(output_details[0]['index'])
print("Model output:", output_data)

# Assuming these are your class labels in the same order used during training
class_labels = ['Walk (Safe)', 'Unsafe Bend', 'Stand', 'Jerky Movement']  # <-- modify if needed

# Get the index of the class with highest probability
predicted_index = np.argmax(output_data)
predicted_label = class_labels[predicted_index]

print("Predicted Label:", predicted_label)
