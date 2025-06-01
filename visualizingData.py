import random
import csv

# Constants
sampling_rate = 50  # Hz
duration_sec = 60
total_samples = sampling_rate * duration_sec

# Function to generate smooth sensor data with occasional jerks
def generate_sensor_data():
    data = []
    for i in range(total_samples):
        # Time in seconds
        timestamp = i / sampling_rate

        # Generate baseline accelerometer and gyroscope values
        ax = round(random.uniform(-0.2, 0.2), 3)
        ay = round(random.uniform(-0.2, 0.2), 3)
        az = round(random.uniform(0.8, 1.2), 3)  # gravity vector

        gx = round(random.uniform(-10, 10), 3)
        gy = round(random.uniform(-10, 10), 3)
        gz = round(random.uniform(-10, 10), 3)

        # Introduce a jerk every 2-3 seconds
        if i % random.randint(90, 160) == 0:
            ax += random.uniform(1.0, 2.0) * random.choice([-1, 1])
            ay += random.uniform(1.0, 2.0) * random.choice([-1, 1])
            az += random.uniform(0.5, 1.0) * random.choice([-1, 1])

            gx += random.uniform(100, 200) * random.choice([-1, 1])
            gy += random.uniform(100, 200) * random.choice([-1, 1])
            gz += random.uniform(100, 200) * random.choice([-1, 1])

        data.append({
            "timestamp": round(timestamp, 2),
            "ax": round(ax, 3),
            "ay": round(ay, 3),
            "az": round(az, 3),
            "gx": round(gx, 3),
            "gy": round(gy, 3),
            "gz": round(gz, 3)
        })

    return data

# Write to CSV
with open("simulated_knee_data.csv", "w", newline="") as csvfile:
    fieldnames = ["timestamp", "ax", "ay", "az", "gx", "gy", "gz"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in generate_sensor_data():
        writer.writerow(row)

print("✅ Simulated sensor data saved to 'simulated_knee_data.csv'")
