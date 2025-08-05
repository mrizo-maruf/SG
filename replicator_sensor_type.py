# replicator_sensor_type.py
import omni.syntheticdata._syntheticdata as sd

for sensor_type in sd.SensorType:
    print(f"Available sensor type: {sensor_type.name}")