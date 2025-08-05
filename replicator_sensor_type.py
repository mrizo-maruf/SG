# replicator_sensor_type.py
from omni.syntheticdata._syntheticdata import SensorType

if __name__ == "__main__":
    names = [m.name for m in SensorType]
    print("Available SensorType entries:\n", "\n".join(names))
