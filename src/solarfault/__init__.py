"""Detecting and classifying faults in infrared images of solar modules."""

CLASSES = [
    "No-Anomaly",
    "Cell",
    "Cell-Multi",
    "Cracking",
    "Diode",
    "Diode-Multi",
    "Hot-Spot",
    "Hot-Spot-Multi",
    "Offline-Module",
    "Shadowing",
    "Soiling",
    "Vegetation",
]
NO_ANOMALY = 0
