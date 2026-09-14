import csv
from pathlib import Path

from patient_generator import generate_patient


NUM_PATIENTS = 10_000

OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = OUTPUT_DIR / "cardia_synthetic_dataset.csv"


FEATURES = [
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
    "map",
    "cardiac_output",
    "stroke_volume",
    "edv",
    "esv",
    "lv_pressure",
    "aortic_pressure",
]

TARGETS = [
    "blood_volume",
    "contractility",
    "svr",
]


def generate_dataset():
    fieldnames = FEATURES + TARGETS

    with open(OUTPUT_FILE, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for _ in range(NUM_PATIENTS):
            patient = generate_patient()

            row = {
                feature: patient[feature]
                for feature in fieldnames
            }

            writer.writerow(row)

    print(f"Generated {NUM_PATIENTS:,} synthetic patients.")
    print(f"Dataset saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    generate_dataset()