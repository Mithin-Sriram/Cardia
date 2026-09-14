import random


PARAMETER_MIN = 70.0
PARAMETER_MAX = 130.0
PARAMETER_BASELINE = 100.0


def generate_patient():
    """
    Generate one synthetic CARDIA patient.

    The three hidden physiological parameters are sampled first.
    Temporary physiological observations are then generated from them.

    This is a temporary replacement for the CARDIA simulator.
    It will later be replaced by the real simulator.
    """

    # Hidden parameters / ML targets
    blood_volume = random.uniform(PARAMETER_MIN, PARAMETER_MAX)
    contractility = random.uniform(PARAMETER_MIN, PARAMETER_MAX)
    svr = random.uniform(PARAMETER_MIN, PARAMETER_MAX)

    # Temporary physiological observations
    # These relationships are ONLY for testing the ML pipeline.
    blood_volume_effect = blood_volume - PARAMETER_BASELINE
    contractility_effect = contractility - PARAMETER_BASELINE
    svr_effect = svr - PARAMETER_BASELINE

    heart_rate = (
        82
        - 0.10 * blood_volume_effect
        - 0.12 * contractility_effect
        + 0.15 * svr_effect
        + random.gauss(0, 2)
    )

    stroke_volume = (
        63
        + 0.30 * blood_volume_effect
        + 0.35 * contractility_effect
        - 0.10 * svr_effect
        + random.gauss(0, 1.5)
    )

    cardiac_output = (
        heart_rate * stroke_volume / 1000
    )

    mean_arterial_pressure = (
        90
        + 0.08 * blood_volume_effect
        + 0.05 * contractility_effect
        + 0.30 * svr_effect
        + random.gauss(0, 1.5)
    )

    systolic_bp = (
        mean_arterial_pressure
        + 28
        + 0.15 * contractility_effect
        + random.gauss(0, 2)
    )

    diastolic_bp = (
        mean_arterial_pressure
        - 14
        + 0.05 * svr_effect
        + random.gauss(0, 1.5)
    )

    edv = (
        128
        + 0.55 * blood_volume_effect
        + random.gauss(0, 2)
    )

    ejection_fraction = max(
        0.30,
        min(
            0.80,
            0.50 + 0.0025 * contractility_effect
        )
    )

    esv = edv * (1 - ejection_fraction)

    lv_pressure = (
        systolic_bp
        - 3
        + 0.08 * contractility_effect
        + random.gauss(0, 1)
    )

    aortic_pressure = (
        mean_arterial_pressure
        + random.gauss(0, 1)
    )

    return {
        "heart_rate": heart_rate,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "map": mean_arterial_pressure,
        "cardiac_output": cardiac_output,
        "stroke_volume": stroke_volume,
        "edv": edv,
        "esv": esv,
        "lv_pressure": lv_pressure,
        "aortic_pressure": aortic_pressure,

        # ML targets
        "blood_volume": blood_volume,
        "contractility": contractility,
        "svr": svr,
    }


if __name__ == "__main__":
    patient = generate_patient()

    for key, value in patient.items():
        print(f"{key}: {value:.2f}")