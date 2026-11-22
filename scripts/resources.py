"""
Target 4: ESP32-S3 memory and speed analysis.

This script calculates model parameters, memory, MACs, SRAM/PSRAM fit,
activation memory, exact SRAM cutoffs, and a board-time estimate for the
required widths 32, 128, 256, 512, and 2048.

Run from the repository root with:

    python scripts/resources.py
"""

import torch

from app.model import FEATURES, BatteryNet


# The assignment gives approximately 300 KB of SRAM available to the program
# and 8 MB of PSRAM on the ESP32-S3.
SRAM_LIMIT_BYTES = 300 * 1024
PSRAM_LIMIT_BYTES = 8 * 1024 * 1024

# This is a conservative engineering assumption for the estimate, not a
# measured benchmark of this project on an ESP32-S3. The official Espressif
# hardware reference lists the ESP32-S3 typical CPU frequency as 240 MHz;
# the report must distinguish that clock fact from this MAC/s assumption.
ASSUMED_MACS_PER_SECOND = 10_000_000


def resource_summary(hidden):
    """Calculate resources from the same BatteryNet used by the API."""

    # Building BatteryNet here prevents the resource table from drifting away
    # from the architecture used by train.py and app/main.py.
    net = BatteryNet(tuple(hidden))
    total_params = 0
    total_macs = 0
    layers = []
    activation_widths = [len(FEATURES)]

    # Count Linear layers only: ReLU has no parameters or MACs.
    for module in net.net:
        if not isinstance(module, torch.nn.Linear):
            continue

        # A Linear(a, b) layer has a*b weights and b bias values.
        params = module.weight.numel() + module.bias.numel()

        # One input value is multiplied by each weight for every output.
        macs = module.in_features * module.out_features

        total_params += params
        total_macs += macs
        activation_widths.append(module.out_features)

        layers.append({
            "name": f"linear{len(layers) + 1}",
            "in": module.in_features,
            "out": module.out_features,
            "params": params,
            "bytes_f32": params * 4,
            "macs": macs,
        })

    # Float32 uses four bytes per parameter; the int8 result is a one-byte
    # estimate and assumes quantisation has already been validated for accuracy.
    weights_f32_bytes = total_params * 4
    weights_int8_bytes = total_params

    # Include the input and output widths when finding the largest activation.
    largest_activation_values = max(activation_widths)
    activation_f32_bytes = largest_activation_values * 4
    activation_int8_bytes = largest_activation_values

    # Include one largest activation in the working-memory fit check. This is
    # the model working set only, not the complete firmware memory budget.
    working_f32_bytes = weights_f32_bytes + activation_f32_bytes
    working_int8_bytes = weights_int8_bytes + activation_int8_bytes

    return {
        "hidden": tuple(hidden),
        "layers": layers,
        "total_params": total_params,
        "weights_f32_bytes": weights_f32_bytes,
        "weights_int8_bytes": weights_int8_bytes,
        "largest_activation_values": largest_activation_values,
        "activation_f32_bytes": activation_f32_bytes,
        "activation_int8_bytes": activation_int8_bytes,
        "working_f32_bytes": working_f32_bytes,
        "working_int8_bytes": working_int8_bytes,
        "total_macs": total_macs,
        "fits_300kb_sram_f32": working_f32_bytes <= SRAM_LIMIT_BYTES,
        "fits_300kb_sram_int8": working_int8_bytes <= SRAM_LIMIT_BYTES,
        "fits_8mb_psram_f32": working_f32_bytes <= PSRAM_LIMIT_BYTES,
        "fits_8mb_psram_int8": working_int8_bytes <= PSRAM_LIMIT_BYTES,
        "estimated_time_ms": total_macs / ASSUMED_MACS_PER_SECOND * 1000,
    }


def first_width_that_does_not_fit(dtype):
    """Return the first equal hidden width that exceeds 300 KB SRAM."""

    width = 1
    while True:
        result = resource_summary((width, width))
        key = f"fits_300kb_sram_{dtype}"

        # The first false result is the exact first width outside the budget.
        if not result[key]:
            return width

        width += 1


def print_table():
    """Print the required width comparison table."""

    widths = [32, 128, 256, 512, 2048]
    print("width | params | f32 bytes | int8 bytes | activation f32 | activation int8 | MACs | SRAM f32 | SRAM int8 | PSRAM f32 | PSRAM int8")
    print("-" * 150)

    for width in widths:
        result = resource_summary((width, width))
        print(
            f"{width:5d} | "
            f"{result['total_params']:6,d} | "
            f"{result['weights_f32_bytes']:9,d} | "
            f"{result['weights_int8_bytes']:10,d} | "
            f"{result['activation_f32_bytes']:14,d} | "
            f"{result['activation_int8_bytes']:15,d} | "
            f"{result['total_macs']:6,d} | "
            f"{str(result['fits_300kb_sram_f32']):8s} | "
            f"{str(result['fits_300kb_sram_int8']):9s} | "
            f"{str(result['fits_8mb_psram_f32']):9s} | "
            f"{str(result['fits_8mb_psram_int8']):10s}"
        )


def main():
    # Print the assignment's required comparison widths first.
    print("Target 4 resource table")
    print_table()

    # Print the deployed 5 -> 32 -> 32 -> 1 model separately.
    current = resource_summary((32, 32))
    print("\nCurrent deployed model: 5 -> 32 -> 32 -> 1")
    for key, value in current.items():
        if key not in {"hidden", "layers"}:
            print(f"{key}: {value}")

    # Find the first width that exceeds the SRAM budget for each format.
    print("\nExact SRAM cutoffs")
    print("First non-fitting float32 width:", first_width_that_does_not_fit("f32"))
    print("First non-fitting int8 width:", first_width_that_does_not_fit("int8"))

    # Report the assumption used for the board-time estimate separately from
    # the official ESP32-S3 clock specification.
    print("\nBoard-time assumption")
    print("Assumed MAC/s:", ASSUMED_MACS_PER_SECOND)
    print("Current model estimate (ms):", current["estimated_time_ms"])


if __name__ == "__main__":
    main()
