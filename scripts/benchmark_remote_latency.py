"""
Target 3 remote latency benchmark.

This script sends 200 valid requests to the public Codespace API twice:

1. With HTTP keep-alive enabled.
2. With HTTP keep-alive disabled.

For both cases it reports P50, P95, and P99 latency for the full client
round trip and for the timing stages returned by the API.

Run from the repository root with:

    python scripts/benchmark_remote_latency.py

The API URL can be replaced without editing this file by setting the
DOORBELL_API_URL environment variable.
"""

import os
import time

import requests


# The public Codespace URL used for the current Target 3 verification.
# Set DOORBELL_API_URL if the Codespace URL changes.
BASE = os.getenv(
    "DOORBELL_API_URL",
    "https://orange-fishstick-69rrj64jppv4hr4vw-8000.app.github.dev",
).rstrip("/")
URL = f"{BASE}/predict"

# Number of measured requests in each benchmark.
N = 200

# A valid reading is used for every request so that the benchmark measures
# request latency rather than validation failures.
ROW = {
    "battery_v": 3.8,
    "temp_c": 25,
    "rings_per_day": 8,
    "wifi_rssi_dbm": -60,
    "age_months": 12,
}

# These names must match the timing fields returned by app/main.py.
STAGES = ["parse", "normalise", "inference", "build", "server_total"]


def percentile(values, percentage):
    """Return a linearly interpolated percentile from a list of timings."""

    # Sorting makes the position calculation deterministic.
    values = sorted(values)

    # Map the requested percentage onto the zero-based sorted list.
    position = (len(values) - 1) * percentage / 100
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower

    # Interpolate when the percentile falls between two measured values.
    return values[lower] + (values[upper] - values[lower]) * fraction


def run_test(name, keep_alive):
    """Run one 200-request benchmark and print its latency distribution."""

    # Store full client round-trip measurements separately from server-stage
    # measurements returned inside the JSON response.
    roundtrips = []
    timings = {stage: [] for stage in STAGES}

    # A Session reuses the TCP connection for keep-alive measurements.
    # Without a Session, requests creates a new connection for each request.
    session = requests.Session() if keep_alive else None
    client = session if session else requests

    # Warm up the API so model loading and the first request do not distort
    # the 200-request distribution.
    warmup = client.post(
        URL,
        json=ROW,
        headers={} if keep_alive else {"Connection": "close"},
        timeout=30,
    )
    warmup.raise_for_status()

    # Measure the requested number of valid API calls.
    for _ in range(N):
        start = time.perf_counter()

        response = client.post(
            URL,
            json=ROW,
            headers={} if keep_alive else {"Connection": "close"},
            timeout=30,
        )

        # This includes the public tunnel, network, HTTP framework, and API.
        elapsed = (time.perf_counter() - start) * 1000
        response.raise_for_status()

        body = response.json()
        roundtrips.append(elapsed)

        # These values measure only the server-side stages reported by the API.
        for stage in STAGES:
            timings[stage].append(float(body["timings_ms"][stage]))

    # Release the persistent connection after the test is complete.
    if session:
        session.close()

    print(f"\n{name}")
    print("-" * len(name))

    # Print the requested P50, P95, and P99 values for each measurement type.
    for label, values in [("roundtrip", roundtrips), *timings.items()]:
        print(
            f"{label:12}"
            f"P50={percentile(values, 50):8.3f} ms  "
            f"P95={percentile(values, 95):8.3f} ms  "
            f"P99={percentile(values, 99):8.3f} ms"
        )


if __name__ == "__main__":
    # Run both connection modes so network setup overhead can be compared.
    run_test("Keep-alive", True)
    run_test("No keep-alive", False)
