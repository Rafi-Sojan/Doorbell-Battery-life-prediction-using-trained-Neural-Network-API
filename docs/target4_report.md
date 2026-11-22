# Target 4 Report: Doorbell Memory, Compute, and Deployment Decision

## 1. Executive summary

Target 4 asks whether the selected battery-life model can run inside an ESP32-S3-class doorbell. The answer is yes by a large margin for the selected `5 -> 32 -> 32 -> 1` model.

The resource analysis was implemented in `scripts/resources.py` and the API's `/model_info` response now also exposes the largest activation size. The analysis uses the actual shared `BatteryNet` architecture rather than a separately typed formula, so the measurements remain aligned with the model served by the API.

For one prediction, the selected model requires:

- 1,281 parameters;
- 5,124 bytes for float32 weights and biases;
- 1,281 bytes for an int8 weight estimate;
- 128 bytes for the largest float32 activation;
- 32 bytes for the largest int8 activation; and
- 1,216 multiply-accumulate operations.

Including one largest activation, the working-memory estimates are 5,252 bytes for float32 and 1,313 bytes for int8. Both are far below the assignment's approximately 300 KB SRAM budget and fit easily in 8 MB PSRAM.

## 2. Target 4 objective and evaluation loop

The target is to measure memory and compute for the deployed model, compare wider alternatives, locate the SRAM cutoffs, estimate on-device time, and make a reasoned deployment decision.

The mission loop was applied as follows:

**Prediction:** the current small network should fit comfortably on the doorbell, and its compute cost should be negligible compared with Wi-Fi communication.

**Measurement:** the model was inspected from `BatteryNet`, resource values were generated for widths 32, 128, 256, 512, and 2048, and the first non-fitting widths were searched exactly.

**Explanation:** the selected network is tiny relative to the memory budget, while wider models eventually fail SRAM or PSRAM limits. The edge-versus-server decision is therefore driven more by connectivity, privacy, power, and maintenance than by model capacity.

## 3. Role allocation

| Role | Work performed | Finding |
|---|---|---|
| Builder | Added `scripts/resources.py` and activation-size fields to `/model_info` without changing the existing comments. | The same `BatteryNet` implementation now drives the resource table and API metadata. |
| Measurer | Ran the resource script with the Python 3.11 environment and recorded width tables, memory fits, MACs, cutoffs, and the current-model time estimate. | The deployed model uses only 5,252 estimated float32 working-memory bytes and 1,216 MACs. |
| Skeptic | Examined the SRAM/PSRAM assumptions, quantisation estimate, and board-speed estimate. | The fit result is robust, but the time estimate is not a hardware benchmark and int8 deployment still needs quantisation validation. |
| Scribe | Recorded the prediction, measurements, assumptions, gaps, and deployment recommendation. | The evidence and limitations are documented in this report. |

## 4. Builder implementation

### 4.1 Resource calculation

`resource_summary(hidden)` constructs `BatteryNet(tuple(hidden))`, walks its `Linear` layers, and calculates each layer's parameters, float32 bytes, and MACs. ReLU layers are excluded because they have no trainable parameters and do not perform matrix multiplications.

For a `Linear(a, b)` layer:

- parameters = `a * b + b`;
- float32 bytes = parameters `* 4`;
- int8 bytes = parameters `* 1`; and
- MACs = `a * b`.

The script also includes the largest single activation in its working-memory fit check. This is a conservative one-inference estimate; it does not include the rest of the firmware, runtime buffers, or communication stacks.

### 4.2 API metadata

`/model_info` already computed the required total parameter, byte, and MAC fields. Target 4 now additionally exposes:

- `largest_activation_values`;
- `largest_activation_bytes_f32`; and
- `largest_activation_bytes_int8`.

For the deployed model, the API reports 32 largest activation values, 128 float32 bytes, and 32 int8 bytes.

### 4.3 Code blocks and explanations

#### Code block A: one shared resource calculation

```python
def resource_summary(hidden):
    # Build the same architecture used by the API rather than duplicating it.
    net = BatteryNet(tuple(hidden))
    total_params = 0
    total_macs = 0
    activation_widths = [len(FEATURES)]

    # ReLU has no parameters or MACs, so count Linear layers only.
    for module in net.net:
        if not isinstance(module, torch.nn.Linear):
            continue

        # A Linear(a, b) layer has a*b weights and b bias values.
        params = module.weight.numel() + module.bias.numel()
        macs = module.in_features * module.out_features
        total_params += params
        total_macs += macs
        activation_widths.append(module.out_features)

    # Four bytes represent one float32; one byte is the int8 estimate.
    weights_f32_bytes = total_params * 4
    weights_int8_bytes = total_params
    largest_activation_values = max(activation_widths)
    activation_f32_bytes = largest_activation_values * 4
    activation_int8_bytes = largest_activation_values

    # Include one largest activation in the model working-memory estimate.
    working_f32_bytes = weights_f32_bytes + activation_f32_bytes
    working_int8_bytes = weights_int8_bytes + activation_int8_bytes

    return {
        "total_params": total_params,
        "weights_f32_bytes": weights_f32_bytes,
        "weights_int8_bytes": weights_int8_bytes,
        "activation_f32_bytes": activation_f32_bytes,
        "activation_int8_bytes": activation_int8_bytes,
        "working_f32_bytes": working_f32_bytes,
        "working_int8_bytes": working_int8_bytes,
        "total_macs": total_macs,
    }
```

This function is the source of truth for the width table. Using `BatteryNet` means a future architecture change will be reflected in the resource calculation instead of leaving a hand-written formula stale.

#### Code block B: activation metadata in `/model_info`

```python
# Include the input width and every Linear output width.
activation_widths = [len(FEATURES)]
for module in net.net:
    if isinstance(module, torch.nn.Linear):
        activation_widths.append(module.out_features)

# Report the largest activation for one prediction.
largest_activation_values = max(activation_widths)
largest_activation_bytes_f32 = largest_activation_values * 4
largest_activation_bytes_int8 = largest_activation_values
```

These values are metadata for one inference. They do not claim that the complete ESP32-S3 firmware uses only this amount of SRAM.

#### Code block C: exact cutoff and board-time estimate

```python
# Search upward until the first width exceeds the chosen SRAM budget.
def first_width_that_does_not_fit(dtype):
    width = 1
    while True:
        result = resource_summary((width, width))
        if not result[f"fits_300kb_sram_{dtype}"]:
            return width
        width += 1


# MAC count divided by assumed MAC/s gives seconds; multiply by 1000 for ms.
estimated_time_ms = total_macs / ASSUMED_MACS_PER_SECOND * 1000
```

The search returned first non-fitting widths of 273 for float32 and 550 for int8. The deployed model's 1,216 MACs produced an estimate of 0.1216 ms at the stated 10 MMAC/s planning assumption.

## 5. Measurer results

The table below includes weights plus one largest activation for the SRAM and PSRAM fit decisions. The weight columns themselves remain separate so the memory contribution of activations is visible.

| Hidden width `w` | Parameters | Float32 weights | Int8 weights | Largest activation f32 | Largest activation int8 | MACs | SRAM f32 | SRAM int8 | PSRAM f32 | PSRAM int8 |
|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|:---:|
| 32 | 1,281 | 5,124 B | 1,281 B | 128 B | 32 B | 1,216 | Yes | Yes | Yes | Yes |
| 128 | 17,409 | 69,636 B | 17,409 B | 512 B | 128 B | 17,152 | Yes | Yes | Yes | Yes |
| 256 | 67,585 | 270,340 B | 67,585 B | 1,024 B | 256 B | 67,072 | Yes | Yes | Yes | Yes |
| 512 | 266,241 | 1,064,964 B | 266,241 B | 2,048 B | 512 B | 265,216 | No | Yes | Yes | Yes |
| 2048 | 4,210,689 | 16,842,756 B | 4,210,689 B | 8,192 B | 2,048 B | 4,206,592 | No | No | No | Yes |

### 5.1 Exact SRAM cutoffs

The first equal hidden width that does not fit when the largest activation is included is:

- float32: width **273** is the first non-fitting width; width 272 is the largest fitting width;
- int8: width **550** is the first non-fitting width; width 549 is the largest fitting width.

These cutoffs are for the model weights plus the largest single activation only. They are not a claim that the entire firmware can use all 300 KB for the model.

### 5.2 Current model board-time estimate

The current model has 1,216 MACs. The script uses a conservative engineering assumption of 10,000,000 MACs per second:

`time = 1,216 / 10,000,000 * 1,000 = 0.1216 ms`

The official Espressif ESP32-S3 chip comparison documents a typical CPU frequency of 240 MHz and 512 KB SRAM. That clock frequency is hardware context, not a direct measured MAC/s result. The 10 MMAC/s value is therefore explicitly an estimate used for planning, not a board benchmark. A real ESP32-S3 implementation should measure the compiled inference path with the selected numeric format.

For sensitivity, the same 1,216-MAC model would estimate approximately:

| Assumed throughput | Estimated inference time |
|---:|---:|
| 1 MMAC/s | 1.216 ms |
| 10 MMAC/s | 0.1216 ms |
| 100 MMAC/s | 0.01216 ms |

## 6. Skeptic findings

### 6.1 The memory result is strong, but it is not the complete firmware budget

The model occupies only a few kilobytes, so the conclusion is not sensitive to small accounting differences. However, the fit calculation includes weights and one largest activation, not the complete application, tensor arena, stack, allocator overhead, Wi-Fi stack, logging, or other firmware data. The model fits; the final firmware still needs a real memory report.

### 6.2 Int8 is an estimate until quantisation is validated

The int8 values assume one byte per parameter and activation. They do not prove that a quantised model has the same accuracy as the float32 model. A deployment should use calibrated or representative-data quantisation and compare MAE before replacing the float32 artifact.

### 6.3 MAC/s is not the same as clock frequency

The ESP32-S3's documented 240 MHz clock does not mean 240 million useful MACs per second. Memory access, instruction scheduling, tensor format, SIMD use, compiler output, and quantisation all affect throughput. The 0.1216 ms estimate must therefore be presented as an assumption-based estimate, not as measured hardware latency.

### 6.4 PSRAM fit does not automatically mean fast execution

The width-2048 int8 model fits the 8 MB PSRAM estimate but does not fit the 300 KB SRAM estimate. Moving active data through external PSRAM may introduce latency and power costs, so a PSRAM fit is a capacity result, not a performance guarantee.

### 6.5 The width sweep is a capacity comparison, not a recommendation to enlarge the model

The selected width 32 model already has validation MAE 5.87 from Target 2. A wider model would increase memory and compute, but capacity alone is not evidence of better generalisation. Any replacement would need the same validation, seed, and hidden-test checks.

## 7. Deployment recommendation

The doorbell should run this prediction locally as the primary path:

- **Latency:** 1,216 MACs and approximately 5 KB of float32 weights are small enough for a fast local calculation; there is no public-network round trip.
- **Privacy:** raw usage, Wi-Fi, temperature, and device-age features do not need to leave the device for a prediction.
- **Cost:** local inference avoids a server request for every battery estimate.
- **Power:** the computation is small, and avoiding a Wi-Fi transaction can be beneficial when the device would otherwise wake the radio solely for inference. The exact power advantage requires a hardware measurement.
- **Wi-Fi outage:** local inference continues to work when connectivity is unavailable.

The server should remain useful for model training, monitoring, model distribution, diagnostics, and optional telemetry. A practical architecture is local inference with occasional server communication for updates and health data rather than server-only prediction.

## 8. Conclusion

Target 4 is complete for the assignment's analysis and code requirements. The selected `5 -> 32 -> 32 -> 1` model fits comfortably in the ESP32-S3 resource envelope, and the resource table, exact SRAM cutoffs, API metadata, board-time estimate, skeptic review, and deployment recommendation are documented.

The remaining engineering step outside this assignment analysis is a real ESP32-S3 benchmark and, if int8 deployment is chosen, a quantisation-accuracy comparison. Those measurements would refine the estimate but are unlikely to change the recommendation for this very small model.

## 9. Source

- Espressif, [ESP32-S3 chip series comparison](https://docs.espressif.com/projects/esp-idf/en/v5.0/esp32s3/hw-reference/chip-series-comparison.html), for the documented 240 MHz typical frequency and 512 KB SRAM context.
