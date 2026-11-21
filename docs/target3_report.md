# Target 3 Report: API Validation and Request Timing

## 1. Executive summary

Target 3 converts the trained battery-life model into a measurable API contract. The implementation was tested locally through the required endpoints and through valid, invalid, single-reading, and batch-reading requests.

The API passed the following checks:

- `/health` returned HTTP 200 and reported team 24, 1,281 parameters, and validation MAE 5.87.
- `/model_info` reported the deployed network structure and resource estimates consistently with the saved model.
- Valid `/predict` requests returned `days_left` together with `parse`, `normalise`, `inference`, `build`, and `server_total` timing fields.
- Five invalid readings were rejected with HTTP 422.
- A batch of 400 readings produced exactly the same predictions as 400 individual requests after stable output rounding was added.
- The 400-reading batch completed in 4.035 ms of server time, compared with 619.692 ms for individual requests, giving a measured 153.59x throughput improvement in the local test.

These measurements satisfy the local Target 3 checks. A Codespace/public-port measurement is still required before claiming the final remote latency result, because network distance and deployment infrastructure are not represented by the local benchmark.

## 2. Target 3 objective

The target is to expose the selected model through a reliable API that:

1. accepts valid single and batch readings;
2. rejects physically impossible or glitch-like readings at the request boundary;
3. reports the required timing stages;
4. exposes model information needed for an edge-resource discussion; and
5. preserves prediction equality between single-request and batch-request execution.

The evaluation loop used was:

**Prediction:** the API should return a valid battery-life prediction and the required timing fields.

**Measurement:** endpoint status codes, response fields, timing distributions, model metadata, and single-versus-batch equality were measured.

**Explanation:** validation rules prevent malformed inputs from reaching the model, stage timings identify where request time is spent, and batch inference reduces repeated request overhead.

## 3. Role allocation

| Role | Work performed | Finding |
|---|---|---|
| Builder | Implemented bounded request validation, stage timing, model metadata, and stable prediction rounding. | The API now exposes the required contract for valid, invalid, and batch inputs. |
| Measurer | Exercised `/health`, `/model_info`, `/predict`, invalid readings, 200-request latency tests, and 400-reading equality tests. | All local functional checks passed; batch processing was substantially faster. |
| Skeptic | Checked whether local measurements could be treated as deployment measurements and investigated single/batch numeric differences. | Local latency is not a substitute for Codespace latency; small floating-point differences required stable rounding. |
| Scribe | Recorded the prediction, measurements, gap, assumptions, and remaining deployment work. | The evidence and limitations are documented in this report. |

## 4. Builder findings

### 4.1 Input validation

The request model applies explicit physical bounds before inference:

| Field | Accepted range | Reason |
|---|---:|---|
| `battery_v` | 3.0 to 4.3 V | Rejects zero and non-physical battery voltages. |
| `temp_c` | -40 to 85 °C | Rejects the dataset glitch value `-127` and extreme unsupported temperatures. |
| `rings_per_day` | 0 to 1,000 | Rejects negative activity and implausibly high activity. |
| `wifi_rssi_dbm` | -100 to -1 dBm | Rejects the dataset glitch value `0` and impossible positive RSSI. |
| `age_months` | 0 to 240 months | Rejects negative device age and limits the supported lifetime domain. |

The bounds are API-domain assumptions. They should be revisited if the intended hardware specification supports a wider range.

### 4.2 Timing instrumentation

The prediction path records five stages:

- `parse`: request decoding and validation;
- `normalise`: feature-array construction and standardisation using the saved training mean and standard deviation;
- `inference`: neural-network execution;
- `build`: output conversion, clipping, rounding, and response construction;
- `server_total`: total measured server-side processing time.

The output is rounded consistently before it is returned. This makes the public numerical contract stable when a value is computed once as a single request and again inside a batch tensor.

### 4.3 Model information

The selected model is a 5-input, two-hidden-layer neural network with hidden widths 32 and 32 and one output. The reported resource summary is:

| Quantity | Value |
|---|---:|
| Total parameters | 1,281 |
| Float32 model bytes | 5,124 |
| Int8 parameter estimate | 1,281 |
| Multiply-accumulate estimate | 1,216 |
| Float32 memory limit check | Pass |
| Int8 memory limit check | Pass |

## 5. Measurer findings

### 5.1 Endpoint checks

`/health` returned HTTP 200 with `status: ok`, team 24, 1,281 parameters, and validation MAE 5.87.

`/model_info` returned the three deployed linear layers:

| Layer | Shape | Parameters | Float32 bytes | MACs |
|---|---|---:|---:|---:|
| `linear1` | 5 to 32 | 192 | 768 | 160 |
| `linear2` | 32 to 32 | 1,056 | 4,224 | 1,024 |
| `linear3` | 32 to 1 | 33 | 132 | 32 |
| Total | — | 1,281 | 5,124 | 1,216 |

A valid `/predict` request returned HTTP 200 and included `days_left` plus all five required timing keys.

### 5.2 Invalid-input checks

The following invalid cases all returned HTTP 422 as expected:

| Test case | Invalid value or condition | Result |
|---|---|---|
| `battery_zero` | `battery_v = 0` | HTTP 422 |
| `temp_glitch` | `temp_c = -127` | HTTP 422 |
| `wifi_glitch` | `wifi_rssi_dbm = 0` | HTTP 422 |
| `temp_below_range` | `temp_c = -41` | HTTP 422 |
| `age_negative` | `age_months = -1` | HTTP 422 |

This confirms that the request boundary blocks known glitch values and out-of-domain values before model inference.

### 5.3 Persistent-connection latency

The first request took 4.2 ms. The following 200-request local benchmark used a persistent HTTP connection:

| Measurement | P50 ms | P95 ms | P99 ms |
|---|---:|---:|---:|
| Full round trip | 1.927 | 2.797 | 2.986 |
| `parse` | 0.027 | 0.035 | 0.043 |
| `normalise` | 0.024 | 0.031 | 0.037 |
| `inference` | 0.155 | 0.231 | 0.329 |
| `build` | 0.014 | 0.023 | 0.025 |
| `server_total` | 0.233 | 0.328 | 0.453 |
| Network/framework remainder | 1.701 | 2.485 | 2.662 |

### 5.4 No-keep-alive latency

The first no-keep-alive request took 11.5 ms. The following 200-request benchmark was:

| Measurement | P50 ms | P95 ms | P99 ms |
|---|---:|---:|---:|
| Full round trip | 12.101 | 14.843 | 15.486 |
| `parse` | 0.036 | 0.070 | 0.148 |
| `normalise` | 0.031 | 0.051 | 0.062 |
| `inference` | 0.206 | 0.423 | 0.658 |
| `build` | 0.016 | 0.027 | 0.033 |
| `server_total` | 0.303 | 0.605 | 0.814 |
| Network/framework remainder | 11.806 | 14.271 | 14.810 |

The comparison shows that connection setup dominates the no-keep-alive round trip. The model stages are a small portion of the total local request time.

### 5.5 Single-versus-batch equality and speed

The same 400 valid readings were sent once as 400 individual requests and once as one batch request:

| Test | Total time |
|---|---:|
| 400 individual requests | 619.692 ms |
| One batch request | 4.035 ms |
| Batch speedup | 153.59x |
| Predictions exactly equal | Yes |

The batch request also returned stage timings of approximately 0.557 ms for parsing, 0.278 ms for normalisation, 0.248 ms for inference, 0.496 ms for response building, and 1.590 ms for total server time.

## 6. Skeptic findings and explanation of the gap

### 6.1 Local latency is not remote latency

The benchmark was run against a local server. It demonstrates API correctness and local processing cost, but it does not establish the final laptop-to-Codespace latency. The remote result may be higher because of internet routing, TLS or proxy layers, public-port forwarding, CPU allocation, and server load.

Therefore, the local P50, P95, and P99 values must be labelled local measurements rather than deployment guarantees.

### 6.2 The original batch mismatch was numerical, not a model disagreement

Before stable output rounding, a single request produced a value such as 81.00 while the corresponding batch output was 80.99. Both paths used the same model and features, but floating-point execution order produced a small difference near a rounding boundary.

The fix was to round the model output to three decimal places before converting the public `days_left` result to two decimal places. After this change, all 400 single predictions matched the corresponding batch predictions exactly.

This improves API determinism, but it does not make the underlying floating-point computation mathematically identical in every possible runtime or hardware environment. The equality test should remain part of regression testing.

### 6.3 Validation ranges are policy, not proof of sensor truth

Range validation catches obvious glitches and unsupported values. It cannot prove that a value is physically correct inside the accepted range. For example, a sensor may report a stable but incorrect voltage or temperature. Future work could add temporal checks, missing-value checks, device-level calibration, or a warning channel for suspicious-but-accepted readings.

### 6.4 Timing values depend on the measurement environment

The reported timings depend on the local operating system, Python runtime, PyTorch configuration, server process state, request client, and warm-up state. A fair comparison should repeat the same test procedure in the Codespace and from the laptop using the public port.

## 7. Scribe record: prediction, measurement, explanation

| Loop stage | Target 3 record |
|---|---|
| Prediction | The API should reject malformed readings, expose timing breakdowns, and produce identical single and batch predictions. |
| Measurement | Endpoint checks passed; five invalid cases returned 422; required timings were present; 400 batch predictions matched 400 single predictions; batch speedup was 153.59x locally. |
| Explanation | Validation protects the model boundary, stage timings separate model work from request overhead, and stable rounding makes the public result deterministic. The remaining gap is the unmeasured laptop-to-Codespace network path. |

## 8. Required remote follow-up

Before final submission, run the same checks with the API exposed on Codespace port 8000 and accessed from the laptop:

1. verify `/health` through the public URL;
2. verify `/model_info` through the public URL;
3. run 200 requests with keep-alive and record P50, P95, and P99;
4. run 200 requests with keep-alive disabled and record P50, P95, and P99;
5. run the 400 single-versus-batch equality test through the public URL;
6. compare remote round-trip latency with the local tables in this report; and
7. confirm that the interactive `/docs` page is reachable on the public port.

## 9. Conclusion

Target 3 is functionally complete for the local environment. The API validates the supported input domain, exposes the required timing breakdown, reports model resource information, rejects known glitch inputs, and preserves exact batch/single prediction equality. The most important measured performance result is the 153.59x local speedup from batching 400 readings.

The only remaining evidence needed for a final deployment claim is the equivalent laptop-to-Codespace benchmark. That measurement should be appended to this report after the public endpoint is running.
