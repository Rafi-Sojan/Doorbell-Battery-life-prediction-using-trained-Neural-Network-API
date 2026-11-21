# Target 1 Report: Data Quality and Baseline Performance

## Executive summary

This investigation establishes the first performance benchmark for the doorbell battery-life prediction problem. The analysis used 2,000 records from `data/team_24_train.csv` and followed the required cycle of prediction, measurement, and explanation. The data contained 24 sensor-failure records, which were removed using three physically justified rules. After cleaning, a deterministic 80/20 split produced 1,581 training records and 395 validation records.

Three predictors were compared. Predicting the training-set mean produced a validation mean absolute error (MAE) of 68.10 days. A least-squares linear regression reduced the error to 28.67 days. The existing `5 → 32 → 32 → 1` neural network achieved 19.44 days without input normalization. The results show that the sensor readings contain substantial predictive information and that nonlinear relationships or feature interactions are important.

## 1. Objective and evaluation principle

The objective of Target 1 is to establish a trustworthy data set and a set of numerical reference points before expanding the neural-network design. Every conclusion follows the same loop:

1. **Prediction:** state the expected outcome before measuring it.
2. **Measurement:** calculate the outcome on held-out validation data.
3. **Explanation:** identify the data pattern or modeling limitation that explains the difference.

The evaluation metric is mean absolute error:

```text
MAE = mean(abs(predicted_days_left - true_days_left))
```

The validation set was not used to fit the baselines. A fixed random seed of `0` was used so that the results are reproducible.

## 2. Role allocation

The work was organized through four rotating roles.

| Role | Responsibility in Target 1 |
|---|---|
| Builder | Implement the diagnostic plots, sensor-glitch rules, baselines, and neural-network run in `train.py`. |
| Measurer | Record row counts, validation MAE values, coefficient values, and prediction ranges. |
| Skeptic | Test whether each cleaning rule is physically justified and whether the plots support the conclusions. |
| Scribe | Document the method, evidence, comparison, limitations, and final conclusion. |

## 3. Dataset and preparation

The data contains five input features and one continuous target:

- `battery_v`: battery voltage in volts
- `temp_c`: measured temperature in degrees Celsius
- `rings_per_day`: daily doorbell-ring count
- `wifi_rssi_dbm`: Wi-Fi signal strength in dBm
- `age_months`: device age in months
- `days_left`: target battery life in days

The original data contains no missing values and no duplicate rows. The following cleaning rules were applied before the train/validation split:

```python
bad = (
    (df["battery_v"] <= 0)
    | (df["temp_c"] == -127)
    | (df["wifi_rssi_dbm"] == 0)
)
```

### 3.1 Sensor-glitch evidence

| Rule | Rows removed | Interpretation |
|---|---:|---|
| `battery_v <= 0` | 8 | Zero voltage is not a valid operating battery reading. |
| `temp_c == -127` | 7 | `-127` is a sensor error or sentinel value, not a physical temperature. |
| `wifi_rssi_dbm == 0` | 9 | Wi-Fi RSSI values are negative; zero represents an invalid or missing measurement in this data. |
| **Total** | **24** | The three rule sets have no overlapping rows. |

The cleaned data therefore contains 1,976 records. The fixed split contains 1,581 training records and 395 validation records.

### 3.2 Review for additional rules

A second audit found no additional rule that could be defended as an impossible reading:

- There are no negative ring counts, ages, or target values.
- `rings_per_day = 40` occurs once, but a high ring count is possible and is not a sensor impossibility.
- `wifi_rssi_dbm = -88` is weak but plausible; it is not equivalent to the invalid zero sentinel.
- `age_months = 0` and `age_months = 36` are valid boundary values.
- `days_left` reaches 397.6 days. It is rare, but target values should not be deleted solely because they are high.
- IQR-based automatic deletion was rejected because it would classify plausible high ring counts and high battery-life values as outliers.

## 4. Visual analysis

The script produces three plots:

- `plots/histograms.png`
- `plots/relationships.png`
- `plots/linear_predictions.png`

### 4.1 Histograms

The battery-voltage histogram contains a main operating distribution between approximately 3.35 V and 4.20 V, together with an isolated group at 0 V. The temperature histogram contains normal readings between 8°C and 46°C, together with the isolated `-127` sentinel. The Wi-Fi histogram contains negative RSSI values and an isolated group at 0 dBm. These isolated groups support the three cleaning rules.

The target distribution is strongly right-skewed. The cleaned target median is approximately 80.2 days, while a smaller number of observations extend to nearly 400 days. This skew explains why a mean-only predictor is a weak reference model and why the linear model tends to regress extreme predictions toward the middle.

### 4.2 Feature relationships

The scatter plots show the following relationships:

- **Battery voltage:** strong positive association with `days_left`, but the relationship is curved rather than a single straight line.
- **Temperature:** a curved relationship. Moderate temperatures have higher average target values than the lowest and highest temperature groups. A simple correlation is close to zero because the curve changes direction.
- **Rings per day:** a generally negative and noisy relationship. More activity tends to reduce remaining battery life, but the feature is discrete and has substantial variation.
- **Wi-Fi RSSI:** a generally positive relationship. Stronger signal values, which are closer to zero, are associated with higher battery-life values, although the relationship begins to flatten and remains noisy.
- **Device age:** a generally negative relationship. Older devices tend to have fewer days remaining.

These patterns predict that a linear model will outperform the mean but will not fully capture the data structure.

## 5. Baseline and neural-network measurements

### 5.1 Baseline 1: training-set mean

**Prediction:** A constant mean prediction should provide a weak but valid reference point because it does not use any sensor information.

The baseline predicts the mean `days_left` from the training rows for every validation record. Its validation MAE was:

```text
68.096199 days
```

Rounded for reporting:

```text
68.10 days
```

### 5.2 Baseline 2: least-squares linear regression

**Prediction:** Linear regression should substantially outperform the mean because the features contain strong additive trends, especially battery voltage.

The model was fitted only on the training rows using an intercept:

```text
predicted_days_left = X beta + intercept
```

The coefficients were calculated with `np.linalg.lstsq`. The independent re-verification reproduced the exact Target 1 split and produced:

```text
28.671143 days
```

Rounded for reporting:

```text
28.67 days
```

The fitted coefficients were:

| Feature | Coefficient |
|---|---:|
| `battery_v` | `+249.6581` |
| `temp_c` | `-0.3884` |
| `rings_per_day` | `-3.8629` |
| `wifi_rssi_dbm` | `+1.8782` |
| `age_months` | `-0.9040` |
| Intercept | `-666.7577` |

The linear model predicted validation values between approximately `-70.27` and `270.34` days. Its unconstrained negative predictions are a limitation of this baseline rather than evidence that the input data contains negative battery life.

### 5.3 Neural-network benchmark

**Prediction:** The neural network should outperform linear regression because it can represent nonlinear relationships and interactions between the features.

The existing Target 1 run used the `5 → 32 → 32 → 1` network without normalization and recorded a validation MAE of:

```text
19.44 days
```

This is a benchmark for Target 1. Normalization, hidden-layer-size comparisons, and model-size analysis belong to Target 2.

## 6. Explanation of the performance gaps

The mean baseline has an MAE of 68.10 days because it assigns the same prediction to every reading. It cannot respond to battery voltage, usage, signal strength, or age.

Linear regression reduces the error to 28.67 days because it captures the dominant additive trends. Its error remains substantial because a straight-line equation cannot represent the curved battery-voltage and temperature relationships. It also cannot adapt its prediction to combinations of features whose joint effect differs from the sum of their individual effects.

The neural network reduces the error further to 19.44 days. The improvement of 9.23 days over linear regression is approximately 32% relative to the linear-regression MAE. This result is consistent with the visual evidence: the problem contains curvature, saturation, noise, and likely feature interactions.

The linear true-versus-predicted plot shows the expected regression-to-the-mean pattern. The model overpredicts many low and middle target values and underpredicts the highest target values. This explains why the linear model performs much better than the mean overall but remains less accurate at the extremes.

## Code block explanations

### Raw-data plots

```python
os.makedirs("plots", exist_ok=True)

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, column in zip(axes.ravel(), df.columns):
    ax.hist(df[column], bins=30, edgecolor="black")
    ax.set_title(f"Histogram: {column}")

plt.tight_layout()
plt.savefig("plots/histograms.png", dpi=150)
plt.close()
```

This block creates a directory for generated figures and places one histogram in each subplot. The loop uses the actual column names from the data frame, so the target and every sensor field are inspected consistently. `tight_layout()` prevents labels from overlapping, while `savefig()` makes the result available in a headless Codespace instead of relying on an interactive display. The plots are generated before cleaning so that the sentinel values remain visible for diagnosis.

The relationship plot follows the same pattern:

```python
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, feature in zip(axes.ravel(), FEATURES):
    ax.scatter(df[feature], df["days_left"], s=8, alpha=0.5)
    ax.set_title(f"days_left vs {feature}")

axes[-1, -1].axis("off")
plt.tight_layout()
plt.savefig("plots/relationships.png", dpi=150)
plt.close()
```

Here, the horizontal axis is one input feature and the vertical axis is the target. The scatter shape distinguishes a straight-line trend from curvature, saturation, clusters, or large unexplained spread. The unused sixth subplot is hidden because there are five input features.

### Sensor-glitch mask

```python
glitch_rules = {
    "battery_v <= 0": df["battery_v"] <= 0,
    "temp_c == -127": df["temp_c"] == -127,
    "wifi_rssi_dbm == 0": df["wifi_rssi_dbm"] == 0,
}

bad = np.zeros(len(df), dtype=bool)
for name, rule in glitch_rules.items():
    print(f"{name}: {int(rule.sum())} rows")
    bad |= rule.to_numpy()

df = df.loc[~bad].reset_index(drop=True)
```

Each dictionary entry is a vectorized boolean rule. The `bad` array combines the rules with logical OR, so a row is removed if any sensor contains a known invalid sentinel. The per-rule counts provide an auditable record of the cleaning decision. The rules are based on physical sensor meaning rather than statistical rarity, which avoids deleting rare but valid readings.

### Reproducible split

```python
rng = np.random.default_rng(0)
idx = rng.permutation(len(df))
n_val = len(df) // 5
val, tr = df.iloc[idx[:n_val]], df.iloc[idx[n_val:]]
```

The fixed seed makes the validation comparison reproducible. The first 20% of the shuffled clean rows are held out, and the remaining 80% are used for fitting. The split is performed before calculating the training mean or fitting linear-regression coefficients, preventing validation leakage.

### Mean baseline

```python
mean_prediction = np.full_like(yv, y.mean())
mean_mae = mae(mean_prediction, yv)
```

This is the simplest meaningful reference model. It predicts the mean target from the training data for every validation row. Using `y.mean()` rather than `yv.mean()` is essential because using the validation mean would allow evaluation data to influence the predictor.

### Least-squares linear regression

```python
A_train = np.column_stack([
    X,
    np.ones(len(X), dtype=np.float32),
])
A_val = np.column_stack([
    Xv,
    np.ones(len(Xv), dtype=np.float32),
])

coefficients, *_ = np.linalg.lstsq(A_train, y, rcond=None)
linear_prediction = A_val @ coefficients
linear_mae = mae(linear_prediction, yv)
```

The columns of `X` represent the five sensor features. The additional column of ones gives the model an intercept; without it, the fitted equation would be forced through zero. `np.linalg.lstsq` fits the coefficients using training data only. The fitted equation is then applied to the unseen validation features and evaluated with MAE.

### Neural-network benchmark

```python
torch.manual_seed(0)
net = BatteryNet(hidden)
opt = torch.optim.Adam(net.parameters(), lr=3e-3)

Xt = torch.tensor((X - mu) / sd)
yt = torch.tensor(y)
Xvt = torch.tensor((Xv - mu) / sd)
```

The seed makes initialization reproducible. The network is trained with Adam and mean-squared loss, while MAE is used for reporting so that the neural network can be compared with both baselines using the same metric. In Target 1, `mu` is zero and `sd` is one, so these expressions leave the raw inputs unchanged. Target 2 replaces those values with training-derived normalization statistics.

## 7. Limitations

The results are validation results from one deterministic 80/20 split, not a guarantee of performance on the company’s future 400 readings. The neural-network benchmark was trained without normalization, so Target 2 may improve its performance. The sensor rules are justified by the values present in this data and by physical interpretation; sensor documentation would provide stronger confirmation for deployment.

## 8. Formal conclusion

Target 1 establishes a reliable initial benchmark for the battery-life prediction task. The data contains 24 clear sensor-failure records and no additional defensible glitch categories. After cleaning, the mean baseline achieved 68.10 days MAE, least-squares linear regression achieved 28.67 days MAE, and the neural network achieved 19.44 days MAE. The results demonstrate that the sensor features contain meaningful predictive information and that a nonlinear model is more appropriate than a constant or purely additive linear predictor. Target 2 should therefore investigate normalization and architecture size while preserving the same cleaning rules, deterministic split, and validation procedure.
