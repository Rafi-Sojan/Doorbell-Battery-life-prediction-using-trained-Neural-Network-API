# Target 2 Report: Normalization, Model Capacity, and Generalization

## Executive summary

Target 2 examined whether input normalization improves optimization, how model capacity affects validation accuracy, and whether the selected model is overfitting. The experiment used the cleaned team 24 data, the same deterministic train/validation split as Target 1, and the same `5 → hidden layers → 1` neural-network family.

Normalization produced the largest improvement. For the `5 → 32 → 32 → 1` architecture, the unnormalized model achieved a validation MAE of 19.44 days, while the normalized model achieved 5.87 days. The architecture sweep selected `5 → 32 → 32 → 1` with 1,281 trainable parameters, a training MAE of 4.97 days, a validation MAE of 5.87 days, and a generalization gap of 0.90 days.

The result supports normalization as an essential part of the training procedure. The selected model is substantially better than the Target 1 baselines, but the validation result should not be treated as an unbiased estimate of performance on the company’s future 400 readings because the validation set was also used to select the architecture.

## 1. Objective and evaluation method

Target 2 used the following prediction–measurement–explanation loop:

1. **Prediction:** normalized inputs should make optimization easier because the five features have very different numerical scales.
2. **Measurement:** compare training and validation MAE for raw and normalized inputs, then compare architectures with one and two hidden layers.
3. **Explanation:** relate changes in MAE and the train/validation gap to scale, model capacity, underfitting, and overfitting.

The data preparation and validation split were kept consistent with Target 1:

- Original rows: 2,000
- Glitch rows removed: 24
- Clean rows: 1,976
- Training rows: 1,581
- Validation rows: 395
- Random seed: 0
- Training epochs per experiment: 1,500
- Optimizer: Adam
- Learning rate: `0.003`
- Batch size: `128`

For normalized training, the mean and standard deviation were calculated from the training rows only:

```python
mu = X_train.mean(axis=0)
sd = X_train.std(axis=0)
X_train_norm = (X_train - mu) / sd
X_val_norm = (X_val - mu) / sd
```

The validation rows were transformed using the training statistics. This prevents validation information from leaking into training.

## 2. Role allocation

| Role | Responsibility in Target 2 |
|---|---|
| Builder | Added the normalization procedure, reusable training experiment, architecture sweep, generalization-gap calculation, and model selection. |
| Measurer | Ran the complete experiment and recorded MAE, parameter counts, and train/validation gaps. |
| Skeptic | Tested whether normalization caused a genuine improvement, examined model capacity, and identified validation-selection limitations. |
| Scribe | Recorded the method, results, interpretation, recommendations, and formal conclusion. |

## 3. Builder implementation

The implementation evaluates two input conditions for the selected architecture:

- Raw inputs, with no scaling
- Normalized inputs, using training-only statistics

It then evaluates ten normalized architectures:

- One hidden layer: `(4)`, `(8)`, `(16)`, `(32)`, `(64)`
- Two hidden layers: `(4, 4)`, `(8, 8)`, `(16, 16)`, `(32, 32)`, `(64, 64)`

For every architecture, the script records:

- Number of trainable parameters
- Training MAE
- Validation MAE
- Training-to-validation generalization gap

The final saved model includes the selected architecture and its normalization statistics in `model/config.json`. The API can therefore apply the same transformation at inference time.

## Code block explanations

### Training-only normalization

```python
if normalize:
    mu = X_train.mean(axis=0).astype(np.float32)
    sd = X_train.std(axis=0).astype(np.float32)
    sd = np.where(sd < 1e-8, 1.0, sd).astype(np.float32)
    X_train_used = (X_train - mu) / sd
    X_val_used = (X_val - mu) / sd
else:
    mu = np.zeros(X_train.shape[1], dtype=np.float32)
    sd = np.ones(X_train.shape[1], dtype=np.float32)
    X_train_used = X_train
    X_val_used = X_val
```

The `normalize` switch allows raw and normalized experiments to use the same model and optimizer. When normalization is enabled, `mu` and `sd` are calculated from `X_train` only. The validation matrix is transformed with those same values, which preserves the separation between fitting and evaluation. The small-standard-deviation guard prevents division by zero for a constant feature.

### Reusable experiment function

```python
with torch.no_grad():
    train_prediction = model(Xt).numpy()
    val_prediction = model(Xvt).numpy()

return {
    "model": model,
    "mu": mu,
    "sd": sd,
    "train_mae": mae(train_prediction, y_train),
    "val_mae": mae(val_prediction, y_val),
    "parameters": sum(
        parameter.numel()
        for parameter in model.parameters()
    ),
}
```

The experiment function returns both predictions and metadata so that the same training procedure can be reused for normalization comparisons and architecture sweeps. Training and validation MAE are calculated after optimization. Parameter count is computed from the actual PyTorch parameters rather than typed manually, reducing the risk of reporting an incorrect model size.

### Architecture sweep

```python
architectures = (
    [(width,) for width in [4, 8, 16, 32, 64]]
    + [(width, width) for width in [4, 8, 16, 32, 64]]
)

for architecture in architectures:
    result = train_network_experiment(
        X, y, Xv, yv,
        architecture,
        a.epochs,
        normalize=True,
    )
```

The first list represents one hidden layer and the second represents two equal-width hidden layers. The sweep keeps the data split, number of epochs, optimizer, and normalization policy constant so that the architecture is the main experimental variable. The resulting table supports a comparison between model capacity and validation error.

### Generalization gap

```python
generalization_gap = result["val_mae"] - result["train_mae"]
```

The generalization gap measures how much worse the model performs on held-out data than on training data. A large positive gap is evidence of overfitting. A small gap with high error suggests underfitting or insufficient model capacity. A negative gap can occur because the validation split is easier than the training split and is not automatically evidence of a problem.

### Persisting normalization statistics

```python
json.dump(
    {
        "hidden": list(result["hidden"]),
        "mu": result["mu"].tolist(),
        "sd": result["sd"].tolist(),
        "val_mae": round(result["val_mae"], 2),
    },
    open("model/config.json", "w"),
    indent=2,
)
```

The model weights alone are not sufficient for inference after normalization. The API must use exactly the same feature means and standard deviations as training. Storing `mu` and `sd` in `config.json` makes the preprocessing part of the saved model configuration rather than an undocumented manual step.

### Model information calculation

```python
for module in net.net:
    if not isinstance(module, torch.nn.Linear):
        continue

    n_in = module.in_features
    n_out = module.out_features
    params = n_in * n_out + n_out
    macs = n_in * n_out
```

The API inspects the actual `Linear` modules and ignores ReLU layers because ReLU has no trainable weights. A linear layer has `in × out` weights and `out` biases. The same dimensions determine the number of multiply-accumulate operations. Calculating these values from the network keeps `/model_info` correct if the architecture changes.

## 4. Measurer: normalization comparison

The `5 → 32 → 32 → 1` model was trained with and without normalization.

| Input treatment | Training MAE | Validation MAE |
|---|---:|---:|
| Raw inputs | 19.07 days | 19.44 days |
| Normalized inputs | 4.97 days | 5.87 days |

Normalization reduced validation MAE by:

```text
19.44 - 5.87 = 13.57 days
```

This is approximately a 69.8% reduction relative to the unnormalized validation error.

The improvement is expected because the input features have very different scales and distributions. Battery voltage is measured near 3–4 volts, temperature spans tens of degrees, ring counts are integer activity values, RSSI is a negative signal-strength value, and age spans months. Without normalization, gradient-based optimization receives features with unequal numerical influence. Normalization places the inputs on comparable scales and allows the optimizer to make more balanced updates.

## 5. Measurer: architecture sweep

All sweep models used normalized inputs.

| Hidden layers | Parameters | Training MAE | Validation MAE | Generalization gap |
|---|---:|---:|---:|---:|
| `(4)` | 29 | 15.64 | 15.63 | -0.01 |
| `(8)` | 57 | 20.74 | 20.37 | -0.37 |
| `(16)` | 113 | 20.74 | 20.33 | -0.41 |
| `(32)` | 225 | 13.08 | 13.29 | 0.21 |
| `(64)` | 449 | 20.57 | 20.28 | -0.29 |
| `(4, 4)` | 49 | 11.35 | 11.27 | -0.08 |
| `(8, 8)` | 129 | 7.07 | 7.86 | 0.79 |
| `(16, 16)` | 385 | 5.88 | 6.49 | 0.61 |
| `(32, 32)` | 1,281 | 4.97 | 5.87 | 0.90 |
| `(64, 64)` | 4,609 | 4.41 | 6.03 | 1.62 |

The parameter count includes weights and biases. For the selected architecture:

```text
5 → 32:    5×32 + 32 = 192 parameters
32 → 32:  32×32 + 32 = 1,056 parameters
32 → 1:   32×1 + 1 = 33 parameters
Total:    1,281 parameters
```

The corresponding float32 storage is `1,281 × 4 = 5,124 bytes`, and the total multiply-accumulate count is:

```text
5×32 + 32×32 + 32×1 = 1,216 MACs
```

## 6. Measurer: selected model and overfitting

The sweep selected `(32, 32)` because it achieved the lowest validation MAE:

```text
Training MAE:       4.97 days
Validation MAE:     5.87 days
Generalization gap: 0.90 days
```

The gap is small relative to the overall error. The model is fitting the training data better than the validation data, as expected, but the difference is not large enough to indicate severe overfitting in this split.

The `(64, 64)` model is a useful comparison. It achieved a lower training MAE of 4.41 days but a slightly worse validation MAE of 6.03 days and a larger 1.62-day gap. This is evidence that additional capacity begins to improve training fit faster than generalization.

The smaller `(4, 4)` and `(16, 16)` models generalize reasonably but have higher validation error than `(32, 32)`. They are less expressive and therefore underfit some of the nonlinear structure.

## 7. Skeptic: critical review

### 7.1 The normalization result is strong but not sufficient by itself

The improvement from 19.44 to 5.87 days is large and consistent with the feature-scale problem. However, the comparison is based on one fixed validation split and one training configuration. A second seed or cross-validation would test whether the improvement is stable rather than specific to this split.

### 7.2 Architecture selection introduces validation-set optimism

The same validation set was used to compare ten architectures and select the best one. Therefore, the selected 5.87-day result is a model-selection result, not an unbiased final test estimate. Repeatedly selecting the lowest validation error can make the selected score look better than performance on genuinely unseen data.

The company’s 400 unseen readings should be treated as a final test set. They must not be used to adjust the architecture, normalization process, or training settings before the final evaluation.

### 7.3 The sweep is not perfectly monotonic

Validation error does not decrease smoothly as parameter count increases. For example, the one-layer `(8)` and `(16)` models perform similarly, while `(32)` is much better. This is normal for neural networks: optimization and representation quality depend on initialization, activation patterns, and architecture, not only on parameter count.

The results should therefore be interpreted as measurements of these particular architectures under the specified training procedure, not as a universal law that larger models are always better.

### 7.4 The selected model is a reasonable compromise

The `(32, 32)` model is preferable to `(64, 64)` because it has:

- Lower validation MAE: 5.87 versus 6.03 days
- Smaller generalization gap: 0.90 versus 1.62 days
- Far fewer parameters: 1,281 versus 4,609
- Lower memory and computation cost

The selection is therefore supported by both predictive accuracy and resource efficiency.

## 8. Scribe: prediction, measurement, and explanation

### Prediction

Normalization was expected to improve training because the input features use different scales. A moderate two-layer network was expected to outperform very small networks, while an excessively large network could begin to overfit.

### Measurement

The prediction was confirmed. Normalization reduced the selected model’s validation MAE from 19.44 to 5.87 days. The architecture sweep selected `(32, 32)` with 1,281 parameters, 4.97 days training MAE, and 5.87 days validation MAE.

### Explanation

The raw inputs produced poorly balanced optimization because the features have different scales. Normalization made the optimization problem better conditioned, allowing the network to fit the underlying relationships more effectively. Increasing capacity from very small networks reduced underfitting, but the `(64, 64)` model gained more on the training set than on validation data. The `(32, 32)` model provided the best measured validation result with a smaller generalization gap and lower resource cost than the largest model.

## 9. Recommendations

1. Keep training-only normalization in the final training and API inference path.
2. Use the `5 → 32 → 32 → 1` architecture as the current candidate model.
3. Do not claim that 5.87 days is final unseen-data performance until the 400 company readings are evaluated once.
4. Before deployment, repeat the experiment with multiple random seeds or cross-validation to test stability.
5. Preserve the validation split for comparison, but use a separate final test set for the final performance claim.
6. Continue to Target 3 only after confirming that the API applies the saved `mu` and `sd` values exactly as training did.

## 10. Skeptic follow-up audit

The robustness recommendation was tested by training the selected normalized `(32, 32)` architecture five times with seeds `0` through `4`, while keeping the cleaned data, validation split, optimizer, learning rate, batch size, and epoch count fixed.

| Seed | Training MAE | Validation MAE | Generalization gap |
|---:|---:|---:|---:|
| 0 | 4.9581 | 5.8554 | 0.8973 |
| 1 | 5.1834 | 6.1809 | 0.9975 |
| 2 | 5.0881 | 6.5096 | 1.4215 |
| 3 | 4.6977 | 6.1770 | 1.4794 |
| 4 | 4.9198 | 6.2990 | 1.3793 |
| **Mean** | — | **6.2044** | **1.2350** |
| **Standard deviation** | — | **0.2373** | — |

The validation MAE varied by approximately `0.65 days` across the five seeds. This is a relatively small spread compared with the 13.57-day improvement from normalization, so the normalization conclusion is robust to initialization. The original selected run, with seed `0`, remains the best of these five runs at 5.8554 days, reported in the saved configuration as 5.87 days after rounding.

The saved preprocessing values were also checked against a fresh calculation from the training rows. Both the stored `mu` and stored `sd` matched exactly within floating-point tolerance. The saved architecture remained `[32, 32]`, and the saved validation MAE remained `5.87` days.

### 10.1 Four-hundred-reading internal holdout

Because the repository does not contain the company’s future 400 readings, an internal holdout was created for a separate robustness check. The 1,976 clean rows were randomly divided with seed `2026` into 1,576 development rows and exactly 400 holdout rows. The holdout was not used during the cross-validation runs.

The selected normalized `(32, 32)` model was trained on the 1,576 development rows five times:

| Seed | Development training MAE | 400-reading holdout MAE |
|---:|---:|---:|
| 0 | 5.1167 | 6.0199 |
| 1 | 4.8839 | 5.6743 |
| 2 | 4.7836 | 5.5748 |
| 3 | 4.8406 | 6.3835 |
| 4 | 5.1049 | 6.0753 |
| **Mean** | — | **5.9456** |
| **Standard deviation** | — | **0.3260** |

The holdout MAE ranged from 5.57 to 6.38 days. This is close to the original validation result of 5.87 days and indicates that the candidate model generalizes consistently across random initializations. This is an internal holdout drawn from the existing training data, not the company’s genuinely unseen 400-reading test set.

### 10.2 Five-fold cross-validation

Five-fold cross-validation was performed on the 1,576 development rows. Each fold used approximately 80% of the development data for training and 20% for validation. Three independent seeds were evaluated in each fold, producing 15 validation measurements in total.

| Fold | Mean validation MAE | Standard deviation |
|---:|---:|---:|
| 1 | 6.4864 | 0.4169 |
| 2 | 7.2558 | 0.0659 |
| 3 | 6.2876 | 0.2525 |
| 4 | 6.0780 | 0.2778 |
| 5 | 6.5046 | 0.1870 |
| **Overall** | **6.5225** | **0.4694** |

The cross-validation values range from 5.81 to 7.31 days. The cross-validation mean is higher than the 400-reading holdout mean because each cross-validation model is trained on only about 80% of the development data, rather than on all 1,576 development rows. The difference is therefore evidence of sampling and training-size sensitivity, not necessarily a contradiction.

The combined evidence supports an expected error of approximately 6–6.5 days under these data conditions. The 5.87-day result remains the best single validation measurement, while the holdout and cross-validation results provide a more cautious estimate of typical performance.

This audit strengthens the recommendation to retain normalization and the `(32, 32)` candidate. It does not remove the model-selection limitation: the original validation set was used to compare architectures. The internal 400-reading holdout and cross-validation are stronger evidence of stability, but the company’s separate 400-reading final test set is still required for the deployment claim.

## 11. Formal conclusion

Target 2 demonstrates that normalization is essential for this problem and that model capacity must be selected using both accuracy and generalization behavior. The normalized `5 → 32 → 32 → 1` network achieved the lowest observed validation MAE of 5.87 days among the tested configurations. Its 0.90-day generalization gap is substantially smaller than the gap of the larger `(64, 64)` model, making it the most defensible current candidate. Five-seed evaluation on an internal 400-reading holdout produced a mean MAE of 5.95 days, while five-fold cross-validation on the development data produced a mean MAE of 6.52 days. These results support a cautious expected range of approximately 6–6.5 days, but a separate company test set is still required before presenting the model’s performance as a deployment guarantee.
