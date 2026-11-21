"""
train.py -- STARTER. Clean the data, fit baselines, train the network, save it for the server.

    python train.py --team 7 --hidden 32,32
Writes model/model.pt and model/config.json, which app/main.py loads.

Look for TODO. The training loop is given; the decisions are yours.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from app.model import FEATURES, BatteryNet

ap = argparse.ArgumentParser()
ap.add_argument("--team", type=int, required=True)
ap.add_argument("--hidden", default="32,32")
ap.add_argument("--epochs", type=int, default=1500)
a = ap.parse_args()
hidden = tuple(int(h) for h in a.hidden.split(","))

df = pd.read_csv(f"data/team_{a.team:02d}_train.csv")
# TODO Mission 1: find the sensor glitches (plot every column first!) and write a rule that
# marks them. Say in your report how many you found and why each rule is safe.
# Plots every column before cleaning.

#plotting against the input features to check the relationships 
os.makedirs("plots", exist_ok=True)

#plots a histogram against every column
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, column in zip(axes.ravel(), df.columns):
    ax.hist(df[column], bins=30, edgecolor="black")
    ax.set_title(f"Histogram: {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("count")
    ax.set_ylabel("Number of Rows")

plt.tight_layout()
plt.savefig("plots/histograms.png", dpi=150)
plt.close()

# Plots days left against every input feature.
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, feature in zip(axes.ravel(), FEATURES):
    ax.scatter(df[feature], df["days_left"], s=8, alpha=0.5)
    ax.set_title(f"days_left vs {feature}")
    ax.set_xlabel(feature)
    ax.set_ylabel("days_left")

axes[-1, -1].axis("off")
plt.tight_layout()
plt.savefig("plots/relationships.png", dpi=150)
plt.close()

# Sensor-glitch rules.
glitch_rules = {

    #battery voltage of zero is not valid
    "battery_v <= 0": df["battery_v"] <= 0,
    #room temperature does not go up to -127 
    "temp_c == -127": df["temp_c"] == -127,
    #should be negative, if 0 the value is missing
    "wifi_rssi_dbm == 0": df["wifi_rssi_dbm"] == 0,
}

#start with no rows marked as bad
bad = np.zeros(len(df), dtype=bool)

#print the number of rows removed by each individual rule
for name, rule in glitch_rules.items():
    print(f"{name}: {int(rule.sum())} rows")
    bad |= rule.to_numpy()

print(f"{len(df)} rows, {int(bad.sum())} glitch rows dropped")

#remove row if satisifes one of the glitch rules set
df = df.loc[~bad].reset_index(drop=True)

#reproducing the mae values
rng = np.random.default_rng(0)
#shuffling indices before splitting
idx = rng.permutation(len(df)); n_val = len(df) // 5
val, tr = df.iloc[idx[:n_val]], df.iloc[idx[n_val:]]
X, y = tr[FEATURES].to_numpy(np.float32), tr.days_left.to_numpy(np.float32)
Xv, yv = val[FEATURES].to_numpy(np.float32), val.days_left.to_numpy(np.float32)
# TODO Mission 2: normalisation. Why must mu and sd come from the TRAINING rows only?
mu, sd = np.zeros(X.shape[1], np.float32), np.ones(X.shape[1], np.float32)

# mean absolute error calculation p = prediction and t = truth
mae = lambda p, t: float(np.mean(np.abs(p - t)))


def train_network_experiment(X_train, y_train, X_val, y_val, hidden, epochs, normalize):
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

    torch.manual_seed(0)
    model = BatteryNet(hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    Xt = torch.tensor(X_train_used, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32)
    Xvt = torch.tensor(X_val_used, dtype=torch.float32)

    for epoch in range(epochs):
        permutation = torch.randperm(len(Xt))
        for start in range(0, len(Xt), 128):
            batch = permutation[start:start + 128]
            loss = torch.nn.functional.mse_loss(model(Xt[batch]), yt[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        train_prediction = model(Xt).numpy()
        val_prediction = model(Xvt).numpy()

    return {
        "model": model,
        "mu": mu,
        "sd": sd,
        "train_mae": mae(train_prediction, y_train),
        "val_mae": mae(val_prediction, y_val),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


def save_model_result(result):
    os.makedirs("model", exist_ok=True)
    torch.save(result["model"].state_dict(), "model/model.pt")
    json.dump(
        dict(
            team=a.team,
            hidden=list(result["hidden"]),
            mu=result["mu"].tolist(),
            sd=result["sd"].tolist(),
            features=FEATURES,
            val_mae=round(result["val_mae"], 2),
            train_mae=round(result["train_mae"], 2),
        ),
        open("model/config.json", "w"),
        indent=2,
    )


def run_model_experiments(X, y, Xv, yv):
    chosen_hidden = hidden

    raw_result = train_network_experiment(
        X, y, Xv, yv, chosen_hidden, a.epochs, normalize=False
    )
    normalized_result = train_network_experiment(
        X, y, Xv, yv, chosen_hidden, a.epochs, normalize=True
    )

    print("\nNormalization comparison")
    print(
        f"raw inputs:        train MAE={raw_result['train_mae']:.2f}, "
        f"val MAE={raw_result['val_mae']:.2f}"
    )
    print(
        f"normalized inputs: train MAE={normalized_result['train_mae']:.2f}, "
        f"val MAE={normalized_result['val_mae']:.2f}"
    )

    architectures = (
        [(width,) for width in [4, 8, 16, 32, 64]]
        + [(width, width) for width in [4, 8, 16, 32, 64]]
    )
    sweep_results = []

    for architecture in architectures:
        result = train_network_experiment(
            X, y, Xv, yv, architecture, a.epochs, normalize=True
        )
        result["hidden"] = architecture
        sweep_results.append(result)
        print(
            f"hidden={architecture!s:10s} "
            f"parameters={result['parameters']:5d} "
            f"train MAE={result['train_mae']:6.2f} "
            f"val MAE={result['val_mae']:6.2f}"
        )

    plt.figure(figsize=(9, 6))
    for result in sweep_results:
        label = str(result["hidden"])
        plt.scatter(result["parameters"], result["val_mae"], s=70)
        plt.annotate(
            label,
            (result["parameters"], result["val_mae"]),
            xytext=(5, 5),
            textcoords="offset points",
        )
    plt.xlabel("Number of trainable parameters")
    plt.ylabel("Validation MAE (days)")
    plt.title("Model size versus validation error")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/mae_vs_parameters.png", dpi=150)
    plt.close()

    best = min(
        sweep_results,
        key=lambda result: (result["val_mae"], result["parameters"]),
    )
    print("\nSelected model")
    print(f"architecture: {best['hidden']}")
    print(f"parameters: {best['parameters']}")
    print(f"training MAE: {best['train_mae']:.2f} days")
    print(f"validation MAE: {best['val_mae']:.2f} days")
    print(
        f"generalization gap: "
        f"{best['val_mae'] - best['train_mae']:.2f} days"
    )

    save_model_result(best)
    print("saved normalized model/model.pt and model/config.json")

# Baseline 1: always predict the training-set mean.
mean_prediction = np.full_like(yv, y.mean())
mean_mae = mae(mean_prediction, yv)

print(f"baseline (predict the mean): {mean_mae:.2f} days")

# Baseline 2: linear regression using np.linalg.lstsq.
# Adds a column of ones so the linear model can learn an intercept.
# Without this column, the fitted line would be forced through zero.
A_train = np.column_stack([X, np.ones(len(X), dtype=np.float32)])
A_val = np.column_stack([Xv, np.ones(len(Xv), dtype=np.float32)])

# Fit the linear model using only the training rows.
# np.linalg.lstsq finds the coefficients that minimize squared error.
coefficients, *_ = np.linalg.lstsq(A_train, y, rcond=None)

linear_prediction = A_val @ coefficients
linear_mae = mae(linear_prediction, yv)

print(f"baseline (linear regression): {linear_mae:.2f} days")

# Plots predicted versus true validation values.
plt.figure(figsize=(6, 6))
plt.scatter(yv, linear_prediction, s=15, alpha=0.6)

low = min(yv.min(), linear_prediction.min())
high = max(yv.max(), linear_prediction.max())

plt.plot([low, high], [low, high], "r--", label="perfect prediction")
plt.xlabel("True days_left")
plt.ylabel("Predicted days_left")
plt.title("Linear regression: validation predictions")
plt.legend()
plt.tight_layout()
plt.savefig("plots/linear_predictions.png", dpi=150)
plt.close()
# TODO Mission 1: a straight-line model (linear regression) as the second baseline.
# Use np.linalg.lstsq on [X, 1]. Print its validation MAE the same way as the line above.

torch.manual_seed(0)
net = BatteryNet(hidden)
opt = torch.optim.Adam(net.parameters(), lr=3e-3)
Xt, yt = torch.tensor((X - mu) / sd), torch.tensor(y)
Xvt = torch.tensor((Xv - mu) / sd)
for ep in range(a.epochs):
    perm = torch.randperm(len(Xt))
    for i in range(0, len(Xt), 128):
        b = perm[i:i + 128]
        loss = torch.nn.functional.mse_loss(net(Xt[b]), yt[b])
        opt.zero_grad(); loss.backward(); opt.step()
    if ep % 300 == 0 or ep == a.epochs - 1:
        with torch.no_grad():
            print(f"  epoch {ep:4d}  val MAE {mae(net(Xvt).numpy(), yv):.2f} days")
with torch.no_grad():
    val_mae = mae(net(Xvt).numpy(), yv)
print(f"network {hidden}:            {val_mae:.2f} days")

os.makedirs("model", exist_ok=True)
torch.save(net.state_dict(), "model/model.pt")
json.dump(dict(team=a.team, hidden=list(hidden), mu=mu.tolist(), sd=sd.tolist(),
               features=FEATURES, val_mae=round(val_mae, 2)), open("model/config.json", "w"), indent=2)
print("saved model/model.pt and model/config.json")

run_model_experiments(X, y, Xv, yv)
