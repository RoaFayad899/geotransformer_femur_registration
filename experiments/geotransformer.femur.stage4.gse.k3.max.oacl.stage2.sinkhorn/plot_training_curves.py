import re
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt


LOG_FILE = "../../output/exp_fulldataset_best4stages_2000_small_01/logs/train-20260811-083852.log"  ##########################

LOG_DIR = os.path.dirname(LOG_FILE)

log_file = LOG_FILE
print("Reading:", log_file)


pattern = re.compile(r"\[(Train|Val)?\]?\s*Epoch:\s*(\d+).*")

rows = []

with open(log_file, "r") as f:
    for line in f:
        if "[CRIT]" not in line:
            continue
        if "Epoch:" not in line or "loss:" not in line:
            continue

        split = "val" if "[Val]" in line else "train"

        epoch_match = re.search(r"Epoch:\s*(\d+)", line)
        if not epoch_match:
            continue

        row = {"split": split, "epoch": int(epoch_match.group(1))}

        metrics = [
            "loss", "c_loss", "f_loss",
            "PIR", "IR",
            "RRE", "RTE", "RTE_mm",
            "RMSE", "RMSE_mm", "RR",
            "IR@18mm", "IR@12mm", "IR@6mm", "IR@3mm",
        ]

        for m in metrics:
            match = re.search(rf"{re.escape(m)}:\s*([0-9.eE+-]+)", line)
            if match:
                row[m] = float(match.group(1))

        rows.append(row)

df = pd.DataFrame(rows)
print(df.tail())

out_csv = os.path.join(LOG_DIR, "parsed_metrics.csv")
df.to_csv(out_csv, index=False)
print("Saved:", out_csv)

# ==========================================================
# Combined plots
# ==========================================================
def plot_group(metrics, title, filename):
    plt.figure(figsize=(10, 6))

    for metric in metrics:
        if metric not in df.columns:
            continue

        for split in ["train", "val"]:
            d = df[df["split"] == split]
            if len(d) > 0:
                plt.plot(d["epoch"], d[metric], label=f"{split}/{metric}")

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out_png = os.path.join(LOG_DIR, filename)
    plt.savefig(out_png, dpi=200)
    print("Saved:", out_png)


plot_group(
    ["loss", "c_loss", "f_loss"],
    "Training and validation losses",
    "combined_losses.png",
)

plot_group(
    ["RRE", "RTE_mm", "RMSE_mm"],
    "Registration errors",
    "combined_registration_errors.png",
)

plot_group(
    ["IR@3mm", "IR@6mm", "IR@12mm", "IR@18mm"],
    "Inlier ratio at different thresholds",
    "combined_IR_thresholds.png",
)


# ==========================================================
# Individual plots (your original code)
# ==========================================================
for metric in ["loss", "c_loss", "f_loss", "RRE", "RTE_mm", "RMSE_mm", "RR", "IR@3mm", "IR@6mm", "IR@12mm", "IR@18mm"]:
    if metric not in df.columns:
        continue

    plt.figure()
    for split in ["train", "val"]:
        d = df[df["split"] == split]
        if len(d) > 0:
            plt.plot(d["epoch"], d[metric], label=split)

    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.title(metric)
    plt.legend()
    plt.grid(True)

    out_png = os.path.join(LOG_DIR, f"{metric.replace('@','at').replace('/','_')}.png")
    plt.savefig(out_png, dpi=200)
    print("Saved:", out_png)