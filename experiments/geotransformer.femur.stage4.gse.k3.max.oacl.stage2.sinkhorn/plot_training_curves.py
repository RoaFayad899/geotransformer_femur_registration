import re
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt


LOG_DIR = "../../output/exp_test_overfit2000_noaugmentation_lrconstant_2/logs"  #########################

log_files = glob.glob(os.path.join(LOG_DIR, "*.log"))
if not log_files:
    raise FileNotFoundError(f"No log files found in {LOG_DIR}")

log_file = sorted(log_files)[-1]
print("Reading:", log_file)

pattern = re.compile(r"\[(Train|Val)?\]?\s*Epoch:\s*(\d+).*")

rows = []

with open(log_file, "r") as f:
    for line in f:
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