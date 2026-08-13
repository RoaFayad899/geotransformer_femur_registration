import os
import pandas as pd

LOG_DIR = "../../output/exp_fulldataset_best4stages_2000_medium_00/logs"     #############################

csv_path = os.path.join(LOG_DIR, "parsed_metrics.csv")
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Missing: {csv_path}\nRun plot_training_curves.py first.")

df = pd.read_csv(csv_path)
val = df[df["split"] == "val"].copy()

best = val.loc[val["RMSE_mm"].idxmin()]

print("\n========== BEST VALIDATION EPOCH BY RMSE_mm ==========")
for key, value in best.items():
    print(f"{key}: {value}")

epoch = int(best["epoch"])
print("\nSuggested checkpoint:")
print(f"epoch-{epoch}.pth.tar")