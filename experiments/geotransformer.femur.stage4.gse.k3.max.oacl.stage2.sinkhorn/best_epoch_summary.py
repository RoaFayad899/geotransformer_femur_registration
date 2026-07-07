import os
import glob
import pandas as pd

LOG_DIR = "../../output/exp_test_overfit1_noaugmentation_lrconst_07/logs"  #######################

csv_path = os.path.join(LOG_DIR, "parsed_metrics.csv")
if not os.path.exists(csv_path):
    raise FileNotFoundError("Run plot_training_curves.py first.")

df = pd.read_csv(csv_path)

val = df[df["split"] == "val"].copy()

# choose best by smallest RMSE_mm
best = val.loc[val["RMSE_mm"].idxmin()]

print("\n========== BEST VALIDATION EPOCH BY RMSE_mm ==========")
for key, value in best.items():
    print(f"{key}: {value}")

print("\nSuggested checkpoint:")
epoch = int(best["epoch"])
print(f"epoch-{epoch}.pth.tar")