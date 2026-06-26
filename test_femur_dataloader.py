import importlib.util
import os

exp_dir = "experiments/geotransformer.femur.stage4.gse.k3.max.oacl.stage2.sinkhorn"

spec = importlib.util.spec_from_file_location("config", os.path.join(exp_dir, "config.py"))
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

spec = importlib.util.spec_from_file_location("dataset", os.path.join(exp_dir, "dataset.py"))
dataset_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataset_mod)

cfg = config.make_cfg()

train_loader, val_loader, neighbor_limits = dataset_mod.train_valid_data_loader(cfg, distributed=False)

batch = next(iter(train_loader))

print("Neighbor limits:", neighbor_limits)
print("Batch keys:", batch.keys())
print("features:", batch["features"].shape)
print("transform:", batch["transform"].shape)
print("Number of point stages:", len(batch["points"]))

for i, points in enumerate(batch["points"]):
    print(f"points[{i}]:", points.shape)

for i, lengths in enumerate(batch["lengths"]):
    print(f"lengths[{i}]:", lengths)