from geotransformer.datasets.registration.femur.dataset import FemurPairDataset

dataset_root = "data/Femur"

dataset = FemurPairDataset(
    dataset_root=dataset_root,
    subset="train",
    point_limit=None,
)

print("=" * 60)
print("Dataset size:", len(dataset))
print("=" * 60)

sample = dataset[0]

print("Keys:")
for key in sample.keys():
    print(" ", key)

print()

print("Reference points :", sample["ref_points"].shape)
print("Source points    :", sample["src_points"].shape)
print("Reference feats  :", sample["ref_feats"].shape)
print("Source feats     :", sample["src_feats"].shape)
print("Transform shape  :", sample["transform"].shape)

print()

print("Transform:")
print(sample["transform"])