import glob
import os

import numpy as np
import torch.utils.data


class FemurPairDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_root,
        subset,
        point_limit=None,
    ):
        super(FemurPairDataset, self).__init__()

        self.dataset_root = dataset_root
        self.subset = subset
        self.point_limit = point_limit

        self.data_root = os.path.join(self.dataset_root, self.subset)

        self.files = sorted(glob.glob(os.path.join(self.data_root, "*.npz")))

        if len(self.files) == 0:
            raise RuntimeError(f"No .npz files found in: {self.data_root}")

    def __len__(self):
        return len(self.files)

    def _limit_points(self, points):
        if self.point_limit is not None and points.shape[0] > self.point_limit:
            indices = np.random.permutation(points.shape[0])[: self.point_limit]
            points = points[indices]
        return points

    def __getitem__(self, index):
        data = np.load(self.files[index])

        src_points = data["source"].astype(np.float32)
        ref_points = data["target"].astype(np.float32)
        transform = data["T_gt"].astype(np.float32)

        src_points = self._limit_points(src_points)
        ref_points = self._limit_points(ref_points)

        data_dict = {}

        data_dict["scene_name"] = "femur"
        data_dict["ref_frame"] = int(data["sample_id"])
        data_dict["src_frame"] = int(data["sample_id"])
        data_dict["overlap"] = 0.0

        data_dict["ref_points"] = ref_points.astype(np.float32)
        data_dict["src_points"] = src_points.astype(np.float32)

        data_dict["ref_feats"] = np.ones((ref_points.shape[0], 1), dtype=np.float32)
        data_dict["src_feats"] = np.ones((src_points.shape[0], 1), dtype=np.float32)

        data_dict["transform"] = transform.astype(np.float32)

        return data_dict