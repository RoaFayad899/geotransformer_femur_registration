from geotransformer.datasets.registration.femur.dataset import FemurPairDataset
from geotransformer.utils.data import (
    registration_collate_fn_stack_mode,
    calibrate_neighbors_stack_mode,
    build_dataloader_stack_mode,
)


def train_valid_data_loader(cfg, distributed):
    train_dataset = FemurPairDataset(
        cfg.data.dataset_root,
        "train",
        point_limit=cfg.train.point_limit,
    )
    # Optional future experiment:
    # If you later modify FemurPairDataset to support augmentation,
    # you can use something like this:
    #
    # train_dataset = FemurPairDataset(
    #     cfg.data.dataset_root,
    #     "train",
    #     point_limit=cfg.train.point_limit,
    #     use_augmentation=cfg.train.use_augmentation,
    #     augmentation_noise=cfg.train.augmentation_noise,
    #     augmentation_rotation=cfg.train.augmentation_rotation,
    # )

    neighbor_limits = calibrate_neighbors_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
    )

    train_loader = build_dataloader_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        distributed=distributed,
    )

    valid_dataset = FemurPairDataset(
        cfg.data.dataset_root,
        "val",
        point_limit=cfg.test.point_limit,
    )

    valid_loader = build_dataloader_stack_mode(
        valid_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
        distributed=distributed,
    )

    return train_loader, valid_loader, neighbor_limits


def test_data_loader(cfg, benchmark):             # This train_dataset is only used to calibrate KPConv neighbor limits.
    train_dataset = FemurPairDataset(
        cfg.data.dataset_root,
        "train",
        point_limit=cfg.train.point_limit,
    )

    # Optional future experiment:
    # If FemurPairDataset is later extended with on-the-fly data augmentation,
    # you can use:
    #
    # train_dataset = FemurPairDataset(
    #     cfg.data.dataset_root,
    #     "train",
    #     point_limit=cfg.train.point_limit,
    #     use_augmentation=cfg.train.use_augmentation,
    #     augmentation_noise=cfg.train.augmentation_noise,
    #     augmentation_rotation=cfg.train.augmentation_rotation,
    # )

    neighbor_limits = calibrate_neighbors_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
    )

    test_dataset = FemurPairDataset(
        cfg.data.dataset_root,
        benchmark,
        point_limit=cfg.test.point_limit,
    )

    test_loader = build_dataloader_stack_mode(
        test_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
    )

    return test_loader, neighbor_limits